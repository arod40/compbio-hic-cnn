import json
from collections import defaultdict
from itertools import combinations, product

import numpy as np
import pandas as pd
import torch
from torch.utils.data import ConcatDataset, random_split


# This set of methods allow to create the list of positive pairs from the list of bio replicates and cell types desired
# A positive pair is created only for matching cell types and matching bio replicates
# A negative pair is created for matching cell types and non-matching bio replicates
# The method creates ALL possible pairs in either case
def create_bio_replicate_to_experiment_dict(
    experiment_to_cell_type,
    experiment_to_bio_replicate,
    cell_type,
    use_experiments=None,
):
    bio_replicate_to_experiment = defaultdict(list)
    for exp in experiment_to_cell_type:
        if use_experiments is not None and exp not in use_experiments:
            continue

        if experiment_to_cell_type[exp] == cell_type:
            bio_replicate_to_experiment[experiment_to_bio_replicate[exp]].append(exp)
    return bio_replicate_to_experiment


def get_positive_pairs_from_list_of_bio_replicates(
    experiment_to_cell_type,
    experiment_to_bio_replicate,
    cell_types,
    bio_replicates,
    use_experiments=None,
):
    positive_pairs = []
    for cell_type in cell_types:
        bio_replicate_to_experiment = create_bio_replicate_to_experiment_dict(
            experiment_to_cell_type,
            experiment_to_bio_replicate,
            cell_type,
            use_experiments=use_experiments,
        )

        for bio_replicate in bio_replicates:
            for exp1, exp2 in combinations(
                bio_replicate_to_experiment[bio_replicate], 2
            ):
                positive_pairs.append((exp1, exp2))

    return positive_pairs


def get_negative_pairs_from_list_of_bio_replicates(
    experiment_to_cell_type,
    experiment_to_bio_replicate,
    cell_types,
    bio_replicates,
    use_experiments=None,
):
    negative_pairs = []
    for cell_type in cell_types:
        bio_replicate_to_experiment = create_bio_replicate_to_experiment_dict(
            experiment_to_cell_type,
            experiment_to_bio_replicate,
            cell_type,
            use_experiments=use_experiments,
        )

        for i, j in combinations(bio_replicates, 2):
            exps1 = bio_replicate_to_experiment[i]
            exps2 = bio_replicate_to_experiment[j]
            for exp1, exp2 in product(exps1, exps2):
                negative_pairs.append((exp1, exp2))
    return negative_pairs


def parse_mmc2_file(mmc2_file_path: str):
    exp_prefix = "GSM1551607_"
    data = read_csv(mm2_file_path)
    data["Library"] = data["Library"].apply(lambda x: exp_prefix + x)
    experiment_to_cell_type = dict(zip(data["Library"], data["Cell type"]))
    experiment_to_bio_replicate = dict(
        zip(data["Library"], data["Biological Replicate number"].astype(int))
    )
    return experiment_to_cell_type, experiment_to_bio_replicate


def train_once(model, train_loader, criterion, optimizer, device):
    print("Training model...")
    model.train()

    running_loss = 0.0
    for i, batch in enumerate(train_loader):
        optimizer.zero_grad()
        input1, input2, label = batch["input1"], batch["input2"], batch["label"]
        output = model(input1.to(device), input2.to(device))
        loss = criterion(output, label.to(device))
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        print(
            f"Batch: {i + 1}/{len(train_loader)}, Loss: {running_loss / (i + 1)}",
            end="\r",
        )

    return running_loss / len(train_loader)


def eval_once(model, test_data, criteria, device):
    print("Evaluating model...")
    model.eval()

    metrics = {name: 0.0 for name, _ in criteria}
    with torch.no_grad():
        for i, batch in enumerate(test_data):
            input1, input2, label = batch["input1"], batch["input2"], batch["label"]
            output = model(input1.to(device), input2.to(device))

            for name, criterion in criteria:
                metric_value = criterion(
                    output.squeeze(1).cpu(), label.squeeze(1).cpu()
                )
                metrics[name] += metric_value
            print(f"Batch: {i + 1}/{len(test_data)}", end="\r")

    return {
        name: metric_value / len(test_data) for name, metric_value in metrics.items()
    }


def train_loop(
    model,
    train_loader,
    eval_loader,
    test_loader,
    batch_size,
    num_epochs,
    criterion,
    optimizer,
    eval_metrics,
    save_dir,
    device,
):
    # Training loop
    history = {
        "train_loss": [],
        "test_loss": [],
        "test_accuracy": [],
    }

    json.dump(
        {
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "criterion": str(criterion),
            "optimizer": str(optimizer),
            "model": str(type(model)),
        },
        open(f"{save_dir}/params.json", "w"),
        indent=4,
    )

    for epoch in range(num_epochs):
        print(f"Epoch: {epoch + 1}")

        history["train_loss"].append(
            train_once(model, train_loader, criterion, optimizer, device)
        )
        epoch_val_metrics = eval_once(model, eval_loader, eval_metrics, device)
        for name, value in epoch_val_metrics.items():
            history[f"test_{name}"].append(value)

        checkpoint_path = f"{save_dir}/epoch_{epoch + 1}.pt"
        torch.save(model.state_dict(), checkpoint_path)

        print(
            f"Train loss: {history['train_loss'][-1]:.4f}, Test loss: {epoch_val_metrics['loss']:.4f} Test accuracy: {epoch_val_metrics['accuracy']:.4f}"
        )
    torch.save(model.state_dict(), f"{save_dir}/final.pt")
    json.dump(history, open(f"{save_dir}/history.json", "w"), indent=4)

    best_checkpoint = save_dir / get_best_checkpoint(history)
    model.load_state_dict(torch.load(best_checkpoint))
    test_metrics = eval_once(model, test_loader, eval_metrics, device)
    json.dump(test_metrics, open(save_dir / "test_metrics.json", "w"), indent=4)


def get_whole_dataset_split(all_cells_dataset, repro_seed=None):
    train_size, val_size = int(0.6 * len(all_cells_dataset)), int(
        0.2 * len(all_cells_dataset)
    )
    test_size = len(all_cells_dataset) - train_size - val_size

    generator = (
        torch.Generator().manual_seed(repro_seed) if repro_seed is not None else None
    )
    return random_split(
        all_cells_dataset, [train_size, val_size, test_size], generator=generator
    )


def get_dataset_split_by_cell_type(datasets_by_cell_type, train_cell_types, test_cell_types, repro_seed=None):
    train_dataset = ConcatDataset(
        [datasets_by_cell_type[cell_type] for cell_type in train_cell_types]
    )
    test_dataset = ConcatDataset(
        [datasets_by_cell_type[cell_type] for cell_type in test_cell_types]
    )

    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size

    generator = (
        torch.Generator().manual_seed(repro_seed) if repro_seed is not None else None
    )
    train_dataset, val_dataset = random_split(
        train_dataset, [train_size, val_size], generator=generator
    )
    return train_dataset, val_dataset, test_dataset


def get_best_checkpoint(history):
    epoch = np.argmax(history["test_accuracy"])
    return f"epoch_{epoch + 1}.pt"


def dump_analytics_to_df(dataset, models, experiment_to_cell_type):
    loaded_models = []
    for _, model_exp, SiameseNetworkXEncoder in models:
        exp = Path(model_exp)
        best_checkpoint = get_best_checkpoint(json.load(open(exp / "history.json")))
        model = SiameseNetworkXEncoder((40, 40)).to(device)
        model.load_state_dict(torch.load(exp / best_checkpoint))
        model.eval()
        model.to(device)
        loaded_models.append(model)

    columns = ["cell_type", "chr", "low", "high", "label"] + [
        model_name for model_name, _, _ in models
    ]


# gm12878_only_dataset_train = HICPairsBioReplicatesDataset(
#     data_path,
#     get_positive_pairs_from_list_of_bio_replicates(experiment_to_cell_type, experiment_to_bio_replicate, cell_types=["GM12878"], bio_replicates=[1,3,4,5,6]),
#     get_negative_pairs_from_list_of_bio_replicates(experiment_to_cell_type, experiment_to_bio_replicate, cell_types=["GM12878"], bio_replicates=[1,3,4,5,6]),
# )
# gm12878_only_dataset_test = HICPairsBioReplicatesDataset(
#     data_path,
#     get_positive_pairs_from_list_of_bio_replicates(experiment_to_cell_type, experiment_to_bio_replicate, cell_types=["GM12878"], bio_replicates=[32,33]),
#     get_negative_pairs_from_list_of_bio_replicates(experiment_to_cell_type, experiment_to_bio_replicate, cell_types=["GM12878"], bio_replicates=[32,33]),
# )
