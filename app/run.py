from datetime import datetime
from pathlib import Path
from typing import List

import torch
from dataset import HICPairsBioReplicatesDataset
from modeling import *
from sklearn.metrics import accuracy_score
from utils import (
    get_dataset_split_by_cell_type,
    get_negative_pairs_from_list_of_bio_replicates,
    get_positive_pairs_from_list_of_bio_replicates,
    get_whole_dataset_split,
    parse_mmc2_file,
    train_loop,
    dump_analytics_to_df,
)


def original_allcells_experiment(
    data_path: str,
    save_to: str,
    batch_size: int = 800,
    num_epochs: int = 30,
    no_exps: int = 3,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_dir = Path(save_to)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    criterion = torch.nn.BCELoss()
    eval_metrics = [
        ("loss", lambda x, y: criterion(x, y).item()),
        ("accuracy", lambda x, y: accuracy_score((x > 0.5).int(), y)),
    ]

    all_cells_dataset = HICPairsBioReplicatesDataset(
        data_path,
        [
            # GM12878
            ("GSM1551552_HIC003", "GSM1551554_HIC005"),
            # IMR90 (CCL-186)
            ("GSM1551599_HIC050", "GSM1551600_HIC051"),
            # HMEC (CC-2551)
            ("GSM1551608_HIC059", "GSM1551609_HIC060"),
            # NHEK (192627)
            ("GSM1551614_HIC065", "GSM1551615_HIC066"),
            # K562 (CCL-243)
            ("GSM1551618_HIC069", "GSM1551619_HIC070"),
            # KBM7
            ("GSM1551625_HIC076", "GSM1551626_HIC077"),
            # HUVEC
            ("GSM1551629_HIC080", "GSM1551630_HIC081"),
        ],
        [
            # GM12878
            ("GSM1551552_HIC003", "GSM1551569_HIC020"),
            # IMR90 (CCL-186)
            ("GSM1551599_HIC050", "GSM1551604_HIC055"),
            # HMEC (CC-2551)
            ("GSM1551607_HIC058", "GSM1551608_HIC059"),
            # K562 (CCL-243)
            ("GSM1551618_HIC069", "GSM1551620_HIC071"),
            # KBM7
            ("GSM1551624_HIC075", "GSM1551625_HIC076"),
        ],
        None,
    )

    for exp_no in range(no_exps):
        # Create data loaders
        train_data, val_data, test_data = get_whole_dataset_split(
            all_cells_dataset, repro_seed=exp_no
        )
        train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=batch_size, shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_data, batch_size=batch_size, shuffle=True
        )
        test_loader = torch.utils.data.DataLoader(
            test_data, batch_size=batch_size, shuffle=True
        )

        for name, SiameseNetworkClass in [
            ("resnet", SiameseNetworkResnetEncoder),
            ("lenet", SiameseNetworkLeNetEncoder),
            ("linear", SiameseNetworkLinearEncoder),
        ]:
            # Create model and optimizer instances
            model = SiameseNetworkClass((40, 40)).to(device)
            optimizer = torch.optim.Adam(model.parameters())

            # Create save directory
            save_dir = (
                checkpoint_dir
                / f"all_{name}-{exp_no}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
            )
            save_dir.mkdir(parents=True)

            # Train model
            train_loop(
                model,
                train_loader,
                val_loader,
                test_loader,
                batch_size,
                num_epochs,
                criterion,
                optimizer,
                eval_metrics,
                save_dir,
                device,
            )


def original_celltype_experiment(
    data_path: str,
    save_to: str,
    batch_size: int = 800,
    num_epochs: int = 30,
    no_exps: int = 3,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_dir = Path(save_to)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train_test_cell_types = [
        (["gm12878", "imr90", "hmec", "k562"], ["kbm7"]),
        (["gm12878", "imr90", "hmec", "kbm7"], ["k562"]),
        (["gm12878", "imr90", "k562", "kbm7"], ["hmec"]),
        (["gm12878", "hmec", "k562", "kbm7"], ["imr90"]),
        (["imr90", "hmec", "k562", "kbm7"], ["gm12878"]),
    ]

    checkpoint_dir = Path(save_to)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    criterion = torch.nn.BCELoss()
    eval_metrics = [
        ("loss", lambda x, y: criterion(x, y).item()),
        ("accuracy", lambda x, y: accuracy_score((x > 0.5).int(), y)),
    ]

    gm12878_dataset = HICPairsBioReplicatesDataset(
        data_path,
        [
            ("GSM1551552_HIC003", "GSM1551554_HIC005"),
        ],
        [
            ("GSM1551552_HIC003", "GSM1551569_HIC020"),
        ],
        None,
    )
    imr90_dataset = HICPairsBioReplicatesDataset(
        data_path,
        [
            ("GSM1551599_HIC050", "GSM1551600_HIC051"),
        ],
        [
            ("GSM1551599_HIC050", "GSM1551604_HIC055"),
        ],
        None,
    )
    hmec_dataset = HICPairsBioReplicatesDataset(
        data_path,
        [
            ("GSM1551608_HIC059", "GSM1551609_HIC060"),
        ],
        [
            ("GSM1551607_HIC058", "GSM1551608_HIC059"),
        ],
        None,
    )
    k562_dataset = HICPairsBioReplicatesDataset(
        data_path,
        [
            ("GSM1551618_HIC069", "GSM1551619_HIC070"),
        ],
        [
            ("GSM1551618_HIC069", "GSM1551620_HIC071"),
        ],
        None,
    )
    kbm7_dataset = HICPairsBioReplicatesDataset(
        data_path,
        [
            ("GSM1551625_HIC076", "GSM1551626_HIC077"),
        ],
        [
            ("GSM1551624_HIC075", "GSM1551625_HIC076"),
        ],
        None,
    )
    datasets_by_cell_type = {
        "gm12878": gm12878_dataset,
        "imr90": imr90_dataset,
        "hmec": hmec_dataset,
        "k562": k562_dataset,
        "kbm7": kbm7_dataset,
    }

    for exp_no, (train_cell_types, test_cell_types) in enumerate(train_test_cell_types):
        train_dataset, val_dataset, test_dataset = get_dataset_split_by_cell_type(
            datasets_by_cell_type, train_cell_types, test_cell_types, repro_seed=exp_no
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, shuffle=True
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size, shuffle=True
        )

        for name, SiameseNetworkClass in [
            ("resnet", SiameseNetworkResnetEncoder),
            ("lenet", SiameseNetworkLeNetEncoder),
            ("linear", SiameseNetworkLinearEncoder),
        ]:
            # Create model and optimizer instances
            model = SiameseNetworkClass((40, 40)).to(device)
            optimizer = torch.optim.Adam(model.parameters())

            # Create save directory
            save_dir = (
                checkpoint_dir
                / f"celltype_{name}-{'-'.join(train_cell_types)}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
            )
            save_dir.mkdir(parents=True)

            # Train model
            train_loop(
                model,
                train_loader,
                val_loader,
                test_loader,
                batch_size,
                num_epochs,
                criterion,
                optimizer,
                eval_metrics,
                save_dir,
                device,
            )


def custom_experiment(
    data_path: str,
    mmc2_file_path: str,
    save_checkpoints_to: str,
    save_analytics_to: str,
    train_cell_types: List[str],
    train_bio_replicates: List[int],
    test_cell_types: List[str] = None,
    test_bio_replicates: List[int] = None,
    batch_size: int = 800,
    num_epochs: int = 30,
    no_exps: int = 3,
    use_experiments: List[str] = None,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if use_experiments is None:
        use_experiments = [x.name for x in Path(data_path).iterdir()]

    experiment_to_cell_type, experiment_to_bio_replicate = parse_mmc2_file(
        mmc2_file_path
    )

    train_positive_pairs = get_positive_pairs_from_list_of_bio_replicates(
        experiment_to_cell_type,
        experiment_to_bio_replicate,
        cell_types=train_cell_types,
        bio_replicates=train_bio_replicates,
        use_experiments=use_experiments,
    )
    train_negative_pairs = get_negative_pairs_from_list_of_bio_replicates(
        experiment_to_cell_type,
        experiment_to_bio_replicate,
        cell_types=train_cell_types,
        bio_replicates=train_bio_replicates,
        use_experiments=use_experiments,
    )
    print(f"Number of train positive pairs: {len(train_positive_pairs)}")
    print(f"Number of train negative pairs: {len(train_negative_pairs)}")

    train_dataset = HICPairsBioReplicatesDataset(
        data_path,
        train_positive_pairs,
        train_negative_pairs,
        None,
    )
    test_dataset = None
    if test_cell_types is not None and test_bio_replicates is not None:
        test_positive_pairs = get_positive_pairs_from_list_of_bio_replicates(
            experiment_to_cell_type,
            experiment_to_bio_replicate,
            cell_types=test_cell_types,
            bio_replicates=test_bio_replicates,
            use_experiments=use_experiments,
        )
        test_negative_pairs = get_negative_pairs_from_list_of_bio_replicates(
            experiment_to_cell_type,
            experiment_to_bio_replicate,
            cell_types=test_cell_types,
            bio_replicates=test_bio_replicates,
            use_experiments=use_experiments,
        )
        print(f"Number of test positive pairs: {len(test_positive_pairs)}")
        print(f"Number of test negative pairs: {len(test_negative_pairs)}")

        test_dataset = HICPairsBioReplicatesDataset(
            data_path,
            test_positive_pairs,
            test_negative_pairs,
            None,
        )

    checkpoint_dir = Path(save_checkpoints_to)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    analytics_dir = Path(save_analytics_to)
    analytics_dir.mkdir(parents=True, exist_ok=True)

    criterion = torch.nn.BCELoss()
    eval_metrics = [
        ("loss", lambda x, y: criterion(x, y).item()),
        ("accuracy", lambda x, y: accuracy_score((x > 0.5).int(), y)),
    ]

    for exp_no in range(no_exps):
        # Create data loaders
        if test_dataset is None:
            train_data, val_data, test_data = get_whole_dataset_split(
                train_dataset, fractions=[0.6, 0.2], repro_seed=exp_no
            )
        else:
            train_data, val_data = get_whole_dataset_split(
                train_dataset, fractions=[0.75], repro_seed=exp_no
            )
            test_data = test_dataset

        train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=batch_size, shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_data, batch_size=batch_size, shuffle=True
        )
        test_loader = torch.utils.data.DataLoader(
            test_data, batch_size=batch_size, shuffle=True
        )

        models = []

        for name, SiameseNetworkClass in [
            ("resnet", SiameseNetworkResnetEncoder),
            ("lenet", SiameseNetworkLeNetEncoder),
            ("linear", SiameseNetworkLinearEncoder),
        ]:
            # Create model and optimizer instances
            model = SiameseNetworkClass((40, 40)).to(device)
            optimizer = torch.optim.Adam(model.parameters())

            # Create save directory
            save_dir = (
                checkpoint_dir
                / f"custom_{name}-{exp_no}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
            )
            save_dir.mkdir(parents=True)

            models.append((name, save_dir, SiameseNetworkClass))

            # Train model
            train_loop(
                model,
                train_loader,
                val_loader,
                test_loader,
                batch_size,
                num_epochs,
                criterion,
                optimizer,
                eval_metrics,
                save_dir,
                device,
            )

        train_df = dump_analytics_to_df(
            train_data, models, experiment_to_cell_type, device
        )
        train_df.to_csv(analytics_dir / f"custom_{exp_no}_train.csv", index=False)
        val_df = dump_analytics_to_df(val_data, models, experiment_to_cell_type, device)
        val_df.to_csv(analytics_dir / f"custom_{exp_no}_val.csv", index=False)
        test_df = dump_analytics_to_df(
            test_data, models, experiment_to_cell_type, device
        )
        test_df.to_csv(analytics_dir / f"custom_{exp_no}_test.csv", index=False)


if __name__ == "__main__":
    import fire

    fire.Fire(custom_experiment)
