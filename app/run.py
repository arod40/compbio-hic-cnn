from datetime import datetime
from pathlib import Path

import torch
from dataset import HICPairsBioReplicatesDataset
from modeling import *
from sklearn.metrics import accuracy_score
from utils import get_whole_dataset_split, train_loop


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

            models.append((name, model))


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

    checkpoint_dir = Path("../checkpoints/" + exp_name)
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
            save_dir = checkpoint_dir / f"celltype_{name}-{'-'.join(train_cell_types)}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
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


if __name__ == "__main__":
    # import fire

    # fire.Fire(original_allcells_experiment)
    original_allcells_experiment(
        "../data/hic_dataset-40x40-5k-VC", "../checkpoints/test"
    )
    original_allcells_experiment(
        "../data/hic_dataset-40x40-5k-VC", "../checkpoints/test"
    )
