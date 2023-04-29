from pathlib import Path
from typing import List, Tuple

# import hicstraw
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm


def normalize_np_array_to_range(np_array: np.array, to_range: Tuple[int, int]):
    to_min, to_max = to_range

    np_array -= np_array.min()
    np_array /= np_array.max()
    np_array *= to_max - to_min
    np_array += to_min

    return np_array


def convert_hic_file_to_images(
    hic_path: str,
    save_to: str,
    resolution: int = 5000,
    target_pixels: int = 40,
    hic_normalization="VC",
):
    print(
        f"Extracting contact matrices from {hic_path} as {resolution}bp resolution with {hic_normalization} normalization"
    )
    hic = hicstraw.HiCFile(hic_path)
    save_to = Path(save_to)
    for chr in hic.getChromosomes():
        if chr.name == "All":
            continue

        print(f"Processing chromosome {chr.name}")

        chr_dir = save_to / chr.name
        chr_dir.mkdir(parents=True, exist_ok=True)

        mzd = hic.getMatrixZoomData(
            chr.name, chr.name, "observed", hic_normalization, "BP", resolution
        )
        for start in tqdm(range(0, chr.length, target_pixels * resolution)):
            end = start + target_pixels * resolution - 1
            numpy_matrix = mzd.getRecordsAsMatrix(start, end, start, end)

            if numpy_matrix.max() == 0:
                continue

            # convert to pixel range of values
            numpy_matrix = normalize_np_array_to_range(numpy_matrix, (0, 255))

            # save image
            Image.fromarray(numpy_matrix.astype(np.uint8)).save(
                chr_dir / f"{start}.{end}.jpg"
            )


class HICPairsBioReplicatesDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        bio_replicates_pairs: List[Tuple[str]],
        non_bio_replicates_pairs: List[Tuple[str]],
        chromosomes: List[str],
    ):
        self.root_dir = Path(root_dir)

        print("Building positive image pairs")
        self.positive_pairs = self.__build_image_pairs(
            bio_replicates_pairs, chromosomes
        )

        print("Building negative image pairs")
        self.negative_pairs = self.__build_image_pairs(
            non_bio_replicates_pairs, chromosomes
        )

        self.all_pairs = self.negative_pairs + self.positive_pairs

        self.transform = T.Compose(
            [
                T.ToTensor(),
            ]
        )

    def __build_image_pairs(self, pairs: List[Tuple[str]], chromosomes: List[str]):
        img_pairs = []
        for exp1, exp2 in pairs:
            print(f"Building image pairs for {exp1} and {exp2}")

            dir1, dir2 = Path(self.root_dir / exp1), Path(self.root_dir / exp2)
            assert dir1.exists() and dir2.exists(), f"{dir1} or {dir2} does not exist"

            if chromosomes is None:
                chromosomes = set(p.name for p in dir1.iterdir()).intersection(
                    set(p.name for p in dir2.iterdir())
                )

            for chr in tqdm(chromosomes):

                chr_dir1, chr_dir2 = dir1 / chr, dir2 / chr
                assert (
                    chr_dir1.exists() and chr_dir2.exists()
                ), f"{chr_dir1} or {chr_dir2} does not exist"

                dir1_imgs, dir2_imgs = set(p.name for p in chr_dir1.iterdir()), set(
                    p.name for p in chr_dir2.iterdir()
                )

                for img_name in dir1_imgs.intersection(dir2_imgs):
                    img_pairs.append((chr_dir1 / img_name, chr_dir2 / img_name))

        return img_pairs

    def __getitem__(self, index):
        img1, img2 = self.all_pairs[index]
        img1, img2 = self.transform(Image.open(img1)), self.transform(Image.open(img2))
        return {
            "input1": img1,
            "input2": img2,
            "label": torch.tensor([0 if index < len(self.negative_pairs) else 1], dtype=torch.float32),
        }

    def __len__(self):
        return len(self.all_pairs)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--hic_data_path", type=str, required=True)
    parser.add_argument("--save_to", type=str, required=True)
    parser.add_argument("--experiments", type=str, nargs="+", required=True)
    args = parser.parse_args()

    hic_data_path = Path(args.hic_data_path)
    assert hic_data_path.exists(), f"{hic_data_path} does not exist"

    save_to = Path(args.save_to)
    save_to.mkdir(parents=True, exist_ok=True)

    for exp in args.experiments:
        convert_hic_file_to_images(hic_data_path / f"{exp}.hic", save_to / exp)
