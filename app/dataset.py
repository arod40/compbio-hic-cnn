from torch.utils.data import Dataset
from PIL import Image
import torch
import torchvision.transforms as T
from tqdm import tqdm
from pathlib import Path
from typing import List, Tuple, Dict


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

    def __get_extra_info(self, path):
        return {
            "experiment": path.parent.parent.name,
            "window": tuple(map(int, path.stem.split("."))),
            "chr": path.parent.name,
            "path": str(path),
        }

    def __getitem__(self, index):
        src_img1, src_img2 = self.all_pairs[index]
        img1, img2 = self.transform(Image.open(src_img1)), self.transform(
            Image.open(src_img2)
        )
        return {
            "input1": img1,
            "input2": img2,
            "label": torch.tensor(
                [0 if index < len(self.negative_pairs) else 1], dtype=torch.float32
            ),
            "extra1": self.__get_extra_info(src_img1),
            "extra2": self.__get_extra_info(src_img2),
        }

    def __len__(self):
        return len(self.all_pairs)
