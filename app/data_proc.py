from pathlib import Path
from typing import List, Tuple

import hicstraw
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


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("-H", "--hic_data_path", type=str, required=True)
    parser.add_argument("-s", "--save_to", type=str, required=True)
    parser.add_argument("-E", "--experiments", type=str, nargs="+", required=True)
    parser.add_argument("-r", "--resolution", type=int, default=5000)
    parser.add_argument(
        "-n",
        "--normalization",
        choices=["VC", "VC_SQRT", "KR", "SCALE", "NONE"],
        default="VC",
    )
    args = parser.parse_args()

    hic_data_path = Path(args.hic_data_path)
    assert hic_data_path.exists(), f"{hic_data_path} does not exist"

    save_to = Path(args.save_to)
    save_to.mkdir(parents=True, exist_ok=True)

    for exp in args.experiments:
        convert_hic_file_to_images(
            str(hic_data_path / f"{exp}.hic"),
            str(save_to / exp),
            resolution=args.resolution,
            hic_normalization=args.normalization,
        )
