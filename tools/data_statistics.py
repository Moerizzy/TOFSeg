import argparse
from pathlib import Path
import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    """
    Parse the command line arguments for the TOF script.

    Returns:
        argparse.Namespace: An object containing the parsed command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--state",
        type=str,
        help="Set it to the foldernames of your sites as string",
    )
    return parser.parse_args()


def get_statistics(mask_path: str, num_classes: int) -> np.ndarray:
    """
    Calculate the class frequencies of a given mask image.

    Args:
        mask_path (str): The path to the mask image file.
        num_classes (int): The number of classes in the mask image.

    Returns:
        numpy.ndarray: An array containing the class frequencies.

    """
    mask = Image.open(mask_path)
    mask_crop = np.array(mask)
    bins = np.array(range(num_classes))
    class_pixel_counts, _ = np.histogram(mask_crop, bins=bins)
    cf = class_pixel_counts / (mask_crop.shape[0] * mask_crop.shape[1])

    return cf


def main() -> None:
    args = parse_args()
    state = args.state
    masks_dir = Path(f"data/sites/{state}/Masks")
    mask_paths = [file for file in masks_dir.iterdir() if file.suffix == ".tif"]
    num_classes = 6
    cf_list = []
    for mask_path in mask_paths:
        cf = get_statistics(str(mask_path), num_classes)
        cf_list.append(cf)

    # Calculate average distribution
    cf_list = np.array(cf_list)
    cf_avg = np.mean(cf_list, axis=0)
    print("Average Distribution:", cf_avg)

    # Choose 5 masks with similar distribution as the average
    mask_paths_selected = []
    cf_avg_select = np.zeros(num_classes - 1)
    while not np.allclose(cf_avg, cf_avg_select, atol=0.01):
        cf_list = []
        mask_paths_selected = []
        for _ in range(5):
            mask_path = np.random.choice(mask_paths)
            mask_paths_selected.append(mask_path)
            cf = get_statistics(str(mask_path), num_classes)
            cf_list.append(cf)
        cf_list = np.array(cf_list)
        cf_avg_select = np.mean(cf_list, axis=0)

    print("Average Distribution Selected Masks:", cf_avg_select)
    print("Selected masks:", mask_paths_selected)

    # Save average distribution and selected masks and their distributions
    stats_dir = Path(f"data/sites/{state}/Stats")
    stats_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(stats_dir / "average_distribution.txt", cf_avg)
    np.savetxt(stats_dir / "selected_masks.txt", mask_paths_selected, fmt="%s")
    np.savetxt(stats_dir / "selected_masks_distribution.txt", cf_list, fmt="%s")
    np.savetxt(stats_dir / "average_distribution_selected_masks.txt", cf_avg_select)


if __name__ == "__main__":
    main()
