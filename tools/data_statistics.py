import os
import numpy as np
from PIL import Image
import argparse


def parse_args():
    """Parse the command line arguments for the TOF script"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--state",
        type=str,
        help="Set it to the foldernames of your sites as string",
    )
    return parser.parse_args()


def get_statistics(mask_path, num_classes):
    mask = Image.open(mask_path)
    mask_crop = np.array(mask)
    bins = np.array(range(num_classes))
    class_pixel_counts, _ = np.histogram(mask_crop, bins=bins)
    cf = class_pixel_counts / (mask_crop.shape[0] * mask_crop.shape[1])

    return cf


if __name__ == "__main__":
    args = parse_args()
    state = args.state
    masks_dir = f"data/sites/{state}/Masks"
    mask_paths = [
        file
        for file in os.listdir(f"data/sites/{state}/Masks")
        if file.endswith(".tif")
    ]
    num_classes = 6
    cf_list = []
    for mask_path in mask_paths:
        mask_path = os.path.join(masks_dir, mask_path)
        cf = get_statistics(mask_path, num_classes)
        cf_list.append(cf)

    # do a average over all the masks
    cf_list = np.array(cf_list)
    cf_avg = np.mean(cf_list, axis=0)
    print("Average Distribution", cf_avg)
    mask_names = [mask_path.split("/")[-1] for mask_path in mask_paths]
    cf_avg_with_names = list(zip(mask_names, cf_list))

    # Save as a txt file

    # Choose 5 of the masks so they have the same distribution as the average
    mask_paths = np.array(mask_paths)
    mask_paths_selected = []
    cf_avg_select = np.zeros(num_classes - 1)
    while not np.allclose(cf_avg, cf_avg_select, atol=0.01):
        cf_list = []
        mask_paths_selected = []
        for i in range(5):
            mask_path = mask_paths[np.random.choice(len(mask_paths))]
            mask_paths_selected.append(mask_path)
            mask_path = os.path.join(masks_dir, mask_path)
            cf = get_statistics(mask_path, num_classes)
            cf_list.append(cf)
        cf_list = np.array(cf_list)
        cf_avg_select = np.mean(cf_list, axis=0)

    print("Selected masks", mask_paths_selected)

    # save average distribution and selected masks
    if not os.path.exists(f"data/sites/{state}/Stats"):
        os.makedirs(f"data/sites/{state}/Stats")
    np.savetxt(f"data/sites/{state}/Stats/average_distribution.txt", cf_avg)
    np.savetxt(
        f"data/sites/{state}/Stats/selected_masks.txt", mask_paths_selected, fmt="%s"
    )
