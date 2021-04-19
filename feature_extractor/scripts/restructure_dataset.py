import glob
import os
import pathlib
from pathlib import Path
import shutil

ID_TO_CLASS_NAME_MAP = {
    "1": "BlackCover",
    "3": "WhiteCover",
    "2": "BlueCover",
    "0": "BottomCover",
    "4": "PCB"
}

def restructure_dataset(dataset_path, output_path):
    output_path_obj = Path(output_path)
    if not output_path_obj.is_dir():
        os.mkdir(output_path)
    images_folder = Path(os.path.join(dataset_path, "images"))
    masks_folder = Path(os.path.join(dataset_path, "masks"))
    img_paths = {}
    for subfolder in images_folder.iterdir():
        if not subfolder.is_dir():
            continue
        for image_path in subfolder.iterdir():
            img_paths[image_path.stem] = image_path
    for subfolder in masks_folder.iterdir():
        if not subfolder.is_dir():
            continue
        for mask_path in subfolder.iterdir():
            img_paths[mask_path.stem] = (img_paths[mask_path.stem], mask_path)
    for index, key in enumerate(img_paths.keys()):
        save_path = os.path.join(output_path, f"img{index}")
        os.mkdir(save_path)
        img_path, mask_path = img_paths[key]
        shutil.copyfile(img_path, os.path.join(save_path, "full_image.png"))
        class_id = key.split("_")[0]
        shutil.copyfile(mask_path, os.path.join(save_path, f"mask_{ID_TO_CLASS_NAME_MAP[class_id]}.png"))


if __name__ == "__main__":
    restructure_dataset("dataset", "output_dataset")
