import cv2
import os
import numpy as np
import json
import glob
import datetime
from pathlib import Path

class CocoDatasetMaker:
    def __init__(self, dataset_dir, img_index_offset=0, label_index_offset=0, output_dir="dataset_output"):
        self.coco = {
            "info": {
                "year": 2020,
                "version": 1.0,
                "description": "sim-dataset",
                "url": "www.YEET.nope",
                "date_created": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "Z"
            },
            "images": [],
            "annotations": [],
            "categories": [
                {
                    "id": 4,
                    "name": "BlackCover",
                    "supercategory": "object"
                },
                {
                    "id": 3,
                    "name": "WhiteCover",
                    "supercategory": "object"
                },
                {
                    "id": 2,
                    "name": "BlueCover",
                    "supercategory": "object"
                },
                {
                    "id": 1,
                    "name": "BottomCover",
                    "supercategory": "object"
                },
                {
                    "id": 0,
                    "name": "PCB",
                    "supercategory": "object"
                }
            ],
            "licenses": []
        }

        self.dataset_dir = dataset_dir
        self.id_to_class_name_map = {
            4: "BlackCover",
            3: "WhiteCover",
            2: "BlueCover",
            1: "BottomCover",
            0: "PCB"
        }
        self.class_name_to_id_map = {
            "BlackCover": 4,
            "WhiteCover": 3,
            "BlueCover": 2,
            "BottomCover": 1,
            "PCB": 0
        }

        self.output_dir = output_dir
        self.img_index_offset = img_index_offset
        self.label_index_offset = label_index_offset
        if not os.path.isdir(self.output_dir):
            os.mkdir(self.output_dir)

    def create_dataset(self):
        image_folders = glob.glob(f'{self.dataset_dir}/*/')
        annotation_index = self.label_index_offset
        contour_folder_path = os.path.join(self.output_dir, 'img_with_contours')
        if not os.path.isdir(contour_folder_path):
            os.mkdir(contour_folder_path)

        for img_index, image_folder in enumerate(image_folders):
            full_image = cv2.imread(os.path.join(image_folder, 'full_image.png'))
            image_name = f'img{img_index + self.img_index_offset}.png'
            image_json = {
                "id": img_index + self.img_index_offset,
                "width": full_image.shape[1],
                "height": full_image.shape[0],
                "file_name": image_name,
                "license": None,
                "flickr_url": '',
                "coco_url": None,
                "date_captured": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "Z"
            }
            self.coco["images"].append(image_json)
            cv2.imwrite(os.path.join(self.output_dir, image_name), full_image)

            contours = []
            for mask_path in glob.glob(f'{image_folder}mask*.png'):
                annotations_made, mask_contours = self.__create_annotations_for_mask(mask_path, annotation_index, img_index + self.img_index_offset)
                annotation_index += annotations_made
                for contour in mask_contours:
                    contours.append(contour)

            contour_img = cv2.drawContours(full_image, contours, -1, (0, 0, 255), 2)
            original_img_name = Path(image_folder).stem
            cv2.imwrite(os.path.join(contour_folder_path, f'cont{img_index + self.img_index_offset}_{original_img_name}.png'), contour_img)


        print('writing json file')
        with open(os.path.join(self.output_dir, 'dataset.json'), 'w') as outfile:
            json.dump(self.coco, outfile)
        print('done')


    def __create_annotations_for_mask(self, mask_path, annotation_index, image_id):
        only_file_name = os.path.basename(mask_path)
        category_name = str(only_file_name.split('_')[1]).split('.')[0]
        category_id = self.class_name_to_id_map[category_name]

        img_mask = cv2.imread(mask_path)
        img_mask = cv2.cvtColor(img_mask, cv2.COLOR_BGR2GRAY)
        contours, _ = cv2.findContours(img_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        label_index = annotation_index
        valid_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= 300.0:
                epsilon = 0.005 * cv2.arcLength(contour, True)
                contour_approx = cv2.approxPolyDP(contour, epsilon, True)
                valid_contours.append(contour_approx)

        for contour in valid_contours:
            x, y, w, h = cv2.boundingRect(contour)
            coco_contour = []
            for xy_pair in contour:
                coco_contour.append(int(xy_pair[0][0]))
                coco_contour.append(int(xy_pair[0][1]))
            annotation_json = {
                "id": label_index,
                "image_id": image_id,
                "category_id": category_id,
                "segmentation": [
                    coco_contour
                ],
                "bbox": [
                    int(x),
                    int(y),
                    int(w),
                    int(h)
                ],
                "area": 0,  # is 0 in our real dataset
                "iscrowd": 0
            }
            self.coco["annotations"].append(annotation_json)
            label_index += 1

        return len(contours), valid_contours


def merge_json_files(main_json, other_json, output_json_path="merged.json"):
    with open(main_json, "r") as file:
        main_json_obj = json.load(file)
    with open(other_json, "r") as file:
        other_json_obj = json.load(file)
    main_json_obj["annotations"].extend(other_json_obj["annotations"])
    main_json_obj["images"].extend(other_json_obj["images"])
    with open(output_json_path, "w") as outfile:
        json.dump(main_json_obj, outfile)


def remove_small_masks(json_file_path, area_threshold=400, output_json_path="filtered.json"):
    with open(json_file_path, "r") as file:
        json_obj = json.load(file)
    annotations_to_keep = []
    for annotation in json_obj["annotations"]:
        contour = annotation["segmentation"][0]
        cv_contour = [(contour[i], contour[i+1]) for i in range(0, len(contour), 2)]
        cv_contour = np.array(cv_contour)
        area = cv2.contourArea(cv_contour)
        if area > area_threshold:
            annotations_to_keep.append(annotation)
    print(f"Had {len(json_obj['annotations'])} annotations, removed {len(json_obj['annotations']) - len(annotations_to_keep)} annotations")
    json_obj["annotations"] = annotations_to_keep
    with open(output_json_path, "w") as outfile:
        json.dump(json_obj, outfile)


if __name__ == '__main__':
    #dataset_maker = CocoDatasetMaker('output_dataset', img_index_offset=200, label_index_offset=2475, output_dir="dataset_output")
    #dataset_maker.create_dataset()
    #merge_json_files("dataset_1.json", "dataset_2.json")
    remove_small_masks("dataset_2/dataset.json", area_threshold=9000)
