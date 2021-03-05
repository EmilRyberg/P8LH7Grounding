import cv2
import os
import numpy as np
import json
import glob
import datetime

class CocoDatasetMaker:
    def __init__(self, dataset_dir):
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
                    "id": 2457650,
                    "name": "BlackCover",
                    "supercategory": "object"
                },
                {
                    "id": 2457649,
                    "name": "WhiteCover",
                    "supercategory": "object"
                },
                {
                    "id": 2457648,
                    "name": "BlueCover",
                    "supercategory": "object"
                },
                {
                    "id": 2457647,
                    "name": "BottomCover",
                    "supercategory": "object"
                },
                {
                    "id": 2457646,
                    "name": "PCB",
                    "supercategory": "object"
                }
            ],
            "licenses": []
        }

        self.dataset_dir = dataset_dir
        self.id_to_class_name_map = {
            2457650: "BlackCover",
            2457649: "WhiteCover",
            2457648: "BlueCover",
            2457647: "BottomCover",
            2457646: "PCB"
        }
        self.class_name_to_id_map = {
            "BlackCover": 2457650,
            "WhiteCover": 2457649,
            "BlueCover": 2457648,
            "BottomCover": 2457647,
            "PCB": 2457646
        }

        self.output_dir = 'dataset_output'
        if not os.path.isdir(self.output_dir):
            os.mkdir(self.output_dir)

    def create_dataset(self):
        image_folders = glob.glob(f'{self.dataset_dir}/*/')
        file_name_map_to_index_map = {}
        annotation_index = 0
        contour_folder_path = os.path.join(self.output_dir, 'img_with_contours')
        if not os.path.isdir(contour_folder_path):
            os.mkdir(contour_folder_path)

        for img_index, image_folder in enumerate(image_folders):
            full_image = cv2.imread(os.path.join(image_folder, 'full_image.png'))
            image_name = f'img{img_index}.png'
            image_json = {
                "id": img_index,
                "width": full_image.shape[1],
                "height": full_image.shape[0],
                "file_name": image_name,
                "license": None,
                "flickr_url": '',
                "coco_url": None,
                "date_captured": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "Z"
            }
            self.coco["images"].append(image_json)
            file_name_map_to_index_map[image_name] = img_index
            cv2.imwrite(os.path.join(self.output_dir, image_name), full_image)

            contours = []
            for mask_path in glob.glob(f'{image_folder}mask*.png'):
                annotations_made, mask_contours = self.__create_annotations_for_mask(mask_path, annotation_index, img_index)
                annotation_index += annotations_made
                for contour in mask_contours:
                    contours.append(contour)

            contour_img = cv2.drawContours(full_image, contours, -1, (0, 0, 255), 2)
            img_name_splitted = image_folder.split('/')
            original_img_name = img_name_splitted[len(img_name_splitted) - 2]
            cv2.imwrite(os.path.join(contour_folder_path, f'cont{img_index}_{original_img_name}.png'), contour_img)


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



if __name__ == '__main__':
    dataset_maker = CocoDatasetMaker('dataset')
    dataset_maker.create_dataset()
