import torch
import numpy as np
from torch.utils.data import Dataset
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageDraw, ImageChops
import os
try:
    from pycocotools.coco import COCO
except ImportError as e:
    from feature_extractor.pycocotools.coco import COCO

Image.MAX_IMAGE_PIXELS = None


class ClassificationDataset(Dataset):
    def __init__(self, dataset_dir, json_name, image_folder_path="."):
        if dataset_dir[-1] != '/':
            dataset_dir += '/'

        self.image_folder_path = image_folder_path
        self.dataset_dir = dataset_dir
        self.coco = COCO(dataset_dir + json_name) #COCO("./dataset_output/dataset.json")

        self.allAnnIds = []
        for annId in self.coco.getAnnIds(iscrowd=0):
            self.allAnnIds.append(annId)
        self.catIdToIndexMap = {}
        for index, catId in enumerate(self.coco.getCatIds()):
            self.catIdToIndexMap[catId] = index
            
        self.data_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.allAnnIds)

    def __getitem__(self, idx):
        annId = self.allAnnIds[idx]
        ann = self.coco.loadAnns(annId)[0]
        imgInfo = self.coco.loadImgs(ann['image_id'])
        img = Image.open(os.path.join(self.dataset_dir, self.image_folder_path, imgInfo[0]['file_name']))
        category_id = ann['category_id']
        class_id = self.catIdToIndexMap[category_id]

        if img.mode != "RGB":
            img = img.convert("RGB")
        mask = Image.new("RGB", img.size, 0)
        draw = ImageDraw.Draw(mask)

        for seg in ann["segmentation"]:
            x = seg[0::2]
            y = seg[1::2]
            xy = list()
            for i in range(len(x)):
                xy.append((x[i], y[i]))

            draw.polygon(xy, fill=(255, 255, 255), outline=None)

        try:
            resultFullImage = ImageChops.multiply(img, mask)
        except ValueError as e:
            print(f"sizes: {img.size} --- {mask.size}")
            raise e

        bbox = ann["bbox"]
        bbox[2] += bbox[0]
        bbox[3] += bbox[1]

        resultNoResize = resultFullImage.crop(bbox)
        image = self.data_transform(resultNoResize)

        return image, class_id

    def get_num_classes(self):
        return len(self.coco.getCatIds())


class TripletDataset(Dataset):
    def __init__(self, dataset_dir, json_name):
        if dataset_dir[-1] != '/':
            dataset_dir += '/'

        self.dataset_dir = dataset_dir
        self.coco = COCO(dataset_dir + json_name)

        self.images_grouped_by_class = []
        self.images_per_class = []
        self.allAnn = []
        class_id = 0
        for Id in self.coco.getCatIds():
            annIds = self.coco.getAnnIds(catIds=Id, iscrowd=0)
            annTemp = self.coco.loadAnns(annIds)
            self.allAnn.append(annTemp)
            self.images_per_class.append(len(annTemp))

            self.images_grouped_by_class.append((class_id, annTemp))
            class_id += 1

        # print(self.images_grouped_by_class[2])

        self.num_classes = len(self.coco.getCatIds())

        self.data_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.sample_triplets()

    def sample_triplets(self):
        self.triplets = []
        for id in range(self.num_classes):
            cid, class_images = self.images_grouped_by_class[id]
            class_images = np.array(class_images).copy()
            other_images = [r[1] for r in self.images_grouped_by_class if r[0] != cid]
            other_images = [item for sub_list in other_images for item in sub_list]
            other_images = np.array(other_images)
            np.random.shuffle(other_images)
            np.random.shuffle(class_images)
            for i in range(self.images_per_class[id] - 1):
                anchor = class_images[0]
                class_images = class_images[1:]
                positive = np.random.choice(class_images, 1)[0]
                self.triplets.append((cid, anchor, positive))

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        cid, a_ann, p_ann = self.triplets[idx]

        anns = [a_ann, p_ann]
        imgs = []

        for ann in anns:
            imgInfo = self.coco.loadImgs(ann['image_id'])
            img = Image.open(self.dataset_dir + imgInfo[0]['file_name'])

            mask = Image.new("RGB", img.size, 0)
            draw = ImageDraw.Draw(mask)

            for seg in ann["segmentation"]:
                x = seg[0::2]
                y = seg[1::2]
                xy = list()
                for i in range(len(x)):
                    xy.append((x[i], y[i]))

                draw.polygon(xy, fill=(255, 255, 255), outline=None)

            resultFullImage = ImageChops.multiply(img, mask)

            bbox = ann["bbox"]
            bbox[2] += bbox[0]
            bbox[3] += bbox[1]

            resultNoResize = resultFullImage.crop(bbox)

            image = self.data_transform(resultNoResize)

            imgs.append(image)

        return cid, imgs[0], imgs[1]
