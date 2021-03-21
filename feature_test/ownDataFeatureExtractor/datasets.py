import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageDraw, ImageChops
from pycocotools.coco import COCO

class ClassificationDataset(Dataset):
    def __init__(self, dataset_dir, json_name):
        if dataset_dir[-1] != '/':
            dataset_dir += '/'

        self.dataset_dir = dataset_dir
        self.coco = COCO(dataset_dir + json_name) #COCO("./dataset_output/dataset.json")

        self.allAnn = []
        for annId in self.coco.getAnnIds(iscrowd=0):
            annTemp = self.coco.loadAnns(annId)
            self.allAnn.append(annTemp[0])
            
        self.data_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.allAnn)

    def __getitem__(self, idx):
        ann = self.allAnn[idx]
        imgInfo = self.coco.loadImgs(ann['image_id'])
        img = Image.open(self.dataset_dir + imgInfo[0]['file_name'])
        class_id = ann['category_id'] - 2457646 # Weird label off-set

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
        image = F.interpolate(image.unsqueeze(0), (224, 244)).squeeze(0)
        #image = image.squeeze(0) # image = F.interpolate(image.unsqueeze(0), (224, 224)).squeeze(0)

        return image, class_id


CD = ClassificationDataset("data/dataset_output/", "dataset.json")

x = CD.__getitem__(0)
