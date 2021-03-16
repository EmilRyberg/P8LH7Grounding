from PIL import Image, ImageDraw, ImageChops
import torch
from torchvision import transforms

import numpy as np
from pycocotools.coco import COCO

from operator import add

import math


class feature_extractor:
    def __init__(self):
        #self.model = torch.load("F:/P8/temp_triplet/triplet-epoch-0-loss-0.094870-14.pth")
        self.model = torch.hub.load('pytorch/vision:v0.7.0', 'mobilenet_v2', pretrained=True)

        self.model.eval()

        if torch.cuda.is_available():
            self.model.cuda()
    
    def bbox_to_embedding(self, img):
        # Convert to PIL image
        PIL_img = Image.fromarray(img)

        # Image transformations
        preprocess = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        input_tensor = preprocess(PIL_img)
        input_batch = input_tensor.unsqueeze(0)

        if torch.cuda.is_available():
            input_batch = input_batch.cuda()

        return self.model(input_batch)

    

# For use with COCO
def get_ann_ids_for_all_categories():
    # get all category ids
    catIds = coco.getCatIds()
    
    # appending image ids of each category
    imgIds = [] # ids for each image for each category
    for id in catIds:
        temp = coco.getAnnIds(catIds=id, iscrowd=0)
        imgIds.append(temp)

    return imgIds

# For use with COCO
def ann_to_mask(ann):
    imgInfo = coco.loadImgs(ann['image_id']) # Get image meta info
    img = Image.open("./data/train2017/" + imgInfo[0]['file_name']) # Use info to load image from data set

    # For when images are not RGB
    if img.mode == 'L':
        return 0

    mask = Image.new("L", img.size, 0) # Mask for instance segmentation
    draw = ImageDraw.Draw(mask)

    for seg in ann["segmentation"]:
        x = seg[0::2]
        y = seg[1::2]
        xy = list()
        for i in range(len(x)):
            xy.append((x[i], y[i]))

        draw.polygon(xy, fill=(255), outline=None)

    return img, mask

if __name__ == '__main__':
    #im = Image.open("bal.jpg")

    """path2json="./data/annotations/instances_train2017.json"

    coco = COCO(path2json)

    AllAnnIds = get_ann_ids_for_all_categories()

    annTemp = coco.loadAnns(AllAnnIds[2][0])
    img, mask = ann_to_mask(annTemp[0])

    fe = feature_extractor()

    fe.bbox_and_mask_to_color(img, mask)

    #data = np.asarray(im)

    #print(np.shape(data))
    #print(data[0, 0, 0])
    """

    img = Image.open("dog.jpg")
    data = np.asarray(img)
    print(type(data))
