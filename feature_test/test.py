import torchvision.datasets as dset
from pycocotools.coco import COCO
from PIL import Image, ImageDraw, ImageChops
import torch
from pytorch_metric_learning import miners

def get_ann_ids_for_all_categories():
    # get all category ids
    catIds = coco.getCatIds()
    
    # appending image ids of each category
    imgIds = [] # ids for each image for each category
    for id in catIds:
        temp = coco.getAnnIds(catIds=id, iscrowd=0)
        imgIds.append(temp)
        print(len(imgIds))
        print(len(temp))

    
    return imgIds

def cropPolygon(ann):
    imgInfo = coco.loadImgs(ann['image_id']) # Get image meta info
    img = Image.open("./data/train2017/" + imgInfo[0]['file_name']) # Use info to load image from data set

    mask = Image.new("RGB", img.size, 0) # Mask for instance segmentation
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

    result = resultFullImage.crop(bbox)

    img.show()
    result.show()

    return result


path2data="./data/train2017"
path2json="./data/annotations/instances_train2017.json"

coco = COCO(path2json)

ids = get_ann_ids_for_all_categories()

model = torch.load("./MobileNetV2/MNV2_1.pt")

annTemp = coco.loadAnns(ids[44][88])




cropPolygon(annTemp[0])