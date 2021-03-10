from PIL import Image, ImageDraw, ImageChops
from pycocotools.coco import COCO

def get_ann_ids_for_all_categories():
    # get all category ids
    catIds = coco.getCatIds()
    
    # appending image ids of each category
    imgIds = [] # ids for each image for each category
    for id in catIds:
        temp = coco.getAnnIds(catIds=id, iscrowd=0)
        imgIds.append(temp)

    return imgIds

def ann_to_crop_no_background(ann):
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

    resultNoResize = resultFullImage.crop(bbox)

    resultNoResize.save("./data2/" + str(idx1) + "-" + str(idx2))


if __name__ == '__main__':

    path2data="./data/train2017"
    path2json="./data/annotations/instances_train2017.json"

    coco = COCO(path2json)

    AllAnnIds = get_ann_ids_for_all_categories()

    annTemp = coco.loadAnns(AllAnnIds[44][88])

    ann_to_crop_no_background(annTemp[0])