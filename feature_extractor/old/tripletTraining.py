from pycocotools.coco import COCO
from PIL import Image, ImageDraw, ImageChops
import torch
import random
from torchvision import transforms
import numpy
from collections import OrderedDict
import gc

def get_ann_ids_for_all_categories():
    # get all category ids
    catIds = coco.getCatIds()
    
    # appending image ids of each category
    imgIds = [] # ids for each image for each category
    for id in catIds:
        temp = coco.getAnnIds(catIds=id, iscrowd=0)
        imgIds.append(temp)

    return imgIds


class feature_CNN:
    def __init__(self):
        self.model = torch.load("F:/P8/temp_triplet/triplet-epoch-0-loss-0.094870-14.pth")

        for param in self.model.parameters():
            param.requires_grad = False

        for param in self.model.classifier.parameters():
            param.requires_grad = True

        self.model.train()

        if torch.cuda.is_available():
            self.model.cuda()

    # This is for training on COCO
    def ann_to_embedding(self, ann):
        imgInfo = coco.loadImgs(ann['image_id']) # Get image meta info
        img = Image.open("./data/train2017/" + imgInfo[0]['file_name']) # Use info to load image from data set

        # For when images are not RGB
        if img.mode == 'L':
            return 0

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

        if resultNoResize.size[0] == 0 or resultNoResize.size[1] == 0:
            return 0

        # sample execution (requires torchvision)
        preprocess = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        input_tensor = preprocess(resultNoResize)
        input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

        # move the input and model to GPU for speed if available
        if torch.cuda.is_available():
            input_batch = input_batch.cuda()

        result = self.model(input_batch)
        del input_batch
        return result

def getTripletLoss(anchor, positive, negative, margin):
    return max(torch.norm(anchor - positive) - torch.norm(anchor - negative) + margin, 0)



if __name__ == '__main__':

    path2data="./data/train2017"
    path2json="./data/annotations/instances_train2017.json"

    coco = COCO(path2json)

    if torch.cuda.is_available():
        print("Cuda ON")

    AllAnnIds = get_ann_ids_for_all_categories()

    FCNN = feature_CNN()

    annTemp = coco.loadAnns(AllAnnIds[0][0])
    FCNN.ann_to_embedding(annTemp[0])

    margin = 0.1
    criterion = torch.nn.modules.loss.TripletMarginLoss(margin=margin, p=2)

    optimizer = torch.optim.SGD(FCNN.model.parameters(), lr=0.15, weight_decay=0.001, momentum=0.9)

    running_loss = 0.0
    mini_batches = 0
    epoch_loss = 0.0
    epoch_mini_batches = 0

    for epoch in range(1000):
        optimizer.zero_grad()

        anchor = torch.empty(0, 128)
        positive = torch.empty(0, 128)
        negative = torch.empty(0, 128)

        if torch.cuda.is_available():
            anchor = anchor.cuda()
            positive = positive.cuda()
            negative = negative.cuda()

        for i in range(512):
            # Getting data for triplet
            r2 = random.sample(AllAnnIds, 2)
            anchorPosAnn = random.sample(r2[0], 2)

            annTemp = coco.loadAnns(anchorPosAnn[0])
            anchorTemp = FCNN.ann_to_embedding(annTemp[0])
            if type(anchorTemp) is int:
                continue

            annTemp = coco.loadAnns(anchorPosAnn[1])
            positiveTemp = FCNN.ann_to_embedding(annTemp[0])
            if type(positiveTemp) is int:
                continue

            for i in range(5):
                negativeAnn = random.sample(r2[1], 1)
                annTemp = coco.loadAnns(negativeAnn[0])
                negativeTemp = FCNN.ann_to_embedding(annTemp[0])
                if type(negativeTemp) is int:
                    continue
                
                if getTripletLoss(anchorTemp, positiveTemp, negativeTemp, margin) > 0:
                    anchor = torch.cat((anchor, anchorTemp))
                    positive = torch.cat((positive, positiveTemp))
                    negative = torch.cat((negative, negativeTemp))
                    break

            del anchorTemp, positiveTemp, negativeTemp

        # Adjusting model parameters
        loss = criterion(anchor, positive, negative)
        loss.backward()
        optimizer.step()
        running_loss += float(loss)
        epoch_loss += float(loss)

        del anchor, positive, negative

        avg_loss = running_loss
        print(f"[{epoch + 1}] loss: {avg_loss:.10f}")
        running_loss = 0.0
        mini_batches += 1
        epoch_mini_batches += 1

        if (epoch + 1) % 50 == 0:
            avg_epoch_loss = epoch_loss / epoch_mini_batches
            epoch_mini_batches = 0
            epoch_loss = 0
            vers = epoch // 50
            print(f"[{vers + 1}] loss: {avg_epoch_loss:.10f}")

            epoch_loss += loss.item()
            checkpoint_name = f"./triplet-epoch-{vers}-loss-{avg_epoch_loss:.5f}" + str(vers) +  "-15" + ".pth"
            torch.save(FCNN.model, checkpoint_name)
            print(f"[{epoch + 1}] Saving checkpoint as {checkpoint_name}")

    print("Done, done diego")





