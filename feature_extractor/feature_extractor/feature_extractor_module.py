from feature_extractor.model import FeatureExtractorNet
import torch
import torch.nn.functional as F
from torchvision.transforms import transforms
import numpy as np


def embedding_distance(features_1, features_2):
    return np.linalg.norm(features_1 - features_2)


class FeatureExtractor:
    def __init__(self, weights_dir, on_gpu=True, image_size=(224, 224)):
        self.model = FeatureExtractorNet()
        self.on_gpu = on_gpu
        self.image_size = image_size
        if on_gpu:
            self.model.load_state_dict(torch.load(weights_dir))
        else:
            self.model.load_state_dict(torch.load(weights_dir, map_location=torch.device('cpu')))
        self.model.eval()
        if on_gpu:
            self.model = self.model.cuda()
        self.data_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def get_features(self, img):
        # transform image from eg. numpy to a tensor and normalise it
        img_transformed = self.data_transform(img)
        if self.on_gpu:
            img_transformed = img_transformed.cuda()
        # run model and L2 normalise the features such that it is within the unit hypersphere
        features = self.model(img_transformed)
        features = F.normalize(features, p=2)
        features_np = features.detach().cpu().data.numpy()
        return features_np