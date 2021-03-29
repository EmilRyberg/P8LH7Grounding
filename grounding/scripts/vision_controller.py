import numpy as np

class VisionController():
    def get_masks_with_features(self):  # dummy function so i can mock it
        features = []
        feature = np.array([1, 1, 1, 1, 1])
        bbox = np.array([1, 2, 3, 4])
        mask = np.array([4, 3, 2 ,1])
        cropped_rbg = np.array([5, 5, 5, 5])
        features.append((feature, bbox, mask, cropped_rbg))
        return features

if __name__ == "__main__":
    vision = VisionController()
    print(vision.get_masks_with_features())