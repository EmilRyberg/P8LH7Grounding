import numpy as np

class VisionController():
    def get_bounding_with_features(self):  # dummy function so i can mock it
        features = []
        features.append((np.array([1, 2, 3, 4]), np.array([1, 1, 1, 1, 1])))
        return features

if __name__ == "__main__":
    vision = VisionController()
    print(vision.get_bounding_with_features())