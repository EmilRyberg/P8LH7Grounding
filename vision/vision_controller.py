import numpy as np
import utility.find_objects.find_objects as FindObjects
import vision.ros_camera_interface as CameraInterface
import feature_extractor.feature_extractor.feature_extractor_module as FeatureExtractor
class VisionController():
    def get_bounding_with_features(self):  # dummy function so i can mock it
        features = []
        features.append((np.array([1, 2, 3, 4]), np.array([1, 1, 1, 1, 1])))
        return features

class ActualVisionController(VisionController):
    def __init__(self):
        self.cam = CameraInterface.ROSCamera()
        self.bkgimage = self.getBkgImage()
        self.object_finder = FindObjects.FindObjects(self.bkgimage, crop_widths=[50, 50, 200, 600])
        self.feature_extractor = FeatureExtractor.FeatureExtractor()

    def get_bounding_with_features(self):  # real deal

        #
        # ok so what do we want to do here?
        # 1) get image from camera, get background img from camera
        # 2) give image to object segmenter and get masks of objects
        # 3) populate a list of object info
        # 4) give masks to feature extractor, update list
        # 5) perhaps preprocess things like bounding box etc at this point to be helpful later
        # 6) return list
        #

        self.image = self.getImage()
        objectInfo = []
        objectInfo = self.object_finder.find_objects(self.image, debug=True)
        return features
    def getImage(self):
        return self.cam.get_image()
    def getBkgImage(self):
        # TODO: Need to make this actually get a background image
        return self.cam.get_bkg_image()
    def getMasksFromSegmenter(self,image):
        pass
    def getFeatures(self,image,masks):
        pass
if __name__ == "__main__":
    vision = ActualVisionController()
    print(vision.get_bounding_with_features())
