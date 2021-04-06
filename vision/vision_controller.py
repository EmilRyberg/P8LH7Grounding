import numpy as np
import utility.find_objects.find_objects as FindObjects
import vision.ros_camera_interface as CameraInterface
import feature_extractor.feature_extractor.feature_extractor_module as FeatureExtractor
weights_dir=""
class ObjectInfoWFeatures(FindObjects.ObjectInfo()):
    # overloads available: none (empty)
    # bbox: array size 4, [x,y,width,height]
    # bbox,mask_full,mask_cropped
    # bbox,mask_full,mask_cropped,img_cutout_full,img_cutout_cropped
    def __init__(self, *args,**kwargs):
        # python overloads make me sad - sam
        super(ObjectInfoWFeatures, self).__init__(*args)
        self.features = kwargs.pop('features')


class VisionController():
    def get_bounding_with_features(self):  # dummy function so i can mock it
        features = []
        features.append((np.array([1, 2, 3, 4]), np.array([1, 1, 1, 1, 1])))
        return features

class ActualVisionController(VisionController):
    def __init__(self):
        global weights_dir
        self.cam = CameraInterface.ROSCamera()
        self.bkgimage = self.getBkgImage()
        self.object_finder = FindObjects.FindObjects(self.bkgimage, crop_widths=[50, 50, 200, 600])
        width, height = cv.GetSize(bkgimage)
        self.feature_extractor = FeatureExtractor.FeatureExtractor(weights_dir=weights_dir,on_gpu=True,image_size=[width,height])

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
        objectInfos = []
        objectInfos = getMasksFromSegmenter(image)
        objInfosWFeatures = []
        for f in objectInfos:
            featuresArray = getFeatures(f.object_img_cutout_cropped)
            upgrObjInfo = ObjectInfoWFeatures(f.GetBBox(),f.mask_full,f.mask_cropped,f.object_img_cutout_full,object_img_cutout_cropped,features=featuresArray)
            objInfosWFeatures.append(upgrObjInfo)

        return objInfosWFeatures
    def getImage(self):
        return self.cam.get_image()
    def getBkgImage(self):
        # TODO: Need to make this actually get a background image
        return self.cam.get_bkg_image()
    def getMasksFromSegmenter(self,image):
        objectInfo = self.object_finder.find_objects(self.image, debug=True)
        return objectInfo
    def getFeatures(self,image):
        feature_info = self.feature_extractor.get_features(image)
        return feature_info

if __name__ == "__main__":
    vision = ActualVisionController()
    print(vision.get_bounding_with_features())
