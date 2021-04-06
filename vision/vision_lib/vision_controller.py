import rospy
from find_objects.find_objects import FindObjects, ObjectInfo
from feature_extractor.feature_extractor_module import FeatureExtractor
from vision_lib.ros_camera_interface import ROSCamera
from typing import List
from PIL import Image
import cv2 as cv


class ObjectInfoWithFeatures(ObjectInfo):
    def __init__(self, object_info: ObjectInfo, features):
        super().__init__()
        self.object_img_cutout_cropped = object_info.object_img_cutout_cropped
        self.object_img_cutout_full = object_info.object_img_cutout_full
        self.mask_full = object_info.mask_full
        self.bbox_xxyy = object_info.bbox_xxyy
        self.bbox_xywh = object_info.bbox_xywh
        self.mask_cropped = object_info.mask_cropped
        self.features = features


class VisionController:
    def __init__(self, background_image_file, weights_path, init_node=False):
        if init_node:
            rospy.init_node("vision_test", anonymous=True)
        background_img = cv.imread(background_image_file)
        self.find_objects = FindObjects(background_img=background_img)
        self.feature_extractor = FeatureExtractor(weights_dir=weights_path)
        self.camera = ROSCamera()

    def get_masks_with_features(self, debug=False):
        image = self.camera.get_image()
        objects: List[ObjectInfo] = self.find_objects.find_objects(image)
        objects_with_features = []
        cropped_images = [(i, cv.cvtColor(x.object_img_cutout_cropped, cv.COLOR_BGR2RGB)) for i, x in enumerate(objects)]
        pil_cropped_images = [Image.fromarray(x) for (i, x) in cropped_images]
        indexes = [i for (i, x) in cropped_images]
        all_features = self.feature_extractor.get_features(pil_cropped_images)
        features_with_index = [(i, features) for (i, features) in zip(indexes, all_features)]
        for i, features in features_with_index:
            object_with_features = ObjectInfoWithFeatures(objects[i], features)
            objects_with_features.append(object_with_features)
        if debug:
            cv.imshow("Image", image)
            for i, obj in enumerate(objects):
                cv.imshow(f"obj-{i+1}", obj.object_img_cutout_cropped)
            cv.waitKey(0)
        return objects_with_features


if __name__ == "__main__":
    vc = VisionController("background.png", "triplet-epoch-9-loss-0.16331.pth", init_node=True)
    vc.get_masks_with_features(debug=True)
