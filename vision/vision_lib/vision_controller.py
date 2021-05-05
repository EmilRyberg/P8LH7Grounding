import rospy
from find_objects.find_objects import FindObjects, ObjectInfo
from feature_extractor.feature_extractor_module import FeatureExtractor
from vision_lib.ros_camera_interface import ROSCamera
from vision_lib.object_info_with_features import ObjectInfoWithFeatures
from typing import List
from PIL import Image
import cv2 as cv


class VisionController:
    def __init__(self, background_image_file, weights_path, init_node=False):
        if init_node:
            rospy.init_node("vision_test", anonymous=True)
        background_img = cv.imread(background_image_file)
        self.find_objects = FindObjects(background_img=background_img)
        self.feature_extractor = FeatureExtractor(weights_dir=weights_path, on_gpu=True)
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
            for i, (obj_info_with_features) in enumerate(objects_with_features):
                cv.imshow(f"obj-{i+1}", obj_info_with_features.object_img_cutout_cropped)
                print(f"obj-{i+1} features: {','.join([str(x) for x in obj_info_with_features.features])}")
            cv.waitKey(0)
        return objects_with_features

    def get_rgb(self):
        return self.camera.get_image()

    def get_depth(self):
        return self.camera.get_depth()


if __name__ == "__main__":
    vc = VisionController("background.png", "../../dialog_flow/nodes/feature_extraction.pth", init_node=True)
    masks_with_features = vc.get_masks_with_features(debug=True)
    obj = masks_with_features[0]
