from database_handler.database_handler import DatabaseHandler
from grounding.spatial import SpatialRelation, StatusEnum
import numpy as np
from typing import Optional
from scripts.vision_controller import VisionController as FakeVisionController
from enum import Enum


class ErrorType(Enum):
    UNKNOWN = "unknown object"
    CANT_FIND = "cant find object"
    ALREADY_KNOWN = "known object"
    TWO_REF = "two reference objects"


class GroundingReturn:
    def __init__(self):
        self.is_success = False
        self.error_code = None
        self.object_info = None


class Grounding:
    def __init__(self, db=DatabaseHandler("../grounding.db"), vision_controller=FakeVisionController(), spatial=SpatialRelation()):
        self.db = db
        self.spatial = spatial
        self.vision = vision_controller
        self.return_object = GroundingReturn()
        self.object_info = None

    def find_object(self, object_entity):
        name = object_entity.name
        spatial_desc = object_entity.spatial_descriptions
        found_object = False
        indexes_below_threshold = []
        distances = []

        db_features = self.db.get_feature(name)
        if db_features is None:
            self.return_object.is_success = False
            self.return_object.error_code = ErrorType.UNKNOWN
            return self.return_object

        object_infos_with_features = self.vision.get_masks_with_features()  # list of
        for i, obj in enumerate(object_infos_with_features):
            feature = obj.features
            distance = self.embedding_distance(db_features, feature)
            is_below_threshold = self.is_same_object(db_features, feature, threshold=0.1) # TODO update threshold
            if is_below_threshold:
                found_object = True
                distances.append(distance)
                indexes_below_threshold.append(i)

        if not found_object:
            self.return_object.is_success = False
            self.return_object.error_code = ErrorType.CANT_FIND
            return self.return_object

        if len(indexes_below_threshold) > 1:
            if spatial_desc:
                self.return_object.object_info, status = self.find_object_with_spatial_desc(object_entity, object_infos_with_features)
                if status == StatusEnum.ERROR_TWO_REF:
                    self.return_object.is_success = False
                    self.return_object.error_code = ErrorType.TWO_REF
                    return self.return_object
                elif status == StatusEnum.ERROR_CANT_FIND:
                    self.return_object.is_success = False
                    self.return_object.error_code = ErrorType.CANT_FIND
                    return self.return_object
                elif self.return_object.object_info:
                    self.return_object.is_success = True
                    return self.return_object

        # This part of the code will be executed if there is only 1 of the requested objects in the scene or
        # if the user does not care about what part is picked up.
        best_match_index = self.find_best_match(indexes_below_threshold, distances)
        self.return_object.object_info = object_infos_with_features[best_match_index]
        self.return_object.is_success = True
        return self.return_object

    def find_object_with_spatial_desc(self, object_entity, object_info_with_features):
        objects = []
        db_objects = self.db.get_all_features()
        features_below_threshold = []
        distances = []

        for i, (name, db_features) in enumerate(db_objects):
            for inner_idx, obj_info in enumerate(object_info_with_features):
                feature = obj_info.features
                bbox = obj_info.bbox_xxyy
                distance = self.embedding_distance(db_features, feature)
                is_below_threshold = self.is_same_object(db_features, feature, threshold=0.15) # TODO update threshold
                if is_below_threshold:
                    distances.append(distance)
                    features_below_threshold.append(obj_info)
                    objects.append((inner_idx, name, bbox))

        target_index, status = self.spatial.locate_specific_object(object_entity, objects)
        if status == StatusEnum.SUCCESS:
            object_info = object_info_with_features[target_index]
            return object_info, status

        return None, status

    def learn_new_object(self, object_entity):
        entity_name = object_entity.name

        db_features = self.db.get_feature(entity_name)
        if db_features is None:
            features = self.vision.get_masks_with_features()
            if not features:
                raise Exception("Failed to get features")
            else:
                self.db.insert_feature(entity_name, features[0].features)
                self.return_object.is_success = True
                return self.return_object
        else:
            self.return_object.is_success = False
            self.return_object.error_code = ErrorType.ALREADY_KNOWN
            return self.return_object

    def update_features(self, object_entity):
        entity = object_entity.name
        db_features = self.db.get_feature(entity)
        if db_features is None:
            self.return_object.is_success = False
            self.return_object.error_code = ErrorType.UNKNOWN
            return self.return_object
        else:
            features = self.vision.get_masks_with_features()
            self.db.update(entity, features[0].features)
            self.return_object.is_success = True
            return self.return_object

    def embedding_distance(self, features_1, features_2):
        return np.linalg.norm(features_1 - features_2)

    def is_same_object(self, features_1, features_2, threshold=1.0):
        return self.embedding_distance(features_1, features_2) < threshold

    def find_best_match(self, indexes_below_threshold, distances) -> Optional[tuple]:
        if isinstance(distances, np.ndarray):
            distances = distances.tolist()
        if len(indexes_below_threshold) != len(distances):
            raise ValueError("list_of_tuples should be same length as distances")
        if len(indexes_below_threshold) == 0 or len(distances) == 0:
            return None
        min_distance = 3
        best_index = None
        for i, distance in enumerate(distances):
            if distance < min_distance:
                min_distance = distance
                best_index = i
        return indexes_below_threshold[best_index]


if __name__ == "__main__":
    grounding = Grounding()
