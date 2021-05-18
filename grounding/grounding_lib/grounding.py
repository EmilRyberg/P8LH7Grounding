from database_handler.database_handler import DatabaseHandler
from grounding_lib.spatial import SpatialRelation, StatusEnum
import numpy as np
from typing import Optional
from scripts.vision_controller import VisionController as FakeVisionController
from enum import Enum


class GroundingErrorType(Enum):
    UNKNOWN = "unknown object"
    CANT_FIND = "cant find object"
    CANT_FIND_RELATION = "can't find object with matching relation"
    ALREADY_KNOWN = "known object"
    TWO_REF = "two reference objects"
    MULTIPLE_REF = "multiple reference objects"


class GroundingReturn:
    def __init__(self):
        self.is_success = False
        self.error_code = None
        self.object_infos = []


class Grounding:
    def __init__(self, db: DatabaseHandler, vision_controller, spatial: SpatialRelation):
        self.db = db
        self.spatial = spatial
        self.vision = vision_controller
        self.return_object = GroundingReturn()
        self.object_info = None

    def find_object(self, object_entity):
        name = object_entity.name.lower()
        spatial_desc = object_entity.spatial_descriptions
        found_object = False
        indexes_below_threshold = []
        distances = []

        db_features = self.db.get_feature(name)
        if db_features is None:
            self.return_object.is_success = False
            self.return_object.error_code = GroundingErrorType.UNKNOWN
            return self.return_object

        object_infos_with_features = self.vision.get_masks_with_features()  # list of
        for i, obj in enumerate(object_infos_with_features):
            feature = obj.features
            distance = self.embedding_distance(db_features, feature)
            is_below_threshold = self.is_same_object(db_features, feature, threshold=0.8) # TODO update threshold
            if is_below_threshold:
                found_object = True
                distances.append(distance)
                indexes_below_threshold.append(i)

        if not found_object:
            self.return_object.is_success = False
            self.return_object.error_code = GroundingErrorType.CANT_FIND
            return self.return_object

        if len(indexes_below_threshold) > 1:
            if spatial_desc:
                object_infos, status = self.find_object_with_spatial_desc(object_entity, object_infos_with_features)
                if status == StatusEnum.NO_VALID_OBJECTS:
                    self.return_object.is_success = False
                    self.return_object.error_code = GroundingErrorType.TWO_REF
                    return self.return_object
                elif status == StatusEnum.ERROR_CANT_FIND:
                    self.return_object.is_success = False
                    self.return_object.error_code = GroundingErrorType.CANT_FIND
                    return self.return_object
                elif status == StatusEnum.ERROR_CANT_FIND_RELATION:
                    self.return_object.is_success = False
                    self.return_object.error_code = GroundingErrorType.CANT_FIND_RELATION
                    return self.return_object
                elif object_infos is not None and len(object_infos) > 0:
                    self.return_object.object_infos = object_infos
                    self.return_object.is_success = True
                    return self.return_object
                else: #this should not happen
                    self.return_object.error_code = GroundingErrorType.UNKNOWN
                    return self.return_object

        # This part of the code will be executed if there is only 1 of the requested objects in the scene or
        # if the user does not care about what part is picked up.
        best_match_index = self.find_best_match(indexes_below_threshold, distances)
        self.return_object.object_infos = [object_infos_with_features[best_match_index]]
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
                is_below_threshold = self.is_same_object(db_features, feature, threshold=0.8) # TODO update threshold
                if is_below_threshold:
                    distances.append(distance)
                    features_below_threshold.append(obj_info)
                    objects.append((inner_idx, name, bbox))

        target_indices, status = self.spatial.locate_specific_object(object_entity, objects)
        if status == StatusEnum.SUCCESS:
            object_infos = [object_info_with_features[i] for i in target_indices]
            return object_infos, status

        return None, status

    def get_location(self, object_entity):
        object_infos_with_features = self.vision.get_masks_with_features()
        if len(object_entity.spatial_descriptions) == 0:
            return None, None
        db_objects = self.db.get_all_features()
        objects = []

        for i, (name, db_features) in enumerate(db_objects):
            for inner_idx, obj_info in enumerate(object_infos_with_features):
                feature = obj_info.features
                bbox = obj_info.bbox_xxyy
                is_below_threshold = self.is_same_object(db_features, feature, threshold=0.8) # TODO update threshold
                if is_below_threshold:
                    objects.append((inner_idx, name, bbox))
        coordinates, status = self.spatial.get_location(object_entity.spatial_descriptions, objects)
        if status != StatusEnum.SUCCESS:
            return None, None
        return coordinates, status

    def learn_new_object(self, entity_name):
        db_features = self.db.get_feature(entity_name)
        if db_features is None:
            features = self.vision.get_masks_with_features()
            if not features:
                raise Exception("Failed to get features")
            else:
                if len(features)>1:
                    self.return_object.is_success = False
                    self.return_object.error_code = GroundingErrorType.MULTIPLE_REF
                    return self.return_object
                self.db.insert_feature(entity_name, features[0].features)
                self.return_object.is_success = True
                return self.return_object
        else:
            self.return_object.is_success = False
            self.return_object.error_code = GroundingErrorType.ALREADY_KNOWN
            return self.return_object

    def update_features(self, object_entity):
        entity = object_entity.name
        db_features = self.db.get_feature(entity)
        if db_features is None:
            self.return_object.is_success = False
            self.return_object.error_code = GroundingErrorType.UNKNOWN
            return self.return_object
        else:
            features = self.vision.get_masks_with_features()
            self.db.update(entity, features[0].features)
            self.return_object.is_success = True
            return self.return_object

    def embedding_distance(self, features_1, features_2):
        return np.linalg.norm(features_1 - features_2)

    def is_same_object(self, features_1, features_2, threshold=0.8):
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
