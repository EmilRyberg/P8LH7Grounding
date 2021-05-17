import numpy as np
from enum import Enum
import copy
import math
from ner_lib.command_builder import SpatialType
from database_handler.database_handler import DatabaseHandler


class StatusEnum(Enum):
    SUCCESS = "SUCCESS"
    NO_VALID_OBJECTS = "ERROR_TWO_REF"
    ERROR_CANT_FIND = "ERROR_CANT_FIND"
    STATIC_LOCATION = "STATIC_LOCATION"


class SpatialRelation:
    def __init__(self, database_handler: DatabaseHandler):
        self.database_handler = database_handler

    def locate_specific_object(self, object_entity, objects):
        local_objects = copy.deepcopy(objects)
        entity_name = object_entity.name
        spatial_desc = object_entity.spatial_descriptions
        last_bboxes, last_spatial_description, local_objects = self.filter_objects(spatial_desc, local_objects)

        if last_spatial_description is None:
            return None, StatusEnum.NO_VALID_OBJECTS

        matching_objects = []
        for x, (id, name, bbox) in enumerate(local_objects):
            if name == entity_name:
                matching_objects.append((id, name, bbox))
        if len(matching_objects) > 1:
            correct_indices = []
            for bbox in last_bboxes:
                if last_spatial_description.spatial_type == SpatialType.OTHER:
                    x, y, z = self.database_handler.get_location_by_name(
                        last_spatial_description.object_entity.name.lower())
                    correct_bbox_index = self.find_closest_object_to_location(local_objects, (x, y))
                    if correct_bbox_index is not None:
                        (idx, name, bbox) = matching_objects[correct_bbox_index]
                        correct_indices.append(idx)
                else:
                    correct_bbox_index = self.find_best_match(matching_objects, last_spatial_description, bbox)
                    if correct_bbox_index is not None:
                        (idx, name, bbox) = matching_objects[correct_bbox_index]
                        correct_indices.append(idx)
            return correct_indices, StatusEnum.SUCCESS
        elif matching_objects:
            (id, name, bbox) = matching_objects[0]
            correct_index = id
            return [correct_index], StatusEnum.SUCCESS
        else:
            return None, StatusEnum.ERROR_CANT_FIND

    def locate_last_object_in_spatial_descriptions(self, spatial_descriptions, objects):
        local_objects = copy.deepcopy(objects)
        last_bboxes, last_spatial_description, _ = self.filter_objects(spatial_descriptions, local_objects)

        if last_spatial_description is None:
            return None, StatusEnum.NO_VALID_OBJECTS

        matching_objects = []
        for x, (idx, name, bbox) in enumerate(local_objects):
            if name == last_spatial_description.object_entity.name:
                matching_objects.append((idx, name, bbox))
        if last_spatial_description.spatial_type == SpatialType.OTHER:
            x, y, z = self.database_handler.get_location_by_name(last_spatial_description.object_entity.name.lower())
            correct_bbox_index = self.find_closest_object_to_location(local_objects, (x, y))
            (idx, name, bbox) = local_objects[correct_bbox_index]
            return [idx], StatusEnum.SUCCESS
        else:
            if len(matching_objects) > 1:
                correct_indices = []
                for bbox in last_bboxes:
                    center_x, center_y, _ = self.get_center_and_size(bbox)
                    correct_bbox_index = self.find_closest_object_to_location(matching_objects, (center_x, center_y))
                    if correct_bbox_index is not None:
                        (idx, name, bbox) = matching_objects[correct_bbox_index]
                        correct_indices.append(idx)
                return correct_indices, StatusEnum.SUCCESS
            elif matching_objects:
                (idx, name, bbox) = matching_objects[0]
                return [idx], StatusEnum.SUCCESS
            else:
                return None, StatusEnum.ERROR_CANT_FIND

    def filter_objects(self, spatial_descriptions, objects):
        local_objects = copy.deepcopy(objects)
        previous_spatial_description = None
        last_spatial_description = spatial_descriptions[-1]

        object_name = last_spatial_description.object_entity.name
        matching_objects = []
        valid_candidate_bboxes = []
        if last_spatial_description.spatial_type != SpatialType.OTHER:
            for (id, name, bbox) in local_objects:
                if name == object_name:
                    matching_objects.append((id, name, bbox))
            for idx, name, bbox in matching_objects:
                target_bbox, target_spatial_description = self.filter_sub_descriptions(list(reversed(spatial_descriptions))[1:],
                                                                                           local_objects, bbox, last_spatial_description)
                if target_bbox is not None:
                    valid_candidate_bboxes.append(target_bbox)
                    previous_spatial_description = target_spatial_description
        else:
            x, y, z = self.database_handler.get_location_by_name(last_spatial_description.object_entity.name.lower())
            if x is None:
                print("WARNING: static location is None")
                return None, None, None
            target_bbox, target_spatial_description = self.filter_sub_descriptions(list(reversed(spatial_descriptions))[1:],
                                                                                   local_objects, [x - 1, x + 1, y - 1, y + 1], last_spatial_description)
            if target_bbox is not None:
                valid_candidate_bboxes.append(target_bbox)
                previous_spatial_description = target_spatial_description

        return valid_candidate_bboxes, previous_spatial_description, local_objects

    def filter_sub_descriptions(self, spatial_descriptions, objects, initial_bbox, initial_spatial_description):
        previous_bbox = initial_bbox
        previous_spatial_description = initial_spatial_description

        if previous_spatial_description is None:
            return None, None

        for instance in spatial_descriptions:
            object_name = instance.object_entity.name
            matching_objects = []
            if instance.spatial_type != SpatialType.OTHER:
                for (id, name, bbox) in objects:
                    if name == object_name:
                        matching_objects.append((id, name, bbox))
                if len(matching_objects) > 1:
                    correct_bbox_index = self.find_best_match(matching_objects, previous_spatial_description, previous_bbox)
                    if correct_bbox_index is None: # this branch cant find object
                        return None, None
                    (last_id, _, previous_bbox) = matching_objects[correct_bbox_index]
                    previous_spatial_description = instance
                elif len(matching_objects) == 1:
                    (last_id, name, bbox) = matching_objects[0]
                    previous_spatial_description = instance
                    previous_bbox = bbox
                else:
                    return None, None
                objects.remove((last_id, object_name, previous_bbox))
            else:
                x, y, z = self.database_handler.get_location_by_name(instance.object_entity.name.lower())
                if x is None:
                    print("WARNING: static location is None")
                    continue
                previous_bbox = [x - 1, x + 1, y - 1, y + 1]
                previous_spatial_description = instance

        return previous_bbox, previous_spatial_description

    def get_location(self, spatial_descriptions, objects):
        if len(spatial_descriptions) == 1 and spatial_descriptions[0].spatial_type == SpatialType.OTHER:
            x, y, z = self.database_handler.get_location_by_name(spatial_descriptions[0].object_entity.name.lower())
            if x is None:
                print(f"WARNING: x for location '{spatial_descriptions[0].object_entity.name.lower()}' is None")
                return None, StatusEnum.NO_VALID_OBJECTS
            return [(x, y)], StatusEnum.SUCCESS

        best_object_indices, status = self.locate_last_object_in_spatial_descriptions(spatial_descriptions, objects)
        if status != StatusEnum.SUCCESS:
            return None, status
        else:
            object_infos = [x for x in objects if x[0] in best_object_indices]
            coordinates = []
            for idx, name, bbox in object_infos:
                center_x, center_y, _ = self.get_center_and_size(bbox)
                coordinates.append((round(center_x), round(center_y)))
            return coordinates, StatusEnum.SUCCESS

    def find_best_match(self, objects, spatial_description, last_bbox, distance_threshold=300):
        (last_x, last_y, last_size) = self.get_center_and_size(last_bbox)
        best_bbox_index = None
        min_angle_error = 10000
        min_dist_error = 10000
        location = spatial_description.spatial_type
        for i, (id, name, bbox) in enumerate(objects):
            (current_x, current_y, current_size) = self.get_center_and_size(bbox)
            distance = math.dist([current_x, current_y], [last_x, last_y])
            angle = self.get_angle([last_x + 10, last_y], [last_x, last_y], [current_x, current_y])
            if location == SpatialType.NEXT_TO or location == SpatialType.OTHER:
                dist_error = self.calculate_error(distance, 0)
                if dist_error < min_dist_error and dist_error < distance_threshold:
                    min_dist_error = dist_error
                    best_bbox_index = i
            elif location == SpatialType.ABOVE or location == SpatialType.TOP_OF:  # Assumes that top left corner of image is (0, 0)
                dist_error = self.calculate_error(distance, 0)
                angle_error = self.calculate_error(angle, 270)  # flipped cos of image coordinates vs cartesian
                if angle_error > 90:  # object is below or next to
                    continue
                elif angle_error < min_angle_error:
                    if dist_error < min_dist_error:
                        best_bbox_index = i
                        min_angle_error = angle_error
                        min_dist_error = dist_error
            elif location == SpatialType.BELOW or location == SpatialType.BOTTOM_OF:  # Assumes that top left corner of image is (0, 0) - for now assume they are equivalent
                dist_error = self.calculate_error(distance, 0)
                angle_error = self.calculate_error(angle, 90)  # flipped cos of image coordinates vs cartesian
                if angle_error > 90:  # object is above or next to
                    continue
                elif angle_error < min_angle_error:
                    if dist_error < min_dist_error:
                        best_bbox_index = i
                        min_angle_error = angle_error
                        min_dist_error = dist_error
            elif location == SpatialType.RIGHT_OF:  # Assumes that top left corner of image is (0, 0)
                dist_error = self.calculate_error(distance, 0)
                angle_error = self.calculate_error(angle, 0)
                if angle_error < 90 or angle_error > 270:  # object is to the right
                    if angle_error < min_angle_error:
                        if dist_error < min_dist_error:
                            best_bbox_index = i
                            min_angle_error = angle_error
                            min_dist_error = dist_error
                else:
                    continue
            elif location == SpatialType.LEFT_OF:  # Assumes that top left corner of image is (0, 0)
                dist_error = self.calculate_error(distance, 0)
                angle_error = self.calculate_error(angle, 180)
                if angle_error > 90:  # object is above or next to
                    continue
                elif angle_error < min_angle_error:
                    if dist_error < min_dist_error:
                        best_bbox_index = i
                        min_angle_error = angle_error
                        min_dist_error = dist_error

        return best_bbox_index

    def find_closest_object_to_location(self, objects, location, distance_threshold=300):
        best_bbox_index = None
        min_dist_error = 100000
        for i, (id, name, bbox) in enumerate(objects):
            (current_x, current_y, current_size) = self.get_center_and_size(bbox)
            x, y = location
            static_distance = math.dist([x, y], [current_x, current_y])
            if x is None:
                # TODO: Error handling for this
                print("WARNING: retrieved location from DB is None")
                continue
            if static_distance < min_dist_error and static_distance < distance_threshold:
                best_bbox_index = i
                min_dist_error = static_distance
        return best_bbox_index

    def get_angle(self, a, b, c):
        ang = math.degrees(math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0]))
        return ang + 360 if ang < 0 else ang

    def calculate_error(self, measured_value, ideal_value):
        return abs(ideal_value-measured_value)

    def get_center_and_size(self, bbox):  # Assumes that bbox is [x1, x2, y1, y2]
        x = bbox[1] - ((bbox[1] - bbox[0]) / 2)
        y = bbox[3] - ((bbox[3] - bbox[2]) / 2)
        diagonal_length = math.dist([bbox[0], bbox[2]], [bbox[1], bbox[3]])
        center = (x, y, diagonal_length)
        return center