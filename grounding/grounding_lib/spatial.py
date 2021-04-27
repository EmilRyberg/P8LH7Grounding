import numpy as np
from enum import Enum
import copy
import math
from ner_lib.command_builder import SpatialType
from database_handler.database_handler import DatabaseHandler


class StatusEnum(Enum):
    SUCCESS = "SUCCESS"
    ERROR_TWO_REF = "ERROR_TWO_REF"
    ERROR_CANT_FIND = "ERROR_CANT_FIND"
    STATIC_LOCATION = "STATIC_LOCATION"


class SpatialRelation:
    def __init__(self, database_handler: DatabaseHandler):
        self.database_handler = database_handler

    def locate_specific_object(self, object_entity, objects):
        local_objects = copy.deepcopy(objects)
        entity_name = object_entity.name
        spatial_desc = object_entity.spatial_descriptions
        last_bbox = []
        last_spatial_description = None

        for instance in reversed(spatial_desc):
            object_name = instance.object_entity.name
            matching_objects = []
            if instance.spatial_type != SpatialType.OTHER:
                for (id, name, bbox) in local_objects:
                    if name == object_name:
                        matching_objects.append((id, name, bbox))
                if last_spatial_description is None and len(matching_objects) > 1:
                    return None, StatusEnum.ERROR_TWO_REF
                if len(matching_objects) > 1:
                    correct_bbox_index = self.find_best_match(matching_objects, last_spatial_description, last_bbox)
                    (last_id, _, last_bbox) = matching_objects[correct_bbox_index]
                    last_spatial_description = instance
                else:
                    (id, name, bbox) = matching_objects[0]
                    last_spatial_description = instance
                    last_id = id
                    last_bbox = bbox
                local_objects.remove((last_id, object_name, last_bbox))
            else:
                x, y, z = self.database_handler.get_location_by_name(instance.object_entity.name.lower())
                if x is None:
                    print("WARNING: static location is None")
                    continue
                last_bbox = [x - 1, x + 1, y - 1, y + 1]
                last_spatial_description = instance

        matching_objects = []
        for x, (id, name, bbox) in enumerate(local_objects):
            if name == entity_name:
                matching_objects.append((id, name, bbox))
        if len(matching_objects) > 1:
            if last_spatial_description.spatial_type == SpatialType.OTHER:
                x, y, z = self.database_handler.get_location_by_name(
                    last_spatial_description.object_entity.name.lower())
                correct_bbox_index = self.find_closest_object_to_location(local_objects, (x, y))
            else:
                correct_bbox_index = self.find_best_match(matching_objects, last_spatial_description, last_bbox)
            (id, name, bbox) = matching_objects[correct_bbox_index]
            correct_index = id
            return correct_index, StatusEnum.SUCCESS
        elif matching_objects:
            (id, name, bbox) = matching_objects[0]
            correct_index = id
            return correct_index, StatusEnum.SUCCESS
        else:
            return None, StatusEnum.ERROR_CANT_FIND

    def locate_last_object_in_spatial_descriptions(self, spatial_descriptions, objects):
        local_objects = copy.deepcopy(objects)
        last_bbox = []
        last_spatial_description = None

        for instance in reversed(spatial_descriptions):
            object_name = instance.object_entity.name
            matching_objects = []
            if instance.spatial_type != SpatialType.OTHER:
                for (id, name, bbox) in local_objects:
                    if name == object_name:
                        matching_objects.append((id, name, bbox))
                if last_spatial_description is None and len(matching_objects) > 1:
                    return None, StatusEnum.ERROR_TWO_REF
                if len(matching_objects) > 1:
                    correct_bbox_index = self.find_best_match(matching_objects, last_spatial_description, last_bbox)
                    (last_id, _, last_bbox) = matching_objects[correct_bbox_index]
                    last_spatial_description = instance
                else:
                    (id, name, bbox) = matching_objects[0]
                    last_spatial_description = instance
                    last_bbox = bbox
            else:
                x, y, z = self.database_handler.get_location_by_name(instance.object_entity.name.lower())
                if x is None:
                    print("WARNING: static location is None")
                    continue
                last_bbox = [x - 1, x + 1, y - 1, y + 1]
                last_spatial_description = instance

        matching_objects = []
        if last_spatial_description.spatial_type == SpatialType.OTHER:
            x, y, z = self.database_handler.get_location_by_name(last_spatial_description.object_entity.name.lower())
            correct_bbox_index = self.find_closest_object_to_location(local_objects, (x, y))
            (idx, name, bbox) = matching_objects[correct_bbox_index]
            return idx, StatusEnum.SUCCESS
        else:
            for x, (idx, name, bbox) in enumerate(local_objects):
                if name == last_spatial_description.object_entity.name:
                    matching_objects.append((idx, name, bbox))
            if len(matching_objects) > 1:
                center_x, center_y, _ = self.get_center_and_size(last_bbox)
                correct_bbox_index = self.find_closest_object_to_location(matching_objects, (center_x, center_y))
                (idx, name, bbox) = matching_objects[correct_bbox_index]
                return idx, StatusEnum.SUCCESS
            elif matching_objects:
                (idx, name, bbox) = matching_objects[0]
                return idx, StatusEnum.SUCCESS
            else:
                return None, StatusEnum.ERROR_CANT_FIND

    def get_location(self, spatial_descriptions, objects):
        if len(spatial_descriptions) == 1 and spatial_descriptions[0].spatial_type == SpatialType.OTHER:
            x, y, z = self.database_handler.get_location_by_name(spatial_descriptions[0].object_entity.name.lower())
            if x is None:
                print(f"WARNING: x for location '{spatial_descriptions[0].object_entity.name.lower()}' is None")
                return None
            return x, y

        best_object_index, status = self.locate_last_object_in_spatial_descriptions(spatial_descriptions, objects)
        if status != StatusEnum.SUCCESS:
            return None, status
        else:
            idx, name, bbox = objects[best_object_index]
            center_x, center_y, _ = self.get_center_and_size(bbox)
            return center_x, center_y


    def find_best_match(self, objects, spatial_description, last_bbox):
        (last_x, last_y, last_size) = self.get_center_and_size(last_bbox)
        best_bbox_index = None
        min_angle_error = None
        min_dist_error = None
        location = spatial_description.spatial_type
        for i, (id, name, bbox) in enumerate(objects):
            (current_x, current_y, current_size) = self.get_center_and_size(bbox)
            distance = math.dist([current_x, current_y], [last_x, last_y])
            angle = self.get_angle([last_x + 10, last_y], [last_x, last_y], [current_x, current_y])
            if location == SpatialType.NEXT_TO or location == SpatialType.OTHER:
                dist_error = self.calculate_error(distance, 0)
                if min_dist_error is None or dist_error < min_dist_error:
                    min_dist_error = dist_error
                    best_bbox_index = i
            elif location == SpatialType.ABOVE or location == SpatialType.TOP_OF:  # Assumes that top left corner of image is (0, 0)
                dist_error = self.calculate_error(distance, 0)
                angle_error = self.calculate_error(angle, 270)  # flipped cos of image coordinates vs cartesian
                if angle_error > 90:  # object is below or next to
                    continue
                elif min_angle_error is None or angle_error < min_angle_error:
                    if min_dist_error is None or dist_error < min_dist_error:
                        best_bbox_index = i
                        min_angle_error = angle_error
                        min_dist_error = dist_error
            elif location == SpatialType.BELOW or location == SpatialType.BOTTOM_OF:  # Assumes that top left corner of image is (0, 0) - for now assume they are equivalent
                dist_error = self.calculate_error(distance, 0)
                angle_error = self.calculate_error(angle, 90)  # flipped cos of image coordinates vs cartesian
                if angle_error > 90:  # object is above or next to
                    continue
                elif min_angle_error is None or angle_error < min_angle_error:
                    if min_dist_error is None or dist_error < min_dist_error:
                        best_bbox_index = i
                        min_angle_error = angle_error
                        min_dist_error = dist_error
            elif location == SpatialType.RIGHT_OF:  # Assumes that top left corner of image is (0, 0)
                dist_error = self.calculate_error(distance, 0)
                angle_error = self.calculate_error(angle, 0)
                if angle_error < 90 or angle_error > 270:  # object is to the right
                    if min_angle_error is None or angle_error < min_angle_error:
                        if min_dist_error is None or dist_error < min_dist_error:
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
                elif min_angle_error is None or angle_error < min_angle_error:
                    if min_dist_error is None or dist_error < min_dist_error:
                        best_bbox_index = i
                        min_angle_error = angle_error
                        min_dist_error = dist_error

        return best_bbox_index

    def find_closest_object_to_location(self, objects, location):
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
            if static_distance < min_dist_error:
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