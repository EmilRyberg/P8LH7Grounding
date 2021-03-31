import numpy as np
from enum import Enum
import copy
import math
from ner_lib.command_builder import SpatialType


class SpatialRelation:
    def locate_specific_object(self, object_entity, objects):
        local_objects = copy.deepcopy(objects)
        entity_name = object_entity.name
        spatial_desc = object_entity.spatial_descriptions
        last_bbox = []
        last_location = None

        for instance in reversed(spatial_desc):
            object_name = instance.object_entity.name
            location = instance.spatial_type
            matching_objects = []

            for (id, name, bbox) in local_objects:
                if name == object_name:
                    matching_objects.append((id, name, bbox))
            if last_location is None and len(matching_objects) > 1:
                return -1
            if len(matching_objects) > 1:
                correct_bbox = self.find_best_match(matching_objects, last_location, last_bbox)
                (last_id, _, last_bbox) = matching_objects[correct_bbox]
                last_location = location
            else:
                (id, name, bbox) = matching_objects[0]
                last_location = location
                last_bbox = bbox
                last_id = id
            local_objects.remove((last_id, object_name, last_bbox))
        matching_objects = []
        for x, (id, name, bbox) in enumerate(local_objects):
            if name == entity_name:
                matching_objects.append((id, name, bbox))
        if len(matching_objects) > 1:
            correct_bbox = self.find_best_match(matching_objects, last_location, last_bbox)
            (id, name, bbox) = matching_objects[correct_bbox]
            correct_index = id
            return correct_index
        elif matching_objects:
            (id, name, bbox) = matching_objects[0]
            correct_index = id
            return correct_index
        else:
            return -2

    def find_best_match(self, objects, location, last_bbox):
        (last_x, last_y, last_size) = self.get_center_and_size(last_bbox)
        best_bbox_index = None
        min_angle_error = None
        min_dist_error = None
        for i, (id, name, bbox) in enumerate(objects):
            (current_x, current_y, current_size) = self.get_center_and_size(bbox)
            distance = math.dist([current_x, current_y], [last_x, last_y])
            angle = self.get_angle([last_x + 10, last_y], [last_x, last_y], [current_x, current_y])
            if location == SpatialType.NEXT_TO:
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

    def get_angle(self, a, b, c):
        ang = math.degrees(math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0]))
        return ang + 360 if ang < 0 else ang

    def calculate_error(self, measured_value, ideal_value):
        return abs(ideal_value-measured_value)

    def get_center_and_size(self, bbox):  # Assumes that bbox is [x1, x2, y1, y2]
        x = bbox[1] - ((bbox[1]-bbox[0])/2)
        y = bbox[3] - ((bbox[3] - bbox[2]) / 2)
        diagonal_length = math.dist([bbox[0], bbox[2]], [bbox[1],bbox[3]])
        center = (x,y, diagonal_length)
        return center