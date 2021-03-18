import numpy as np
from enum import Enum
import math

class Spatial_Relations():
    def locate_specific_object(self, object_entity, objects):
        (id, entity_name, spatial_desc) = object_entity

        last_bbox = []
        last_location = None

        for instance in reversed(spatial_desc):
            object_name = instance.object_entity.name
            location = instance.spatial_type.value

            matching_objects = []

            for x, (name, bbox) in enumerate(objects):
                if name == object_name:
                    matching_objects.append((name, bbox))
            if last_location is None and len(matching_objects)>1:
                raise Exception("Two instances of reference object found, please choose another reference object")
            if len(matching_objects) > 1:
                correct_bbox = self.find_best_match(matching_objects, last_location, last_bbox)
                last_location = location
                last_bbox = correct_bbox
            else:
                (name, bbox) = matching_objects[0]
                last_location = location
                last_bbox = bbox
            objects.remove((object_name, last_bbox))
        matching_objects = []
        for x, (name, bbox) in enumerate(objects):
            if name == entity_name:
                matching_objects.append((name, bbox))
        if len(matching_objects) > 1:
            correct_bbox = self.find_best_match(matching_objects, last_location, last_bbox)
            return entity_name, correct_bbox
        else:
            (name, correct_bbox) = matching_objects[0]
            return entity_name, correct_bbox
        print("Couldn't find object: ", entity_name)

    def find_best_match(self, objects, location, last_bbox):
        (last_x, last_y, last_size) = self.get_center_and_size(last_bbox)
        best_bbox = None
        min_angle_error = None
        min_dist_error = None
        for i, (name, bbox) in enumerate(objects):
            (current_x, current_y, current_size) = self.get_center_and_size(bbox)
            distance = math.dist([current_x, current_y], [last_x, last_y])
            angle = self.get_angle([last_x + 10, last_y], [last_x, last_y], [current_x, current_y])
            if location == "next":
                dist_error = self.calculate_error(distance, 0)
                if min_dist_error is None or dist_error < min_dist_error:
                    min_dist_error = dist_error
                    best_bbox = bbox

            elif location == "above":  # Assumes that top left corner of image is (0, 0)
                dist_error = self.calculate_error(distance, 0)
                angle_error = self.calculate_error(angle, 270)  # flipped cos of image coordinates vs cartesian
                if angle_error > 90:  # object is below or next to
                    continue
                elif min_angle_error is None or angle_error < min_angle_error:
                    if min_dist_error is None or dist_error < min_dist_error:
                        best_bbox = bbox
                        min_angle_error = angle_error
                        min_dist_error = dist_error

            elif location == "below":  # Assumes that top left corner of image is (0, 0)
                dist_error = self.calculate_error(distance, 0)
                angle_error = self.calculate_error(angle, 90)  # flipped cos of image coordinates vs cartesian
                if angle_error > 90:  # object is above or next to
                    continue
                elif min_angle_error is None or angle_error < min_angle_error:
                    if min_dist_error is None or dist_error < min_dist_error:
                        best_bbox = bbox
                        min_angle_error = angle_error
                        min_dist_error = dist_error

            elif location == "right":  # Assumes that top left corner of image is (0, 0)
                dist_error = self.calculate_error(distance, 0)
                angle_error = self.calculate_error(angle, 0)
                if angle_error < 90 or angle_error > 270:  # object is to the right
                    if min_angle_error is None or angle_error < min_angle_error:
                        if min_dist_error is None or dist_error < min_dist_error:
                            best_bbox = bbox
                            min_angle_error = angle_error
                            min_dist_error = dist_error
                else:
                    continue

            elif location == "left":  # Assumes that top left corner of image is (0, 0)
                dist_error = self.calculate_error(distance, 0)
                angle_error = self.calculate_error(angle, 180)
                if angle_error > 90:  # object is above or next to
                    continue
                elif min_angle_error is None or angle_error < min_angle_error:
                    if min_dist_error is None or dist_error < min_dist_error:
                        best_bbox = bbox
                        min_angle_error = angle_error
                        min_dist_error = dist_error

            elif location == "bottom":
                return True

            elif location == "top":
                return True

        return best_bbox

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

    def calculate_dist(self):
        return

    def calculate_angle(self):
        return