import numpy as np
from enum import Enum
import math

class Spatial_Relations():
    def locate_specific_object(self, object_entity, objects):
        (id, entity_name, spatial_desc) = object_entity
        last_bbox = []
        last_location = None
        is_first_run = True

        correct_object = False

        for instance in reversed(spatial_desc):
            object = instance.object_entity.name
            location = instance.spatial_type.value
            for x, (name, bbox) in enumerate(objects):
                if name == object:
                    if is_first_run:  # Means it is the object with which everything is related
                        correct_object = True
                        is_first_run = False
                    else:
                        correct_object = self.match_bounding_boxes(last_location, bbox, last_bbox)
                    if correct_object:
                        correct_object = False
                        last_location = location
                        last_bbox = bbox
                        break
        for x, (name, bbox) in enumerate(objects):
            if name == entity_name:
                correct_object = self.match_bounding_boxes(last_location, bbox, last_bbox)
                if correct_object:
                    return name, bbox


    def match_bounding_boxes(self, location, bbox, last_bbox):
        (current_x, current_y, current_size) = self.get_center_and_size(bbox)
        (last_x, last_y, last_size) = self.get_center_and_size(last_bbox)

        if location == "next":
            if math.sqrt((current_x-last_x) ** 2+(current_y-last_y) ** 2) < (last_size+current_size)*2:  # TODO Tune dis
                return True
            else:
                return False

        elif location == "above":  # Assumes that top left corner of image is (0, 0)
            if current_y < last_y+last_size and math.sqrt((current_x-last_x) ** 2) < current_size*2:
                return True
            else:
                return False

        elif location == "below":  # Assumes that top left corner of image is (0, 0)
            if current_y > last_y - last_size and math.sqrt((current_x - last_x) ** 2) < current_size * 2:
                return True
            else:
                return False

        elif location == "right":  # Assumes that top left corner of image is (0, 0)
            if current_x > last_x + last_size and math.sqrt((current_y - last_y) ** 2) < current_size * 2:
                return True
            else:
                return False

        elif location == "left":  # Assumes that top left corner of image is (0, 0)
            if current_x < last_x - last_size and math.sqrt((current_y - last_y) ** 2) < current_size * 2:
                return True
            else:
                return False

        elif location == "bottom":
            return True

        elif location == "top":
            return True

    def get_center_and_size(self, bbox):  # Assumes that bbox is [x1, x2, y1, y2]
        x = bbox[1] - ((bbox[1]-bbox[0])/2)
        y = bbox[3] - ((bbox[3] - bbox[2]) / 2)
        diagonal_length = math.sqrt((bbox[1]-bbox[2]) ** 2+(bbox[3]-bbox[2]) ** 2)
        center = (x,y, diagonal_length)
        return center