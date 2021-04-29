from ner_lib.ner import NER, EntityType
from enum import Enum


class SpatialType(Enum):
    NEXT_TO = "next"
    ABOVE = "above"
    BELOW = "below"
    RIGHT_OF = "right"
    LEFT_OF = "left"
    BOTTOM_OF = "bottom"
    TOP_OF = "top"
    OTHER = "other"


class TaskType(Enum):
    PLACE = "place"
    PICK = "pick"
    FIND = "find"
    MOVE = "move"
    OTHER = "other"
    NOT_SET = "not_set"

WORD_TO_SPATIAL_TYPE_MAPPING = { # Maybe add this to database
    "next": SpatialType.NEXT_TO,
    "next to": SpatialType.NEXT_TO,
    "above": SpatialType.ABOVE,
    "above of": SpatialType.ABOVE,
    "right": SpatialType.RIGHT_OF,
    "right of": SpatialType.RIGHT_OF,
    "left": SpatialType.LEFT_OF,
    "left of": SpatialType.LEFT_OF,
    "bottom": SpatialType.BOTTOM_OF,
    "bottom of": SpatialType.BOTTOM_OF,
    "below": SpatialType.BOTTOM_OF,
    "below of": SpatialType.BELOW,
    "top": SpatialType.TOP_OF,
    "top of": SpatialType.TOP_OF
}


class SpatialDescription:
    def __init__(self, spatial_type):
        self.spatial_type = spatial_type
        self.object_entity = ObjectEntity()

    def __str__(self):
        return f"({self.spatial_type}){self.object_entity}"

    def get_sub_descriptions(self):
        return [self].extend(self.object_entity.spatial_descriptions)


class ObjectEntity:
    def __init__(self, name=None):
        self.name = name
        self.object_descriptors = []
        self.spatial_descriptions = []

    def build_name(self):
        name = ""
        for descriptor in self.object_descriptors:
            name += f"{descriptor} "
        self.name = name.strip()

    def build_object(self, entities):
        is_building_own_name = True
        current_spatial_descriptor = None
        has_encountered_object_entity = False  # This flag is used to switch out of building object,
                                               # if a new object is found that is not a spatial description
        index_offset = 0
        is_object = False
        stop_building_objects = False
        for index, (entity_type, word) in enumerate(entities):
            index_offset = index
            if entity_type == EntityType.COLOUR or entity_type == EntityType.OBJECT:
                if has_encountered_object_entity:
                    break  # second item
                if entity_type == EntityType.OBJECT:
                    has_encountered_object_entity = True
                if is_building_own_name:
                    is_object = True
                    self.object_descriptors.append(word)
                elif self.spatial_descriptions[current_spatial_descriptor].spatial_type != SpatialType.OTHER:
                    self.spatial_descriptions[current_spatial_descriptor].object_entity.object_descriptors.append(word)
            elif entity_type == EntityType.LOCATION:
                has_encountered_object_entity = False
                spatial_type = SpatialType.OTHER
                if word.lower() in WORD_TO_SPATIAL_TYPE_MAPPING.keys():
                    spatial_type = WORD_TO_SPATIAL_TYPE_MAPPING[word.lower()]
                spatial_description = SpatialDescription(spatial_type)
                if is_building_own_name:
                    is_building_own_name = False
                    self.build_name()
                    current_spatial_descriptor = 0
                else:
                    self.spatial_descriptions[current_spatial_descriptor].object_entity.build_name()
                    current_spatial_descriptor += 1
                if spatial_type == SpatialType.OTHER:
                    spatial_description.object_entity.object_descriptors.append(word)
                self.spatial_descriptions.append(spatial_description)
            elif entity_type == EntityType.TASK:
                break
            if index == len(entities) - 1:
                stop_building_objects = True
        if current_spatial_descriptor is not None:
            self.spatial_descriptions[current_spatial_descriptor].object_entity.build_name() # build name for last object
        else:
            self.build_name()
        if not is_object:
            stop_building_objects = True
        return is_object, stop_building_objects, index_offset

    def __str__(self):
        output = f"[{self.name}]"
        if len(self.spatial_descriptions) > 0:
            for spatial_description in self.spatial_descriptions:
                output += f" --> {spatial_description}"
        return output


class Task:
    def __init__(self, name=None):
        self.child_tasks = []
        self.name = name
        self.task_type = TaskType.NOT_SET  # this will be set in task grounding
        self.objects_to_execute_on = []

    def build_task(self, entities):
        stop_building = False
        index_offset = 0
        while not stop_building:
            new_object = ObjectEntity()
            is_object, stop_building, local_index_offset = new_object.build_object(entities[index_offset:])
            index_offset += local_index_offset
            if is_object:
                self.objects_to_execute_on.append(new_object)
        return self

    def __str__(self):
        if len(self.child_tasks) == 0:
            return ""
        output_str = f"Task type: {self.task_type.name}\n\t" \
                     f"Objects to handle: {self.objects_to_execute_on}\n" \
                     f"Child tasks:\n"
        for task in self.child_tasks:
            output_str += f"\t{task}"
        return output_str


class CommandBuilder:
    def __init__(self, ner):
        self.ner = ner

    def get_task(self, sentence):
        entities = self.ner.get_entities(sentence)
        task = None
        is_main_task = True
        for index, (entity_type, word) in enumerate(entities):
            if entity_type == EntityType.TASK:
                if not is_main_task:
                    task.child_tasks.append(Task(word).build_task(entities[index + 1:]))
                else:
                    is_main_task = False
                    task = Task(word).build_task(entities[index + 1:])
        return task

