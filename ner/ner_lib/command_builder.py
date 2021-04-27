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
        for index, (entity_type, word) in enumerate(entities):
            if entity_type == EntityType.COLOUR or entity_type == EntityType.OBJECT:
                if is_building_own_name:
                    self.object_descriptors.append(word)
                else:
                    self.spatial_descriptions[current_spatial_descriptor].object_entity.object_descriptors.append(word)
            elif entity_type == EntityType.LOCATION:
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
        if current_spatial_descriptor is not None:
            self.spatial_descriptions[current_spatial_descriptor].object_entity.build_name() # build name for last object
        else:
            self.build_name()
        return self

    def __str__(self):
        output = f"[{self.name}]"
        if len(self.spatial_descriptions) > 0:
            for spatial_description in self.spatial_descriptions:
                output += f" --> {spatial_description}"
        return output


class BaseTask:
    def __init__(self):
        self.child_tasks = []
        self.object_to_execute_on = ObjectEntity()
        self.plaintext_name = "base task"

    def build_task(self, entities):
        self.object_to_execute_on.build_object(entities)
        return self

    def __str__(self):
        if len(self.child_tasks) == 0:
            return ""
        output_str = f"Child tasks:\n"
        for task in self.child_tasks:
            output_str += f"\t{task}"
        return output_str


class PickUpTask(BaseTask):
    def __init__(self):
        super().__init__()
        self.plaintext_name = "pick up"

    def build_task(self, entities):
        self.object_to_execute_on.build_object(entities)
        return self

    def get_name(self):
        return PickUpTask.__name__

    def __str__(self):
        return f"Task type: {PickUpTask.__name__}\n\tObject to pick up: {self.object_to_execute_on}\n{super().__str__()}"


class FindTask(BaseTask):
    def __init__(self):
        super().__init__()
        self.plaintext_name = "find"

    def build_task(self, entities):
        self.object_to_execute_on.build_object(entities)
        return self

    def get_name(self):
        return FindTask.__name__

    def __str__(self):
        return f"Task type: {FindTask.__name__}\n\tObject to find: {self.object_to_execute_on}\n{super().__str__()}"


class MoveTask(BaseTask):
    def __init__(self):
        super().__init__()
        self.plaintext_name = "move"

    def build_task(self, entities):
        self.object_to_execute_on.build_object(entities)
        return self

    def get_name(self):
        return MoveTask.__name__

    def __str__(self):
        return f"Task type: {MoveTask.__name__}\n\tObject to move: {self.object_to_execute_on}\n{super().__str__()}"


class PlaceTask(BaseTask):
    def __init__(self):
        super().__init__()
        self.plaintext_name = "place"

    def build_task(self, entities):
        self.object_to_execute_on.build_object(entities)
        return self

    def get_name(self):
        return PlaceTask.__name__

    def __str__(self):
        return f"Task type: {PlaceTask.__name__}\n\tObject to place next to: {self.object_to_execute_on}\n{super().__str__()}"


class CommandBuilder:
    def __init__(self, model_path, tag_path, ner=None):
        self.ner = ner if ner is not None else NER(model_path, tag_path)

    def get_task(self, sentence):
        entities = self.ner.get_entities(sentence)
        task = None
        is_main_task = True
        for index, (entity_type, word) in enumerate(entities):
            if entity_type == EntityType.TASK:
                if not is_main_task:
                    task.child_tasks.append(BaseTask().build_task(entities[index+1:]))
                else:
                    is_main_task = False
                    task = BaseTask().build_task(entities[index+1:])
        return task

