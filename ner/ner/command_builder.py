from ner.ner.ner import NER, EntityType
from enum import Enum


class SpatialType(Enum):
    NEXT_TO = "next"
    ABOVE = "above"
    BELOW = "below"
    RIGHT_OF = "right"
    LEFT_OF = "left"
    BOTTOM_OF = "bottom"
    TOP_OF = "top"


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
        self.name = None
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
                spatial_type = SpatialType(word.lower())
                if is_building_own_name:
                    is_building_own_name = False
                    self.build_name()
                    current_spatial_descriptor = 0
                else:
                    self.spatial_descriptions[current_spatial_descriptor].object_entity.build_name()
                    current_spatial_descriptor += 1
                self.spatial_descriptions.append(SpatialDescription(spatial_type))
            elif entity_type == EntityType.TAKE or entity_type == EntityType.FIND:
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
        self.object_to_pick_up = ObjectEntity()

    def build_task(self, entities):
        self.object_to_pick_up.build_object(entities)
        return self

    def __str__(self):
        return f"Task type: {PickUpTask.__name__}\n\tObject to pick up: {self.object_to_pick_up}\n{super().__str__()}"


class FindTask(BaseTask):
    def __init__(self):
        super().__init__()
        self.object_to_find = ObjectEntity()

    def build_task(self, entities):
        self.object_to_find.build_object(entities)
        return self

    def __str__(self):
        return f"Task type: {FindTask.__name__}\n\tObject to find: {self.object_to_find}\n{super().__str__()}"


class CommandBuilder:
    def __init__(self, model_path, tag_path, ner=None):
        self.ner = ner if ner is not None else NER(model_path, tag_path)

    def get_task(self, sentence):
        entities = self.ner.get_entities(sentence)
        task = None
        task_type = None
        for index, (entity_type, word) in enumerate(entities):
            if entity_type == EntityType.TAKE:
                if task_type is not None:
                    task.child_tasks.append(PickUpTask().build_task(entities[index+1:]))
                else:
                    task_type = EntityType.TAKE
                    task = PickUpTask().build_task(entities[index+1:])
            elif entity_type == EntityType.FIND:
                if task_type is not None:
                    task.child_tasks.append(FindTask().build_task(entities[index+1:]))
                else:
                    task_type = EntityType.FIND
                    task = FindTask().build_task(entities[index+1:])
        return task

