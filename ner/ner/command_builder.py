from ner.ner import NER, EntityType
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
    def __init__(self, spatial_type, entities):
        self.spatial_type = spatial_type
        self.object_entity = ObjectEntity().build_object(entities)

    def __str__(self):
        return f"({self.spatial_type}){self.object_entity}"


class ObjectEntity:
    def __init__(self):
        self.name = None
        self.object_descriptors = []
        self.spatial_descriptor = None

    def build_name(self):
        name = ""
        for descriptor in self.object_descriptors:
            name += f"{descriptor} "
        self.name = name.strip()

    def build_object(self, entities):
        for index, (entity_type, word) in enumerate(entities):
            if entity_type == EntityType.COLOUR:
                self.object_descriptors.append(word)
            elif entity_type == EntityType.OBJECT:
                self.object_descriptors.append(word)
            elif entity_type == EntityType.LOCATION:
                spatial_type = SpatialType(word.lower())
                self.spatial_descriptor = SpatialDescription(spatial_type, entities[index+1:])
                break
            elif entity_type == EntityType.TAKE or entity_type == EntityType.FIND:
                break
        self.build_name()
        return self

    def __str__(self):
        output = f"[{self.name}]"
        if self.spatial_descriptor is not None:
            output += f" --> {self.spatial_descriptor}"
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

