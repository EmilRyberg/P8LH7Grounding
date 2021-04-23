from database_handler import DatabaseHandler
from ner_lib.ner import NER, EntityType

from ner_lib.command_builder import CommandBuilder, PlaceTask, PickUpTask, FindTask, MoveTask
import numpy as np
from typing import Optional
from enum import Enum


class ErrorType(Enum):
    UNKNOWN = "unknown word"
    NO_OBJECT = "no object specified when required"


class TaskGroundingReturn:
    def __init__(self):
        self.is_success = False
        self.error_code = None
        self.task_info = []
        self.error_task = None


class TaskGrounding:
    def __init__(self, db=DatabaseHandler("../../dialog_flow/nodes/grounding.db")):
        self.db = db
        self.return_object = TaskGroundingReturn()

    def get_task_from_entity(self, ner_task_word, task_entities=None):
        (_, task_name) = self.db.get_task(ner_task_word)
        self.task_switch(task_name, task_entities, ner_task_word)
        return self.return_object

    def task_switch(self, task_name, task_entities, ner_word):
        if task_name is None:
            self.unknown_task_error(ner_word)
            return
        elif task_name == "pick up":  # Check the default skills first.
            self.handle_pick_task(task_entities)
        elif task_name == "find":
            self.handle_find_task(task_entities)
        elif task_name == "move":
            self.handle_move_task(task_entities)
        elif task_name == "place":
            self.handle_place_task(task_entities)
        else:
            sub_tasks = self.db.get_sub_tasks(task_name)
            if sub_tasks is not None:
                sub_tasks = np.fromstring(sub_tasks, dtype=int, sep=',')
            self.handle_advanced_task(sub_tasks, task_entities, ner_word)

    def handle_pick_task(self, task_entities):
        task = PickUpTask()
        if task_entities is None:
            self.missing_entities_error(task)
            return
        task.build_task(task_entities)
        self.return_object.is_success = True
        self.return_object.task_info.append(task)

    def handle_move_task(self, task_entities):
        task = MoveTask()
        if task_entities is None:
            self.missing_entities_error(task)
            return
        task.build_task(task_entities)
        self.return_object.is_success = True
        self.return_object.task_info.append(task)

    def handle_find_task(self, task_entities):
        task = FindTask()
        if task_entities is None:
            self.missing_entities_error(task)
            return
        task.build_task(task_entities)
        self.return_object.is_success = True
        self.return_object.task_info.append(task)

    def handle_place_task(self, task_entities):
        task = PlaceTask()
        if task_entities is None:
            self.missing_entities_error(task)
            return
        task.build_task(task_entities)
        self.return_object.is_success = True
        self.return_object.task_info.append(task)

    def handle_advanced_task(self, sub_tasks, task_entities, ner_word):
        for sub_task in sub_tasks:  # task idx
            task_name = self.db.get_task_name(int(sub_task))
            self.task_switch(task_name, task_entities, ner_word)

    def teach_new_task(self, task_name, sub_tasks, words):
        self.db.add_task(task_name, words)
        for task in sub_tasks:
            (sub_task_id, _) = self.db.get_task(task)
            if sub_task_id is None:
                self.unknown_task_error(task)
                return self.return_object
            self.db.add_sub_task(task_name, sub_task_id)
        self.return_object.is_success = True
        return self.return_object

    def add_word_to_task(self, task_word, word_to_add):
        (_, task_name) = self.db.get_task(task_word)
        if task_name is None:
            self.unknown_task_error(task_word)
            return self.return_object
        task_id = self.db.get_task_id(task_name)
        self.db.add_word_to_task(task_id, word_to_add)
        self.return_object.is_success = True
        return self.return_object

    def add_sub_task(self, task_word, sub_tasks):
        (_, task_name) = self.db.get_task(task_word)
        if task_name is None:
            self.unknown_task_error(task_word)
            self.return_object.error_code = ErrorType.UNKNOWN
            self.return_object.error_task = task_word
            return self.return_object
        for task in sub_tasks:
            (sub_task_id, _) = self.db.get_task(task)
            if sub_task_id is None:
                self.unknown_task_error(task)
                return self.return_object
            self.db.add_sub_task(task_name, sub_task_id)
        self.return_object.is_success = True
        return self.return_object

    def missing_entities_error(self, task):
        self.return_object.task_info.append(task)
        self.return_object.error_code = ErrorType.NO_OBJECT
        return self.return_object

    def unknown_task_error(self, task_word):
        self.return_object.error_code = ErrorType.UNKNOWN
        self.return_object.error_task = task_word
        return self.return_object


if __name__ == "__main__":
    task_grounding = TaskGrounding()
    entities = [
        (EntityType.COLOUR, "blue"),
        (EntityType.OBJECT, "cover"),
        (EntityType.LOCATION, "next"),
        (EntityType.COLOUR, "black"),
        (EntityType.OBJECT, "bottom cover"),
        (EntityType.LOCATION, "above"),
        (EntityType.OBJECT, "bottom cover")
    ]
    task = task_grounding.get_task_from_entity("get", entities)
    print(task)