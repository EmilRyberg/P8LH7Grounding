from database_handler.database_handler import DatabaseHandler
from task_grounding.task_grounding.task_definitions import PickUpTask, FindTask, PlaceTask, MoveTask
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

class TaskGrounding:
    def __init__(self, db=DatabaseHandler("../../dialog_flow/nodes/grounding.db")):
        self.db = db
        self.return_object = TaskGroundingReturn()
        self.task_info = []

    def get_task_from_entity(self, ner_task_word, object_entity=None):
        (_, task_name, sub_tasks) = self.db.get_task(ner_task_word)
        if task_name is None:
            self.return_object.error_code = ErrorType.UNKNOWN
            return self.return_object
        elif task_name == "pick up":  # Check the default skills first.
            self.handle_pick_task(object_entity)
        elif task_name == "find":
            self.handle_find_task(object_entity)
        elif task_name == "move":
            self.handle_move_task(object_entity)
        elif task_name == "place":
            self.handle_place_task()
        else:
            tasks = []

        return self.return_object

    def handle_pick_task(self, object_entity, is_sub_task=False):
        if not is_sub_task:
            task = PickUpTask()
            if object_entity is None:
                self.task_info.append(task)
                self.return_object.error_code = ErrorType.NO_OBJECT
                return self.return_object
            task.object_to_pick_up = object_entity
            self.return_object.is_success = True
            self.return_object.task_info.append(task)

    def handle_move_task(self, object_entity, is_sub_task=False):
        if not is_sub_task:
            task = MoveTask()
            if object_entity is None:
                self.task_info.append(task)
                self.return_object.error_code = ErrorType.NO_OBJECT
                return self.return_object
            task.object_to_move = object_entity
            self.return_object.is_success = True
            self.return_object.task_info.append(task)

    def handle_find_task(self, object_entity, is_sub_task=False):
        if not is_sub_task:
            task = FindTask()
            if object_entity is None:
                self.task_info.append(task)
                self.return_object.error_code = ErrorType.NO_OBJECT
                return self.return_object
            task.object_to_find = object_entity
            self.return_object.is_success = True
            self.return_object.task_info.append(task)

    def handle_place_task(self, is_sub_task=False):
        if not is_sub_task:
            task = PlaceTask()
            self.return_object.is_success = True
            self.return_object.task_info.append(task)

    def teach_new_task(self, task_name, sub_tasks, words):
        self.db.add_task(task_name, words)
        for task in sub_tasks:
            (sub_task_id, _, _) = self.db.get_task(task)
            if sub_task_id is None:
                self.return_object.error_code = ErrorType.UNKNOWN
                return self.return_object
            self.db.add_sub_task(task_name, sub_task_id)
        self.return_object.is_success = True
        return self.return_object

    def add_word_to_task(self, task_name, word_to_add):
        task_id = self.db.get_task_id(task_name)
        self.db.add_word_to_task(task_id, word_to_add)
        self.return_object.is_success = True
        return self.return_object


if __name__ == "__main__":
    task_grounding = TaskGrounding()
    task_grounding.teach_new_task("clear table", ["get", "move", "place"], ["clear", "clean", "tidy"])
