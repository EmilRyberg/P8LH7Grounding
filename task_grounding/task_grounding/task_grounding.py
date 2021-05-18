from database_handler.database_handler import DatabaseHandler
from ner_lib.ner import EntityType

from ner_lib.command_builder import CommandBuilder, ObjectEntity, Task, TaskType
import numpy as np
from enum import Enum


class TaskErrorType(Enum):
    UNKNOWN = "unknown word"
    NO_OBJECT = "no object specified when required"
    NO_SUBTASKS = "no specified subtasks"
    NO_SPATIAL = "no spatial specified"
    ALREADY_USED_WORD = "word already used in other task"
    ALREADY_KNOWN_TASK = "I already know this task"


class TaskGroundingError:
    def __init__(self):
        self.error_task_name = None
        self.error_code = None


class TaskGroundingReturn:
    def __init__(self):
        self.is_success = False
        self.task_info = []
        self.error = None


class TaskGrounding:
    def __init__(self, db=DatabaseHandler("../../dialog_flow/nodes/grounding.db")):
        self.db = db
        self.return_object = TaskGroundingReturn()

    def get_specific_task_from_task(self, task: Task):
        task_name_lower = task.name.lower().replace(".", "")
        (task_id, task_name) = self.db.get_task(task_name_lower)
        return_object = TaskGroundingReturn()
        tasks, error = self.task_switch(task_id, task_name, task)
        if error:
            return_object.error = error
            return_object.is_success = False
        else:
            return_object.is_success = True
            return_object.task_info = tasks
            if len(task.child_tasks) > 0:
                for child_task in task.child_tasks:
                    child_task_name_lower = child_task.name.lower().replace(".", "")
                    child_task_id, child_task_name = self.db.get_task(child_task_name_lower)
                    child_sub_tasks, error = self.task_switch(child_task_id, child_task_name, child_task)
                    if error:
                        return_object.error = error
                        return_object.is_success = False
                        break
                    else:
                        tasks.extend(child_sub_tasks)
        return return_object

    def task_switch(self, task_id, task_name, task: Task):
        tasks = []
        if task_name is None:
            return None, self.unknown_task_error(task.name)
        elif task_name == "pick up":  # Check the default skills first.
            task.task_type = TaskType.PICK
            tasks.append(task)
            if len(task.objects_to_execute_on) == 0:
                error = self.missing_entities_error(task.name)
                return tasks, error
        elif task_name == "find":
            task.task_type = TaskType.FIND
            tasks.append(task)
            if len(task.objects_to_execute_on) == 0:
                error = self.missing_entities_error(task.name)
                return tasks, error
        elif task_name == "move":
            task.task_type = TaskType.MOVE
            tasks.append(task)
            if len(task.objects_to_execute_on) == 0:
                error = self.missing_entities_error(task.name)
                return tasks, error
        elif task_name == "place":
            task.task_type = TaskType.PLACE
            tasks.append(task)
            if len(task.objects_to_execute_on) == 0:
                error = self.missing_entities_error(task.name)
                return tasks, error
        else:
            sub_tasks = self.db.get_sub_tasks(task_id)
            if sub_tasks is None:
                error = TaskGroundingError()
                error.error_task_name = task_name
                error.error_code = TaskErrorType.NO_SUBTASKS
                return None, error
            tasks = self.handle_custom_task(sub_tasks)
        return tasks, None

    def handle_custom_task(self, sub_tasks):
        tasks = []
        for i in range(len(sub_tasks[0])):  # task idx
            task_id = sub_tasks[0][i]
            task_name = sub_tasks[1][i]
            db_task = sub_tasks[2][i]
            task = Task()
            if db_task is not None:
                task.name = task_name
                task.objects_to_execute_on = db_task.objects_to_execute_on
                task.child_tasks = db_task.child_tasks
            sub_task_tasks, error = self.task_switch(task_id, task_name, task)
            tasks.extend(sub_task_tasks)
        return tasks

    def teach_new_task(self, task_name, sub_tasks, words):
        task_name_lower = task_name.lower().replace(".", "")
        task_exists = self.db.get_task_id(task_name_lower)
        return_object = TaskGroundingReturn()
        if task_exists:
            error = TaskGroundingError()
            error.error_task_name = task_name_lower
            error.error_code = TaskErrorType.ALREADY_KNOWN_TASK
            return_object.error = error
            return return_object
        words_lower = [x.lower().replace(".", "") for x in words]
        task_id, error_words = self.db.add_task(task_name_lower, words_lower)
        for task in sub_tasks:
            sub_task_name_lower = task.name.lower().replace(".", "")
            (sub_task_id, _) = self.db.get_task(sub_task_name_lower)
            if sub_task_id is None:
                error = self.unknown_task_error(sub_task_name_lower)
                return_object.error = error
                return return_object
            self.db.add_sub_task(task_id, sub_task_id, task)
        return_object.is_success = True
        if error_words:
            return_object.is_success = False
            error = TaskGroundingError()
            error.error_task_name = ", ".join(error_words)
            error.error_code = TaskErrorType.ALREADY_USED_WORD
            return_object.error = error
        return return_object

    def add_word_to_task(self, task_word, word_to_add):
        (_, task_name) = self.db.get_task(task_word)
        return_object = TaskGroundingReturn()
        if task_name is None:
            error = self.unknown_task_error(task_word)
            return_object.error = error
            return return_object
        task_id = self.db.get_task_id(task_name)
        self.db.add_word_to_task(task_id, word_to_add)
        return_object.is_success = True
        return return_object

    def add_sub_task(self, task_word, sub_task_words):
        (task_id, task_name) = self.db.get_task(task_word)
        return_object = TaskGroundingReturn()
        if task_name is None:
            error = self.unknown_task_error(task_word)
            return_object = error
            return return_object
        for task in sub_task_words:
            (sub_task_id, _) = self.db.get_task(task)
            if sub_task_id is None:
                error = self.unknown_task_error(task)
                return_object.error = error
                return return_object
            self.db.add_sub_task(task_id, sub_task_id, task)
        return_object.is_success = True
        return return_object

    def missing_entities_error(self, task_word):
        error = TaskGroundingError()
        error.error_task_name = task_word
        error.error_code = TaskErrorType.NO_OBJECT
        return error

    def unknown_task_error(self, task_word):
        error = TaskGroundingError()
        error.error_code = TaskErrorType.UNKNOWN
        error.error_task_name = task_word
        return error


"""if __name__ == "__main__":
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
    print(task)"""