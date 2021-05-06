from task_grounding.task_grounding import TaskGrounding, TaskGroundingReturn, TaskErrorType
from database_handler.database_handler import DatabaseHandler
import unittest
from unittest.mock import Mock
from ner_lib.ner import EntityType
from ner_lib.command_builder import Task, TaskType, ObjectEntity, SpatialType, SpatialDescription

################################# ISOLATED UNIT TESTS ----- BEGIN ##########################################################


class SimpleSkillTest(unittest.TestCase):
    def setUp(self):
        self.db_mock = Mock()
        self.task_grounding = TaskGrounding(db=self.db_mock)
        self.entities = [
            (EntityType.COLOUR, "blue"),
            (EntityType.OBJECT, "cover"),
            (EntityType.LOCATION, "next"),
            (EntityType.COLOUR, "black"),
            (EntityType.OBJECT, "bottom cover"),
            (EntityType.LOCATION, "above"),
            (EntityType.OBJECT, "bottom cover")
        ]

    def test_get_specific_task_from_task__task_is_pick_up__returns_task_with_pick_up(self):
        self.db_mock.get_task = Mock(return_value=(1, "pick up"))
        task = Task(name="pick up")
        task.objects_to_execute_on = [ObjectEntity()]

        returned = self.task_grounding.get_specific_task_from_task(task)

        self.assertEqual(TaskType.PICK, returned.task_info[0].task_type)

    def test_get_specific_task_from_task__task_is_move__returns_task_with_move(self):
        self.db_mock.get_task = Mock(return_value=(1, "move"))
        task = Task(name="move")
        task.objects_to_execute_on = [ObjectEntity()]

        returned = self.task_grounding.get_specific_task_from_task(task)

        self.assertEqual(TaskType.MOVE, returned.task_info[0].task_type)

    def test_get_specific_task_from_task__task_is_place__returns_task_with_place(self):
        self.db_mock.get_task = Mock(return_value=(1, "place"))
        task = Task(name="place")
        task.objects_to_execute_on = [ObjectEntity()]

        returned = self.task_grounding.get_specific_task_from_task(task)

        self.assertEqual(TaskType.PLACE, returned.task_info[0].task_type)

    def test_get_specific_task_from_task__task_is_find__returns_task_with_find(self):
        self.db_mock.get_task = Mock(return_value=(1, "find"))
        task = Task(name="find")
        task.objects_to_execute_on = [ObjectEntity()]

        returned = self.task_grounding.get_specific_task_from_task(task)

        self.assertEqual(TaskType.FIND, returned.task_info[0].task_type)

    def test_get_specific_task_from_task__task_is_unknown__returns_error_code_unknown(self):
        self.db_mock.get_task = Mock(return_value=(1, None))
        task = Task(name="asdasd")

        returned = self.task_grounding.get_specific_task_from_task(task)

        self.assertFalse(returned.is_success)
        self.assertEqual(TaskErrorType.UNKNOWN, returned.error.error_code)

    def test_get_specific_task_from_task__task_has_no_object__returns_error_code_no_object(self):
        self.db_mock.get_task = Mock(return_value=(1, "pick up"))
        task = Task(name="pick up")

        returned = self.task_grounding.get_specific_task_from_task(task)

        self.assertFalse(returned.is_success)
        self.assertEqual(TaskErrorType.NO_OBJECT, returned.error.error_code)


class AdvancedTaskTest(unittest.TestCase):
    def setUp(self):
        self.db_mock = Mock()
        self.task_grounding = TaskGrounding(db=self.db_mock)
        self.entities = [
            (EntityType.COLOUR, "blue"),
            (EntityType.OBJECT, "cover"),
            (EntityType.LOCATION, "next"),
            (EntityType.COLOUR, "black"),
            (EntityType.OBJECT, "bottom cover"),
            (EntityType.LOCATION, "above"),
            (EntityType.OBJECT, "bottom cover")
        ]

    def test_get_specific_task_from_task__task_is_custom_task__returns_list_of_primary_skills(self):
        pick_up_task = Task("pick up")
        pick_up_task.task_type = TaskType.PICK
        pick_up_task.objects_to_execute_on = [ObjectEntity()]
        move_task = Task("pick up")
        move_task.task_type = TaskType.MOVE
        move_task.objects_to_execute_on = [ObjectEntity()]
        place_task = Task("pick up")
        place_task.task_type = TaskType.PICK
        place_task.objects_to_execute_on = [ObjectEntity()]
        sub_tasks = [[1, 2, 3], ["pick up", "move", "place"], [pick_up_task, move_task, place_task]]
        tasks = [TaskType.PICK, TaskType.MOVE, TaskType.PLACE]
        self.db_mock.get_task = Mock(return_value=(1, "clear table"))
        self.db_mock.get_sub_tasks = Mock(return_value=sub_tasks)
        task = Task("tidy")

        returned = self.task_grounding.get_specific_task_from_task(task)

        returned_tasks = [returned.task_info[0].task_type,
                          returned.task_info[1].task_type,
                          returned.task_info[2].task_type]

        self.assertEqual(tasks, returned_tasks)

    def test_get_specific_task_from_tasks__task_is_custom_task_without_sub_tasks__returns_error_code_no_sub_tasks(self):
        self.db_mock.get_task = Mock(return_value=(1, "clear table"))
        self.db_mock.get_sub_tasks = Mock(return_value=None)
        task = Task("tidy")

        returned = self.task_grounding.get_specific_task_from_task(task)
        self.assertFalse(returned.is_success)
        self.assertEqual(TaskErrorType.NO_SUBTASKS, returned.error.error_code)


class TeachSystemTest(unittest.TestCase):
    def setUp(self):
        self.db_mock = Mock()
        self.task_grounding = TaskGrounding(db=self.db_mock)

    def test_teach_new_task__valid_input__returns_success(self):
        self.db_mock.add_task = Mock()
        self.db_mock.get_task = Mock()
        self.db_mock.get_task.side_effect = [(1, None), (2, None), (3, None)]
        self.db_mock.add_sub_task = Mock()

        returned = self.task_grounding.teach_new_task("nice task name", [Task("take"), Task("move"), Task("put")], "nice task keyword")
        self.assertTrue(returned.is_success)

    def test_teach_new_task__contains_unknown_task__returns_unknown_error_code(self):
        self.db_mock.add_task = Mock()
        self.db_mock.get_task = Mock()
        self.db_mock.get_task.side_effect = [(None, None)]
        self.db_mock.add_sub_task = Mock()

        returned = self.task_grounding.teach_new_task("nice task name", [Task("take"), Task("move"), Task("put")], "nice task keyword")

        self.assertFalse(returned.is_success)
        self.assertEqual(TaskErrorType.UNKNOWN, returned.error.error_code)

    def test_add_sub_task__valid_input__returns_success(self):
        self.db_mock.get_task = Mock()
        self.db_mock.add_sub_task = Mock()
        self.db_mock.get_task.side_effect = [(5, "clear table"), (1, "pick up")]

        returned = self.task_grounding.add_sub_task("tidy", ["get"])

        self.assertTrue(returned.is_success)

################################# ISOLATED UNIT TESTS ----- END ##########################################################

################################# INTEGRATION TESTS ----- BEGIN ##########################################################
class SimpleSkillIntegration(unittest.TestCase):
    def setUp(self):
        self.task_grounding = TaskGrounding(DatabaseHandler("test_grounding.db"))
        self.returned = TaskGroundingReturn()
        self.entities = [
            (EntityType.COLOUR, "blue"),
            (EntityType.OBJECT, "cover"),
            (EntityType.LOCATION, "next"),
            (EntityType.COLOUR, "black"),
            (EntityType.OBJECT, "bottom cover"),
            (EntityType.LOCATION, "above"),
            (EntityType.OBJECT, "bottom cover")
        ]

    def test_Pick(self):
        self.returned = self.task_grounding.get_specific_task_from_task("take", self.entities)
        self.assertEqual(self.returned.task_info[0].get_name(), "PickUpTask")

    def test_Move(self):
        self.returned = self.task_grounding.get_specific_task_from_task("relocate", self.entities)
        self.assertEqual(self.returned.task_info[0].get_name(), "MoveTask")

    def test_Place(self):
        self.returned = self.task_grounding.get_specific_task_from_task("put", self.entities)
        self.assertEqual(self.returned.task_info[0].get_name(), "PlaceTask")

    def test_Find(self):
        self.returned = self.task_grounding.get_specific_task_from_task("locate", self.entities)
        self.assertEqual(self.returned.task_info[0].get_name(), "FindTask")

    def test_UnknownObject(self):
        self.returned = self.task_grounding.get_specific_task_from_task("asdasd")
        self.assertFalse(self.returned.is_success)
        self.assertEqual(self.returned.error_code, TaskErrorType.UNKNOWN)

    def test_NoObjectSpecified(self):
        self.returned = self.task_grounding.get_specific_task_from_task("take")
        self.assertFalse(self.returned.is_success)
        self.assertEqual(self.returned.error_code, TaskErrorType.NO_OBJECT)


class AdvancedTaskIntegration(unittest.TestCase):
    def setUp(self):
        self.task_grounding = TaskGrounding(DatabaseHandler("test_grounding.db"))
        self.returned = TaskGroundingReturn()
        self.entities = [
            (EntityType.COLOUR, "blue"),
            (EntityType.OBJECT, "cover"),
            (EntityType.LOCATION, "next"),
            (EntityType.COLOUR, "black"),
            (EntityType.OBJECT, "bottom cover"),
            (EntityType.LOCATION, "above"),
            (EntityType.OBJECT, "bottom cover")
        ]

    def test_MoveBlue(self):
        tasks = [TaskType.PICK, TaskType.PLACE]
        returned = self.task_grounding.get_specific_task_from_task(Task("blue1"))
        returned_tasks = [returned.task_info[0].task_type,
                          returned.task_info[1].task_type]
        self.assertEqual(tasks, returned_tasks)

    def test_ClearTable(self):
        tasks = ["PickUpTask", "MoveTask", "PlaceTask"]
        self.returned = self.task_grounding.get_specific_task_from_task("tidy", self.entities)
        returned_tasks = [self.returned.task_info[0].get_name(),
                          self.returned.task_info[1].get_name(),
                          self.returned.task_info[2].get_name()]
        self.assertEqual(tasks, returned_tasks)


class TeachSystemIntegration(unittest.TestCase):
    def setUp(self):
        self.db = DatabaseHandler("test_grounding.db")
        self.task_grounding = TaskGrounding(self.db)
        self.returned = TaskGroundingReturn()

    def test_TeachTask(self):
        returned = self.task_grounding.teach_new_task("test_task1", ["take", "move", "put"], ["test1-1", "test1-2"])
        self.assertTrue(returned.is_success)
        self.clean_test_db("test_task1")

    def test_AddWord(self):
        returned = self.task_grounding.add_word_to_task("blue1", "blue2")
        self.assertTrue(returned.is_success)
        self.db.conn.execute("delete from TASK_WORDS where WORD='blue2';")
        self.db.conn.commit()

    def test_TeachTaskUnknownSubTask(self):
        returned = self.task_grounding.teach_new_task("test_task2", ["UNKNOWN TASK"], ["test1", "test2-1"])
        self.assertFalse(returned.is_success)
        self.assertEqual(returned.error_code, TaskErrorType.UNKNOWN)
        self.clean_test_db("test_task2")

    def test_AddWordsToTask(self):
        #self.task_grounding.teach_new_task("test_task3", ["take", "move", "put"], ["test3-1", "test3-2"])
        #returned = self.task_grounding.add_word_to_task("test_task3-1", "TEST WORD")
        #self.assertTrue(returned.is_success)
        self.clean_test_db("test_task3")

    def test_AddSubTask(self):
        self.task_grounding.teach_new_task("test_task4", ["take", "move", "put"], ["test4-1", "test4-2"])
        returned = self.task_grounding.add_sub_task("test_task4", ["get"])
        self.assertTrue(returned.is_success)
        self.clean_test_db("test_task4")

    def clean_test_db(self, task_name):
        task_id = self.db.get_task_id(task_name)
        self.db.conn.execute("delete from TASK_WORDS where TASK_ID=?;", (task_id,))
        self.db.conn.execute("delete from TASK_INFO where TASK_NAME=?;", (task_name,))
        self.db.conn.commit()





################################# INTEGRATION TESTS ----- END ##########################################################