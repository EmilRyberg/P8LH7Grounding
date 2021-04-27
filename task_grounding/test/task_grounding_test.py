from task_grounding.task_grounding import TaskGrounding, TaskGroundingReturn, ErrorType
from database_handler import DatabaseHandler
import unittest
from unittest.mock import MagicMock, Mock
from ner_lib.command_builder import CommandBuilder, PlaceTask, PickUpTask, FindTask, MoveTask
from ner_lib.ner import EntityType

################################# ISOLATED UNIT TESTS ----- BEGIN ##########################################################


class SimpleSkillTest(unittest.TestCase):
    def setUp(self):
        self.db_mock = Mock()
        self.task_grounding = TaskGrounding(db=self.db_mock)
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
        self.db_mock.get_task = Mock(return_value=(1, "pick up"))
        self.returned = self.task_grounding.get_task_from_entity("take", self.entities)
        self.assertEqual(self.returned.task_info[0].get_name(), "PickUpTask")

    def test_Move(self):
        self.db_mock.get_task = Mock(return_value=(1, "move"))
        self.returned = self.task_grounding.get_task_from_entity("relocate", self.entities)
        self.assertEqual(self.returned.task_info[0].get_name(), "MoveTask")

    def test_Place(self):
        self.db_mock.get_task = Mock(return_value=(1, "place"))
        self.returned = self.task_grounding.get_task_from_entity("put", self.entities)
        self.assertEqual(self.returned.task_info[0].get_name(), "PlaceTask")

    def test_Find(self):
        self.db_mock.get_task = Mock(return_value=(1, "find"))
        self.returned = self.task_grounding.get_task_from_entity("locate", self.entities)
        self.assertEqual(self.returned.task_info[0].get_name(), "FindTask")

    def test_UnknownObject(self):
        self.db_mock.get_task = Mock(return_value=(1, None))
        self.returned = self.task_grounding.get_task_from_entity("asdasd")
        self.assertFalse(self.returned.is_success)
        self.assertEqual(self.returned.error_code, ErrorType.UNKNOWN)

    def test_NoObjectSpecified(self):
        self.db_mock.get_task = Mock(return_value=(1, "pick up"))
        self.returned = self.task_grounding.get_task_from_entity("take")
        self.assertFalse(self.returned.is_success)
        self.assertEqual(self.returned.error_code, ErrorType.NO_OBJECT)

class AdvancedTaskTest(unittest.TestCase):
    def setUp(self):
        self.db_mock = Mock()
        self.task_grounding = TaskGrounding(db=self.db_mock)
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

    def test_ClearTable(self):
        sub_tasks = [[1, 2, 3], ["pick up", "move", "place"], [None, None, None], [None, None, None]]
        tasks = ["PickUpTask", "MoveTask", "PlaceTask"]
        self.db_mock.get_task = Mock(return_value=(1, "clear table"))
        #self.db_mock.get_task.side_effect = [(1, "clear table"), (1, "pick up"), (1, "move"), (1, "place")]
        self.db_mock.get_task_name = Mock()
        self.db_mock.get_task_name.side_effect = ["pick up", "move", "place"]
        self.db_mock.get_sub_tasks = Mock(return_value=sub_tasks)
        self.returned = self.task_grounding.get_task_from_entity("tidy", self.entities)
        returned_tasks = [self.returned.task_info[0].get_name(),
                          self.returned.task_info[1].get_name(),
                          self.returned.task_info[2].get_name()]
        self.assertEqual(tasks, returned_tasks)

    def test_NoSubTasks(self):
        self.db_mock.get_task = Mock(return_value=(1, "clear table"))
        self.db_mock.get_sub_tasks = Mock(return_value=None)
        self.returned = self.task_grounding.get_task_from_entity("tidy", self.entities)
        self.assertFalse(self.returned.is_success)
        self.assertEqual(self.returned.error_code, ErrorType.NO_SUBTASKS)

class TeachSystemTest(unittest.TestCase):
    def setUp(self):
        self.db_mock = Mock()
        self.task_grounding = TaskGrounding(db=self.db_mock)
        self.returned = TaskGroundingReturn()

    def test_TeachTask(self):
        self.db_mock.add_task = Mock()
        self.db_mock.get_task = Mock()
        self.db_mock.get_task.side_effect = [(1, None), (2, None), (3, None)]
        self.db_mock.add_sub_task = Mock()
        self.returned = self.task_grounding.teach_new_task("nice task name", ["take", "move", "put"], "nice task keyword")
        self.assertTrue(self.returned.is_success)

    def test_TeachTaskUnknownSubTask(self):
        self.db_mock.add_task = Mock()
        self.db_mock.get_task = Mock()
        self.db_mock.get_task.side_effect = [(None, None)]
        self.db_mock.add_sub_task = Mock()
        self.returned = self.task_grounding.teach_new_task("nice task name", ["take", "move", "put"], "nice task keyword")
        self.assertFalse(self.returned.is_success)
        self.assertEqual(self.returned.error_code, ErrorType.UNKNOWN)

    def test_AddSubTask(self):
        self.db_mock.get_task = Mock()
        self.db_mock.add_sub_task = Mock()
        self.db_mock.get_task.side_effect = [(5, "clear table"), (1, "pick up")]
        self.returned = self.task_grounding.add_sub_task("tidy", ["get"])
        self.assertTrue(self.returned.is_success)

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
        self.returned = self.task_grounding.get_task_from_entity("take", self.entities)
        self.assertEqual(self.returned.task_info[0].get_name(), "PickUpTask")

    def test_Move(self):
        self.returned = self.task_grounding.get_task_from_entity("relocate", self.entities)
        self.assertEqual(self.returned.task_info[0].get_name(), "MoveTask")

    def test_Place(self):
        self.returned = self.task_grounding.get_task_from_entity("put", self.entities)
        self.assertEqual(self.returned.task_info[0].get_name(), "PlaceTask")

    def test_Find(self):
        self.returned = self.task_grounding.get_task_from_entity("locate", self.entities)
        self.assertEqual(self.returned.task_info[0].get_name(), "FindTask")

    def test_UnknownObject(self):
        self.returned = self.task_grounding.get_task_from_entity("asdasd")
        self.assertFalse(self.returned.is_success)
        self.assertEqual(self.returned.error_code, ErrorType.UNKNOWN)

    def test_NoObjectSpecified(self):
        self.returned = self.task_grounding.get_task_from_entity("take")
        self.assertFalse(self.returned.is_success)
        self.assertEqual(self.returned.error_code, ErrorType.NO_OBJECT)

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
        tasks = ["PickUpTask", "PlaceTask"]
        self.returned = self.task_grounding.get_task_from_entity("blue1", self.entities)
        returned_tasks = [self.returned.task_info[0].get_name(),
                          self.returned.task_info[1].get_name()]
        self.assertEqual(tasks, returned_tasks)

    def test_ClearTable(self):
        tasks = ["PickUpTask", "MoveTask", "PlaceTask"]
        self.returned = self.task_grounding.get_task_from_entity("tidy", self.entities)
        returned_tasks = [self.returned.task_info[0].get_name(),
                          self.returned.task_info[1].get_name(),
                          self.returned.task_info[2].get_name()]
        self.assertEqual(tasks, returned_tasks)

        def test_ClearTable(self):
            tasks = ["PickUpTask", "MoveTask", "PlaceTask"]
            self.returned = self.task_grounding.get_task_from_entity("tidy", self.entities)
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
        self.assertEqual(returned.error_code, ErrorType.UNKNOWN)
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