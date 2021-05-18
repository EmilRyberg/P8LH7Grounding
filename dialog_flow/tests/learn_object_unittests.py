
import unittest
from unittest.mock import Mock
from ner_lib.command_builder import Task, TaskType, ObjectEntity, SpatialType, SpatialDescription
from ner_lib.command_builder import CommandBuilder, SpatialType, Task, TaskType, ObjectEntity as ObjectEntityType
from ner_lib.ner import NER, EntityType
from grounding_lib.grounding import Grounding, GroundingErrorType, GroundingReturn
from task_grounding.task_grounding import TaskGrounding, TaskGroundingError, TaskErrorType, TaskGroundingReturn
from dialog_flow.nodes import dialog_flow
import random


class StartTeachObjectStateTest(unittest.TestCase):
    def setUp(self):
        self.state_dict = {
            "last_received_sentence": None,
            "last_received_sentence_timestamp": None,
            "websocket_is_connected": False,
            "task_grounding_return": None,
            "base_task": Task,
            "wait_response_called_from": None,
            "tasks_to_perform": None,
            "carrying_object": False,
            "grounding_error": None,
        }
        self.container = Mock()

    def test_start_teach_object__success__returns_verify_task_state(self):
        extract_task_State = dialog_flow.StartTeachObjectState(self.state_dict, self.container)
        extract_task_State.state_dict["last_received_sentence"] = "Dummy sentence"
        return_state = extract_task_State.execute()
        self.assertTrue(isinstance(return_state, dialog_flow.VerifyObjectNameState))

class VerifyObjectNameStateTest(unittest.TestCase):
    def setUp(self):
        self.state_dict = {
            "last_received_sentence": None,
            "last_received_sentence_timestamp": None,
            "websocket_is_connected": False,
            "task_grounding_return": None,
            "base_task": Task,
            "wait_response_called_from": None,
            "tasks_to_perform": None,
            "carrying_object": False,
            "grounding_error": None,
        }
        self.container = Mock()

    def test_verify_object_name_state__first_call__returns_wait(self):
        validate_task_state = dialog_flow.VerifyObjectNameState(self.state_dict, self.container)
        self.container.speak = Mock()
        return_state = validate_task_state.execute()
        self.assertTrue(isinstance(return_state, dialog_flow.WaitForResponseState))

    def test_verify_object_name_state__invalid_name__returns_self(self):
        validate_task_state = dialog_flow.VerifyObjectNameState(self.state_dict, self.container)
        validate_task_state.is_first_call = False
        validate_task_state.state_dict["last_received_sentence"] = "On top of the table"
        test_entities = [(EntityType.LOCATION, "")]
        self.container.ner.get_entities = Mock(return_value=test_entities)
        self.container.speak = Mock()
        return_state = validate_task_state.execute()
        self.assertTrue(isinstance(return_state, dialog_flow.VerifyObjectNameState))
        self.assertTrue(validate_task_state.is_first_call)

    def test_verify_object_name_state__multiple_names__returns_self(self):
        validate_task_state = dialog_flow.VerifyObjectNameState(self.state_dict, self.container)
        validate_task_state.is_first_call = False
        validate_task_state.state_dict["last_received_sentence"] = "A screwdriver and a blue hammer"
        test_entities = [(EntityType.OBJECT, "screwdriver"),(EntityType.OBJECT, "hammer")]
        self.container.ner.get_entities = Mock(return_value=test_entities)
        self.container.speak = Mock()
        return_state = validate_task_state.execute()
        self.assertTrue(isinstance(return_state, dialog_flow.VerifyObjectNameState))
        self.assertTrue(validate_task_state.is_first_call)

    def test_verify_object_name_state__existing_name__returns_self(self):
        validate_task_state = dialog_flow.VerifyObjectNameState(self.state_dict, self.container)
        validate_task_state.is_first_call = False
        validate_task_state.state_dict["last_received_sentence"] = "A pcb"
        test_entities = [(EntityType.OBJECT, "pcb")]
        self.container.ner.get_entities = Mock(return_value=test_entities)
        self.container.speak = Mock()
        self.container.grounding.db.object_exists = Mock(return_value=True)
        return_state = validate_task_state.execute()
        self.assertTrue(isinstance(return_state, dialog_flow.VerifyObjectNameState))
        self.assertTrue(validate_task_state.is_first_call)

    def test_verify_object_name_state__valid_name__returns_ask_clear_table_state(self):
        validate_task_state = dialog_flow.VerifyObjectNameState(self.state_dict, self.container)
        validate_task_state.is_first_call = False
        validate_task_state.state_dict["last_received_sentence"] = "A pcb"
        test_entities = [(EntityType.OBJECT, "pcb")]
        self.container.ner.get_entities = Mock(return_value=test_entities)
        self.container.speak = Mock()
        self.container.grounding.db.object_exists = Mock(return_value=False)
        return_state = validate_task_state.execute()
        self.assertTrue(isinstance(return_state, dialog_flow.AskClearTableState))

class AskClearTableStateTest(unittest.TestCase):
    def setUp(self):
        self.state_dict = {
            "last_received_sentence": None,
            "last_received_sentence_timestamp": None,
            "websocket_is_connected": False,
            "task_grounding_return": TaskGroundingReturn(),
            "base_task": Task,
            "wait_response_called_from": None,
            "tasks_to_perform": None,
            "carrying_object": False,
            "grounding_error": None,
        }
        self.container = Mock()

    def test_ask_clear_table__first_call__returns_wait_response_state(self):
        error = TaskGroundingError()
        error.error_code = TaskErrorType.UNKNOWN
        error.error_task_name = "Dummy Task"
        clarify_state = dialog_flow.AskClearTableState(self.state_dict, self.container,"newobject")
        clarify_state.error = error
        self.container.speak = Mock()
        return_state = clarify_state.execute()
        self.assertTrue(isinstance(return_state, dialog_flow.WaitForResponseState))

    def test_ask_clear_table__negative_response__returns_wait_response_state(self):
        error = TaskGroundingError()
        error.error_code = TaskErrorType.UNKNOWN
        error.error_task_name = "Dummy Task"
        clarify_state = dialog_flow.AskClearTableState(self.state_dict, self.container,"newobject")
        test_entities = [(EntityType.DENIAL, "nah mate")]
        clarify_state.is_first_call=False
        self.container.ner.get_entities = Mock(return_value=test_entities)
        clarify_state.error = error
        self.container.speak = Mock()
        return_state = clarify_state.execute()
        self.assertTrue(isinstance(return_state, dialog_flow.WaitForResponseState))

    def test_ask_clear_table__unclear_response__returns_wait_response_state(self):
        error = TaskGroundingError()
        error.error_code = TaskErrorType.UNKNOWN
        error.error_task_name = "Dummy Task"
        clarify_state = dialog_flow.AskClearTableState(self.state_dict, self.container,"newobject")
        test_entities = [(EntityType.GREETING, "some nonsense")]
        clarify_state.is_first_call=False
        self.container.ner.get_entities = Mock(return_value=test_entities)
        clarify_state.error = error
        self.container.speak = Mock()
        return_state = clarify_state.execute()
        self.assertTrue(isinstance(return_state, dialog_flow.WaitForResponseState))

    def test_ask_clear_table__existing_object__returns_start_teach_object(self):
        error = TaskGroundingError()
        error.error_code = TaskErrorType.UNKNOWN
        error.error_task_name = "Dummy Task"
        clarify_state = dialog_flow.AskClearTableState(self.state_dict, self.container,"newobject")
        test_entities = [(EntityType.AFFIRMATION, "yeah mate")]
        clarify_state.is_first_call=False
        self.container.ner.get_entities = Mock(return_value=test_entities)

        return_object = GroundingReturn()
        return_object.is_success = False
        return_object.error_code = GroundingErrorType.ALREADY_KNOWN

        self.container.grounding.learn_new_object = Mock(return_value=return_object)
        clarify_state.error = error
        self.container.speak = Mock()
        return_state = clarify_state.execute()
        self.assertIsInstance(return_state, dialog_flow.StartTeachObjectState)

    def test_ask_clear_table__multiple_object__returns_wait_response_state(self):
        error = TaskGroundingError()
        error.error_code = TaskErrorType.UNKNOWN
        error.error_task_name = "Dummy Task"
        clarify_state = dialog_flow.AskClearTableState(self.state_dict, self.container,"newobject")
        test_entities = [(EntityType.AFFIRMATION, "yeah mate")]
        clarify_state.is_first_call=False
        self.container.ner.get_entities = Mock(return_value=test_entities)

        return_object = GroundingReturn()
        return_object.is_success = False
        return_object.error_code = GroundingErrorType.MULTIPLE_REF

        self.container.grounding.learn_new_object = Mock(return_value=return_object)
        clarify_state.error = error
        self.container.speak = Mock()
        return_state = clarify_state.execute()
        self.assertIsInstance(return_state, dialog_flow.WaitForResponseState)


    def test_ask_clear_table__success__returns_goodness_knows_what(self):
        error = TaskGroundingError()
        error.error_code = TaskErrorType.UNKNOWN
        error.error_task_name = "Dummy Task"
        clarify_state = dialog_flow.AskClearTableState(self.state_dict, self.container,"newobject")
        test_entities = [(EntityType.AFFIRMATION, "yeah mate")]
        clarify_state.is_first_call=False
        self.container.ner.get_entities = Mock(return_value=test_entities)

        return_object = GroundingReturn()
        return_object.is_success = True
        return_object.error_code = None

        self.container.grounding.learn_new_object = Mock(return_value=return_object)
        clarify_state.error = error
        self.container.speak = Mock()
        return_state = clarify_state.execute()
        '''TODO: No idea what a success should return here'''
        self.assertIsInstance(return_state, dialog_flow.WaitForResponseState)