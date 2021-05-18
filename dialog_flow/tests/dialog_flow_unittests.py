
import unittest
from unittest.mock import Mock
from ner_lib.command_builder import Task, TaskType, ObjectEntity, SpatialType, SpatialDescription
from ner_lib.command_builder import CommandBuilder, SpatialType, Task, TaskType, ObjectEntity as ObjectEntityType
from ner_lib.ner import NER, EntityType
from grounding_lib.grounding import Grounding, GroundingErrorType, GroundingReturn
from task_grounding.task_grounding import TaskGrounding, TaskGroundingError, TaskErrorType, TaskGroundingReturn
from dialog_flow.nodes import dialog_flow
import random


class GreetStateTest(unittest.TestCase):
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

    def test_greet__dont_receive_command__returns_ask_for_command_state(self):
        greetstate = dialog_flow.GreetState(self.state_dict, self.container)
        self.container.speak = Mock()
        greetstate.is_first_call = False
        greetstate.state_dict["last_received_sentence"] = None
        return_state = greetstate.execute()
        self.assertTrue(isinstance(return_state, dialog_flow.AskForCommandState))

    def test_greet__first_call__returns_wait_for_command_state(self):
        greetstate = dialog_flow.GreetState(self.state_dict, self.container)
        self.container.speak = Mock()
        return_state = greetstate.execute()
        self.assertTrue(isinstance(return_state, dialog_flow.WaitForResponseState))

    def test_greet__sentence_with_task__returns_verify_command_state(self):
        entities = [
            (EntityType.COLOUR, "blue"),
            (EntityType.TASK, "cover"),
            (EntityType.LOCATION, "next"),
            (EntityType.COLOUR, "black"),
            (EntityType.OBJECT, "bottom cover"),
            (EntityType.LOCATION, "above"),
            (EntityType.OBJECT, "bottom cover")
        ]
        greetstate = dialog_flow.GreetState(self.state_dict, self.container)
        self.container.speak = Mock()
        self.container.send_human_sentence_to_gui = Mock()
        self.container.ner.get_entities = Mock(return_value=entities)
        greetstate.is_first_call = False
        greetstate.state_dict["last_received_sentence"] = "Dummy sentence"
        return_state = greetstate.execute()
        self.assertTrue(isinstance(return_state, dialog_flow.VerifyCommandState))

    def test_greet__sentence_with_teach__returns_start_teach_state(self):
        entities = [
            (EntityType.COLOUR, "blue"),
            (EntityType.TASK, "cover"),
            (EntityType.LOCATION, "next"),
            (EntityType.TEACH, "black"),
            (EntityType.OBJECT, "bottom cover"),
            (EntityType.LOCATION, "above"),
            (EntityType.OBJECT, "bottom cover")
        ]
        greetstate = dialog_flow.GreetState(self.state_dict, self.container)
        self.container.speak = Mock()
        self.container.send_human_sentence_to_gui = Mock()
        self.container.ner.get_entities = Mock(return_value=entities)
        greetstate.is_first_call = False
        greetstate.state_dict["last_received_sentence"] = "Dummy sentence"
        return_state = greetstate.execute()
        self.assertTrue(isinstance(return_state, dialog_flow.StartTeachState))

class AskForCommandStateTest(unittest.TestCase):
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

    def test_ask_for_command__previous_greet_first_run__returns_wait_for_command_state(self):
        greet_state = dialog_flow.GreetState(self.state_dict, self.container)
        ask_for_command_state = dialog_flow.AskForCommandState(self.state_dict, self.container, greet_state)
        self.container.speak = Mock()
        return_state = ask_for_command_state.execute()
        self.assertTrue(isinstance(return_state, dialog_flow.WaitForResponseState))

    def test_ask_for_command__not_teach_has_task__returns_verify_command_state(self):
        entities = [
            (EntityType.COLOUR, "blue"),
            (EntityType.TASK, "cover"),
            (EntityType.LOCATION, "next"),
            (EntityType.LOCATION, "black"),
            (EntityType.OBJECT, "bottom cover"),
            (EntityType.LOCATION, "above"),
            (EntityType.OBJECT, "bottom cover")
        ]
        ask_for_command_state = dialog_flow.AskForCommandState(self.state_dict, self.container)
        ask_for_command_state.is_first_call = False
        self.container.speak = Mock()
        self.container.ner.get_entities = Mock(return_value=entities)
        ask_for_command_state.state_dict["last_received_sentence"] = "Dummy sentence"
        return_state = ask_for_command_state.execute()
        self.assertTrue(isinstance(return_state, dialog_flow.VerifyCommandState))

    def test_ask_for_command__has_teach__returns_start_teach_state(self):
        entities = [
            (EntityType.COLOUR, "blue"),
            (EntityType.TASK, "cover"),
            (EntityType.TEACH, "next"),
            (EntityType.LOCATION, "black"),
            (EntityType.OBJECT, "bottom cover"),
            (EntityType.LOCATION, "above"),
            (EntityType.OBJECT, "bottom cover")
        ]
        ask_for_command_state = dialog_flow.AskForCommandState(self.state_dict, self.container)
        ask_for_command_state.is_first_call = False
        self.container.speak = Mock()
        self.container.ner.get_entities = Mock(return_value=entities)
        ask_for_command_state.state_dict["last_received_sentence"] = "Dummy sentence"
        return_state = ask_for_command_state.execute()
        self.assertTrue(isinstance(return_state, dialog_flow.StartTeachState))

    def test_ask_for_command__affirmation__returns_wait_response_state(self):
        entities = [
            (EntityType.AFFIRMATION, "blue"),
            (EntityType.LOCATION, "cover"),
            (EntityType.OBJECT, "next"),
            (EntityType.LOCATION, "black"),
            (EntityType.OBJECT, "bottom cover"),
            (EntityType.LOCATION, "above"),
            (EntityType.OBJECT, "bottom cover")
        ]
        ask_for_command_state = dialog_flow.AskForCommandState(self.state_dict, self.container)
        ask_for_command_state.is_first_call = False
        self.container.speak = Mock()
        self.container.ner.get_entities = Mock(return_value=entities)
        ask_for_command_state.state_dict["last_received_sentence"] = "Dummy sentence"
        return_state = ask_for_command_state.execute()
        self.assertTrue(isinstance(return_state, dialog_flow.WaitForResponseState))

    def test_ask_for_command__denial__returns_wait_greet_state(self):
        entities = [
            (EntityType.DENIAL, "blue"),
            (EntityType.LOCATION, "cover"),
            (EntityType.OBJECT, "next"),
            (EntityType.LOCATION, "black"),
            (EntityType.OBJECT, "bottom cover"),
            (EntityType.LOCATION, "above"),
            (EntityType.OBJECT, "bottom cover")
        ]
        ask_for_command_state = dialog_flow.AskForCommandState(self.state_dict, self.container)
        ask_for_command_state.is_first_call = False
        self.container.speak = Mock()
        self.container.ner.get_entities = Mock(return_value=entities)
        ask_for_command_state.state_dict["last_received_sentence"] = "Dummy sentence"
        return_state = ask_for_command_state.execute()
        self.assertTrue(isinstance(return_state, dialog_flow.WaitForGreetingState))

    def test_ask_for_command__no_task_teach_or_affirmation__returns_wait_response_state(self):
        entities = [
            (EntityType.LOCATION, "blue"),
            (EntityType.LOCATION, "cover"),
            (EntityType.OBJECT, "next"),
            (EntityType.LOCATION, "black"),
            (EntityType.OBJECT, "bottom cover"),
            (EntityType.LOCATION, "above"),
            (EntityType.OBJECT, "bottom cover")
        ]
        ask_for_command_state = dialog_flow.AskForCommandState(self.state_dict, self.container)
        ask_for_command_state.is_first_call = False
        self.container.speak = Mock()
        self.container.ner.get_entities = Mock(return_value=entities)
        ask_for_command_state.state_dict["last_received_sentence"] = "Dummy sentence"
        return_state = ask_for_command_state.execute()
        self.assertTrue(isinstance(return_state, dialog_flow.WaitForResponseState))

class VerifyCommandStateTest(unittest.TestCase):
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

    def test_verify_command__no_teach_first_run__returns_wait_response_state(self):
        entities = [
            (EntityType.LOCATION, "blue"),
            (EntityType.LOCATION, "cover"),
            (EntityType.OBJECT, "next"),
            (EntityType.LOCATION, "black"),
            (EntityType.OBJECT, "bottom cover"),
            (EntityType.LOCATION, "above"),
            (EntityType.OBJECT, "bottom cover")
        ]
        task = Task(name="Dummy name")
        verify_command_state = dialog_flow.VerifyCommandState(self.state_dict, self.container)
        self.container.speak = Mock()
        self.container.ner.get_entities = Mock(return_value=entities)
        self.container.command_builder.get_task = Mock(return_value=task)
        verify_command_state.state_dict["last_received_sentence"] = "Dummy sentence"
        return_state = verify_command_state.execute()
        self.assertTrue(isinstance(return_state, dialog_flow.WaitForResponseState))

    def test_verify_command__has_teach_first_run__returns_start_teach_state(self):
        entities = [
            (EntityType.TEACH, "blue"),
            (EntityType.LOCATION, "cover"),
            (EntityType.OBJECT, "next"),
            (EntityType.LOCATION, "black"),
            (EntityType.OBJECT, "bottom cover"),
            (EntityType.LOCATION, "above"),
            (EntityType.OBJECT, "bottom cover")
        ]
        verify_command_state = dialog_flow.VerifyCommandState(self.state_dict, self.container)
        self.container.speak = Mock()
        self.container.ner.get_entities = Mock(return_value=entities)
        verify_command_state.state_dict["last_received_sentence"] = "Dummy sentence"
        return_state = verify_command_state.execute()
        self.assertTrue(isinstance(return_state, dialog_flow.StartTeachState))

    def test_verify_command__affirmation__returns_extract_task_state(self):
        entities = [
            (EntityType.AFFIRMATION, "blue"),
            (EntityType.LOCATION, "cover"),
            (EntityType.OBJECT, "next"),
            (EntityType.LOCATION, "black"),
            (EntityType.OBJECT, "bottom cover"),
            (EntityType.LOCATION, "above"),
            (EntityType.OBJECT, "bottom cover")
        ]
        verify_command_state = dialog_flow.VerifyCommandState(self.state_dict, self.container)
        verify_command_state.is_first_call = False
        self.container.speak = Mock()
        self.container.ner.get_entities = Mock(return_value=entities)
        verify_command_state.state_dict["last_received_sentence"] = "Dummy sentence"
        return_state = verify_command_state.execute()
        self.assertTrue(isinstance(return_state, dialog_flow.ExtractTaskState))

    def test_verify_command__denial__returns_ask_for_command_state(self):
        entities = [
            (EntityType.DENIAL, "blue"),
            (EntityType.LOCATION, "cover"),
            (EntityType.OBJECT, "next"),
            (EntityType.LOCATION, "black"),
            (EntityType.OBJECT, "bottom cover"),
            (EntityType.LOCATION, "above"),
            (EntityType.OBJECT, "bottom cover")
        ]
        verify_command_state = dialog_flow.VerifyCommandState(self.state_dict, self.container)
        verify_command_state.is_first_call = False
        self.container.speak = Mock()
        self.container.ner.get_entities = Mock(return_value=entities)
        verify_command_state.state_dict["last_received_sentence"] = "Dummy sentence"
        return_state = verify_command_state.execute()
        self.assertTrue(isinstance(return_state, dialog_flow.AskForCommandState))

    def test_verify_command__invalid_response__returns_wait_response_state(self):
        entities = [
            (EntityType.LOCATION, "blue"),
            (EntityType.LOCATION, "cover"),
            (EntityType.OBJECT, "next"),
            (EntityType.LOCATION, "black"),
            (EntityType.OBJECT, "bottom cover"),
            (EntityType.LOCATION, "above"),
            (EntityType.OBJECT, "bottom cover")
        ]
        task = Task(name="Dummy name")
        verify_command_state = dialog_flow.VerifyCommandState(self.state_dict, self.container)
        verify_command_state.is_first_call = False
        self.container.speak = Mock()
        verify_command_state.state_dict["base_task"]=task
        self.container.ner.get_entities = Mock(return_value=entities)
        verify_command_state.state_dict["last_received_sentence"] = "Dummy sentence"
        return_state = verify_command_state.execute()
        self.assertTrue(isinstance(return_state, dialog_flow.WaitForResponseState))


class ExtractTaskStateTest(unittest.TestCase):
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

    def test_extract_task__success__returns_perform_task_state(self):
        task_return = TaskGroundingReturn()
        task_return.is_success = True
        extract_task_State = dialog_flow.ExtractTaskState(self.state_dict, self.container)
        self.container.speak = Mock()
        self.container.task_grounding.get_specific_task_from_task = Mock(return_value=task_return)
        extract_task_State.state_dict["last_received_sentence"] = "Dummy sentence"
        return_state = extract_task_State.execute()
        self.assertTrue(isinstance(return_state, dialog_flow.PerformTaskState))

    def test_extract_task__failure__returns_validate_task_state(self):
        task_return = TaskGroundingReturn()
        task_return.is_success = False
        extract_task_State = dialog_flow.ExtractTaskState(self.state_dict, self.container)
        self.container.speak = Mock()
        self.container.task_grounding.get_specific_task_from_task = Mock(return_value=task_return)
        extract_task_State.state_dict["last_received_sentence"] = "Dummy sentence"
        return_state = extract_task_State.execute()
        self.assertTrue(isinstance(return_state, dialog_flow.ValidateTaskState))

class ValidateTaskStateTest(unittest.TestCase):
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

    def test_validate_task__any_error__returns_ask_for_clarification_state(self):
        task_grounding_return = TaskGroundingReturn()
        error = TaskGroundingError()
        error.error_code = TaskErrorType.UNKNOWN
        error.error_task_name = "Dummy Task"
        task_grounding_return.error = error
        validate_task_state = dialog_flow.ValidateTaskState(self.state_dict, self.container)
        validate_task_state.state_dict['task_grounding_return'] = task_grounding_return
        self.container.speak = Mock()
        return_state = validate_task_state.execute()
        self.assertTrue(isinstance(return_state, dialog_flow.AskForClarificationState))

class AskForClarificationStateTest(unittest.TestCase):
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

    def test_clarify_task__first_run__returns_wait_response_state(self):
        error = TaskGroundingError()
        error.error_code = TaskErrorType.UNKNOWN
        error.error_task_name = "Dummy Task"
        clarify_state = dialog_flow.AskForClarificationState(self.state_dict, self.container)
        clarify_state.error = error
        self.container.speak = Mock()
        return_state = clarify_state.execute()
        self.assertTrue(isinstance(return_state, dialog_flow.WaitForResponseState))

    def test_clarify_task__unknown_task_with_affirmation__returns_start_teach_state(self):
        entities = [
            (EntityType.AFFIRMATION, "blue"),
            (EntityType.LOCATION, "cover"),
            (EntityType.OBJECT, "next"),
            (EntityType.LOCATION, "black"),
            (EntityType.OBJECT, "bottom cover"),
            (EntityType.LOCATION, "above"),
            (EntityType.OBJECT, "bottom cover")
        ]
        self.container.ner.get_entities = Mock(return_value=entities)
        error = TaskGroundingError()
        error.error_code = TaskErrorType.UNKNOWN
        error.error_task_name = "Dummy Task"
        clarify_state = dialog_flow.AskForClarificationState(self.state_dict, self.container)
        clarify_state.is_first_run = False
        clarify_state.error = error
        self.container.speak = Mock()
        return_state = clarify_state.execute()
        self.assertTrue(isinstance(return_state, dialog_flow.StartTeachState))

    def test_clarify_task__unknown_task_with_denial__returns_wait_for_greet_state(self):
        entities = [
            (EntityType.DENIAL, "blue"),
            (EntityType.LOCATION, "cover"),
            (EntityType.OBJECT, "next"),
            (EntityType.LOCATION, "black"),
            (EntityType.OBJECT, "bottom cover"),
            (EntityType.LOCATION, "above"),
            (EntityType.OBJECT, "bottom cover")
        ]
        self.container.ner.get_entities = Mock(return_value=entities)
        error = TaskGroundingError()
        error.error_code = TaskErrorType.UNKNOWN
        error.error_task_name = "Dummy Task"
        clarify_state = dialog_flow.AskForClarificationState(self.state_dict, self.container)
        clarify_state.is_first_run = False
        clarify_state.error = error
        self.container.speak = Mock()
        return_state = clarify_state.execute()
        self.assertTrue(isinstance(return_state, dialog_flow.WaitForGreetingState))

    def test_clarify_task__no_object_with_task__returns_extract_task_state(self):
        entities = [
            (EntityType.TASK, "blue"),
            (EntityType.LOCATION, "cover"),
            (EntityType.OBJECT, "next"),
            (EntityType.LOCATION, "black"),
            (EntityType.OBJECT, "bottom cover"),
            (EntityType.LOCATION, "above"),
            (EntityType.OBJECT, "bottom cover")
        ]
        self.container.ner.get_entities = Mock(return_value=entities)
        error = TaskGroundingError()
        error.error_code = TaskErrorType.NO_OBJECT
        error.error_task_name = "Dummy Task"
        clarify_state = dialog_flow.AskForClarificationState(self.state_dict, self.container)
        clarify_state.is_first_run = False
        clarify_state.error = error
        self.container.speak = Mock()
        return_state = clarify_state.execute()
        self.assertTrue(isinstance(return_state, dialog_flow.ExtractTaskState))

    def test_clarify_task__no_object_without_task__returns_wait_for_greet_state(self):
        entities = [
            (EntityType.LOCATION, "blue"),
            (EntityType.LOCATION, "cover"),
            (EntityType.OBJECT, "next"),
            (EntityType.LOCATION, "black"),
            (EntityType.OBJECT, "bottom cover"),
            (EntityType.LOCATION, "above"),
            (EntityType.OBJECT, "bottom cover")
        ]
        self.container.ner.get_entities = Mock(return_value=entities)
        error = TaskGroundingError()
        error.error_code = TaskErrorType.UNKNOWN
        error.error_task_name = "Dummy Task"
        clarify_state = dialog_flow.AskForClarificationState(self.state_dict, self.container)
        clarify_state.is_first_run = False
        clarify_state.error = error
        self.container.speak = Mock()
        return_state = clarify_state.execute()
        self.assertTrue(isinstance(return_state, dialog_flow.WaitForGreetingState))

class PerformTaskStateTest(unittest.TestCase):
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

    def test_perform_task__grounding_fails__returns_clarify_objects_state(self):
        grounding_return = GroundingReturn()
        grounding_return.is_success = False
        self.container.speak = Mock()
        self.container.robot.move_out_of_view = Mock()
        self.container.camera.get_image = Mock()
        self.container.camera.get_depth = Mock()
        self.container.grounding.find_object = Mock(return_value=grounding_return)
        perform_task_state = dialog_flow.PerformTaskState(self.state_dict, self.container)
        base_task = Task(name="Dummy name")
        base_task.objects_to_execute_on.append("dummy")
        task_grounding_return = TaskGroundingReturn()
        task_grounding_return.task_info.append(base_task)
        perform_task_state.state_dict["base_task"] = base_task
        perform_task_state.state_dict["task_grounding_return"] = task_grounding_return
        return_state = perform_task_state.execute()
        self.assertTrue(isinstance(return_state, dialog_flow.ClarifyObjects))

    def test_perform_task__fails_to_perform_task__returns_wait_for_greet_state(self):
        grounding_return = GroundingReturn()
        grounding_return.is_success = True
        self.container.speak = Mock()
        self.container.robot.move_out_of_view = Mock()
        self.container.camera.get_image = Mock()
        self.container.camera.get_depth = Mock()
        self.container.grounding.find_object = Mock(return_value=grounding_return)
        perform_task_state = dialog_flow.PerformTaskState(self.state_dict, self.container)
        base_task = Task(name="Dummy name")
        dummy_object = ObjectEntity(name="dummy")
        base_task.objects_to_execute_on.append(dummy_object)
        task_grounding_return = TaskGroundingReturn()
        task_grounding_return.task_info.append(base_task)
        perform_task_state.state_dict["base_task"] = base_task
        perform_task_state.state_dict["task_grounding_return"] = task_grounding_return
        return_state = perform_task_state.execute()
        self.assertTrue(isinstance(return_state, dialog_flow.WaitForGreetingState))

    def test_perform_task__succeeds_to_perform_task__returns_perform_task_state(self):
        grounding_return = GroundingReturn()
        grounding_return.is_success = True
        grounding_return.object_infos.append("dummy")
        self.container.speak = Mock()
        self.container.robot.move_out_of_view = Mock()
        self.container.camera.get_image = Mock()
        self.container.camera.get_depth = Mock()
        self.container.grounding.find_object = Mock(return_value=grounding_return)
        self.container.robot.pick_up = Mock(return_value=True)
        perform_task_state = dialog_flow.PerformTaskState(self.state_dict, self.container)
        base_task = Task(name="Dummy name")
        dummy_object = ObjectEntity(name="dummy")
        base_task.objects_to_execute_on.append(dummy_object)
        base_task.task_type = TaskType.PICK
        task_grounding_return = TaskGroundingReturn()
        task_grounding_return.task_info.append(base_task)
        perform_task_state.state_dict["base_task"] = base_task
        perform_task_state.state_dict["task_grounding_return"] = task_grounding_return
        return_state = perform_task_state.execute()
        self.assertTrue(isinstance(return_state, dialog_flow.PerformTaskState))

class ClarifyObjectStateTest(unittest.TestCase):
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

    def test_clarify_objects__first_run_unknown_object__returns_wait_for_greet(self):
        clarify_objects_state = dialog_flow.ClarifyObjects(self.state_dict, self.container)
        clarify_objects_state.error = GroundingErrorType.UNKNOWN
        self.container.speak = Mock()
        base_task = Task(name="Dummy name")
        dummy_object = ObjectEntity(name="dummy")
        base_task.objects_to_execute_on.append(dummy_object)
        base_task.task_type = TaskType.PICK
        task_grounding_return = TaskGroundingReturn()
        task_grounding_return.task_info.append(base_task)
        clarify_objects_state.state_dict["base_task"] = base_task
        clarify_objects_state.state_dict["task_grounding_return"] = task_grounding_return
        return_state = clarify_objects_state.execute()
        self.assertTrue(isinstance(return_state, dialog_flow.WaitForGreetingState))

    def test_clarify_objects__first_run_cant_find_object__returns_wait_response_state(self):
        clarify_objects_state = dialog_flow.ClarifyObjects(self.state_dict, self.container)
        clarify_objects_state.error = GroundingErrorType.CANT_FIND
        self.container.speak = Mock()
        base_task = Task(name="Dummy name")
        dummy_object = ObjectEntity(name="dummy")
        base_task.objects_to_execute_on.append(dummy_object)
        base_task.task_type = TaskType.PICK
        task_grounding_return = TaskGroundingReturn()
        task_grounding_return.task_info.append(base_task)
        clarify_objects_state.state_dict["base_task"] = base_task
        clarify_objects_state.state_dict["task_grounding_return"] = task_grounding_return
        return_state = clarify_objects_state.execute()
        self.assertTrue(isinstance(return_state, dialog_flow.WaitForResponseState))

    def test_clarify_objects__denial_no_restart_request__returns_wait_greet_state(self):
        entities = [
            (EntityType.DENIAL, "blue"),
            (EntityType.LOCATION, "cover"),
            (EntityType.OBJECT, "next"),
            (EntityType.LOCATION, "black"),
            (EntityType.OBJECT, "bottom cover"),
            (EntityType.LOCATION, "above"),
            (EntityType.OBJECT, "bottom cover")
        ]
        self.container.ner.get_entities = Mock(return_value=entities)
        clarify_objects_state = dialog_flow.ClarifyObjects(self.state_dict, self.container)
        clarify_objects_state.error = GroundingErrorType.CANT_FIND
        clarify_objects_state.is_first_run = False
        self.container.speak = Mock()
        return_state = clarify_objects_state.execute()
        self.assertTrue(isinstance(return_state, dialog_flow.WaitForGreetingState))

    def test_clarify_objects__cant_find_no_restart_request_grounding_fail__returns_wait_greet_state(self):
        entities = [
            (EntityType.LOCATION, "blue"),
            (EntityType.LOCATION, "cover"),
            (EntityType.OBJECT, "next"),
            (EntityType.LOCATION, "black"),
            (EntityType.OBJECT, "bottom cover"),
            (EntityType.LOCATION, "above"),
            (EntityType.OBJECT, "bottom cover")
        ]
        grounding_return = GroundingReturn()
        grounding_return.is_success = False
        grounding_return.error_code = GroundingErrorType.CANT_FIND
        self.container.ner.get_entities = Mock(return_value=entities)
        self.container.grounding.find_object = Mock(return_value=grounding_return)
        clarify_objects_state = dialog_flow.ClarifyObjects(self.state_dict, self.container)
        base_task = Task(name="Dummy name")
        dummy_object = ObjectEntity(name="dummy")
        base_task.objects_to_execute_on.append(dummy_object)
        base_task.task_type = TaskType.PICK
        task_grounding_return = TaskGroundingReturn()
        task_grounding_return.task_info.append(base_task)
        clarify_objects_state.state_dict["base_task"] = base_task
        clarify_objects_state.state_dict["task_grounding_return"] = task_grounding_return
        clarify_objects_state.error = GroundingErrorType.CANT_FIND
        clarify_objects_state.is_first_run = False
        self.container.speak = Mock()
        return_state = clarify_objects_state.execute()
        self.assertTrue(isinstance(return_state, dialog_flow.WaitForGreetingState))

    def test_clarify_objects__cant_find_no_restart_request_grounding_succeed__returns_perform_task_state(self):
        entities = [
            (EntityType.LOCATION, "blue"),
            (EntityType.LOCATION, "cover"),
            (EntityType.OBJECT, "next"),
            (EntityType.LOCATION, "black"),
            (EntityType.OBJECT, "bottom cover"),
            (EntityType.LOCATION, "above"),
            (EntityType.OBJECT, "bottom cover")
        ]
        grounding_return = GroundingReturn()
        grounding_return.is_success = True
        self.container.ner.get_entities = Mock(return_value=entities)
        self.container.grounding.find_object = Mock(return_value=grounding_return)
        clarify_objects_state = dialog_flow.ClarifyObjects(self.state_dict, self.container)
        base_task = Task(name="Dummy name")
        dummy_object = ObjectEntity(name="dummy")
        base_task.objects_to_execute_on.append(dummy_object)
        base_task.task_type = TaskType.PICK
        task_grounding_return = TaskGroundingReturn()
        task_grounding_return.task_info.append(base_task)
        clarify_objects_state.state_dict["base_task"] = base_task
        clarify_objects_state.state_dict["task_grounding_return"] = task_grounding_return
        clarify_objects_state.error = GroundingErrorType.CANT_FIND
        clarify_objects_state.is_first_run = False
        self.container.speak = Mock()
        return_state = clarify_objects_state.execute()
        self.assertTrue(isinstance(return_state, dialog_flow.PerformTaskState))

    def test_clarify_objects__cant_find_no_restart_request_grounding_fail_new_error__returns_clarify_state(self):
        entities = [
            (EntityType.LOCATION, "blue"),
            (EntityType.LOCATION, "cover"),
            (EntityType.OBJECT, "next"),
            (EntityType.LOCATION, "black"),
            (EntityType.OBJECT, "bottom cover"),
            (EntityType.LOCATION, "above"),
            (EntityType.OBJECT, "bottom cover")
        ]
        grounding_return = GroundingReturn()
        grounding_return.is_success = False
        grounding_return.error_code = GroundingErrorType.ALREADY_KNOWN
        self.container.ner.get_entities = Mock(return_value=entities)
        self.container.grounding.find_object = Mock(return_value=grounding_return)
        clarify_objects_state = dialog_flow.ClarifyObjects(self.state_dict, self.container)
        base_task = Task(name="Dummy name")
        dummy_object = ObjectEntity(name="dummy")
        base_task.objects_to_execute_on.append(dummy_object)
        base_task.task_type = TaskType.PICK
        task_grounding_return = TaskGroundingReturn()
        task_grounding_return.task_info.append(base_task)
        clarify_objects_state.state_dict["base_task"] = base_task
        clarify_objects_state.state_dict["task_grounding_return"] = task_grounding_return
        clarify_objects_state.error = GroundingErrorType.CANT_FIND
        clarify_objects_state.is_first_run = False
        self.container.speak = Mock()
        return_state = clarify_objects_state.execute()
        self.assertTrue(isinstance(return_state, dialog_flow.ClarifyObjects))

    def test_clarify_objects__requested_restart_affirmation__returns_wait_for_greet_state(self):
        entities = [
            (EntityType.AFFIRMATION, "blue"),
            (EntityType.LOCATION, "cover"),
            (EntityType.OBJECT, "next"),
            (EntityType.LOCATION, "black"),
            (EntityType.OBJECT, "bottom cover"),
            (EntityType.LOCATION, "above"),
            (EntityType.OBJECT, "bottom cover")
        ]
        grounding_return = GroundingReturn()
        grounding_return.is_success = True
        base_task = Task(name="Dummy name")
        base_task.task_type = TaskType.PICK
        task_grounding_return = TaskGroundingReturn()
        task_grounding_return.task_info.append(base_task)
        self.container.ner.get_entities = Mock(return_value=entities)
        self.container.grounding.find_object = Mock(return_value=grounding_return)
        clarify_objects_state = dialog_flow.ClarifyObjects(self.state_dict, self.container)
        clarify_objects_state.asked_for_restart = True
        clarify_objects_state.state_dict["base_task"] = base_task
        clarify_objects_state.state_dict["task_grounding_return"] = task_grounding_return
        clarify_objects_state.is_first_run = False
        self.container.speak = Mock()
        return_state = clarify_objects_state.execute()
        self.assertTrue(isinstance(return_state, dialog_flow.WaitForGreetingState))

    def test_clarify_objects__requested_restart_denial__returns_perform_task_state(self):
        entities = [
            (EntityType.DENIAL, "blue"),
            (EntityType.LOCATION, "cover"),
            (EntityType.OBJECT, "next"),
            (EntityType.LOCATION, "black"),
            (EntityType.OBJECT, "bottom cover"),
            (EntityType.LOCATION, "above"),
            (EntityType.OBJECT, "bottom cover")
        ]
        grounding_return = GroundingReturn()
        grounding_return.is_success = True
        base_task = Task(name="Dummy name")
        base_task.task_type = TaskType.PICK
        task_grounding_return = TaskGroundingReturn()
        task_grounding_return.task_info.append(base_task)
        self.container.ner.get_entities = Mock(return_value=entities)
        self.container.grounding.find_object = Mock(return_value=grounding_return)
        clarify_objects_state = dialog_flow.ClarifyObjects(self.state_dict, self.container)
        clarify_objects_state.asked_for_restart = True
        clarify_objects_state.state_dict["base_task"] = base_task
        clarify_objects_state.state_dict["task_grounding_return"] = task_grounding_return
        clarify_objects_state.is_first_run = False
        self.container.speak = Mock()
        return_state = clarify_objects_state.execute()
        self.assertTrue(isinstance(return_state, dialog_flow.PerformTaskState))

    def test_clarify_objects__no_proper_answer__returns_wait_for_response_state(self):
        entities = [
            (EntityType.LOCATION, "blue"),
            (EntityType.LOCATION, "cover"),
            (EntityType.OBJECT, "next"),
            (EntityType.LOCATION, "black"),
            (EntityType.OBJECT, "bottom cover"),
            (EntityType.LOCATION, "above"),
            (EntityType.OBJECT, "bottom cover")
        ]
        grounding_return = GroundingReturn()
        grounding_return.is_success = True
        base_task = Task(name="Dummy name")
        base_task.task_type = TaskType.PICK
        task_grounding_return = TaskGroundingReturn()
        task_grounding_return.task_info.append(base_task)
        self.container.ner.get_entities = Mock(return_value=entities)
        self.container.grounding.find_object = Mock(return_value=grounding_return)
        clarify_objects_state = dialog_flow.ClarifyObjects(self.state_dict, self.container)
        clarify_objects_state.state_dict["base_task"] = base_task
        clarify_objects_state.state_dict["task_grounding_return"] = task_grounding_return
        clarify_objects_state.is_first_run = False
        self.container.speak = Mock()
        return_state = clarify_objects_state.execute()
        self.assertTrue(isinstance(return_state, dialog_flow.WaitForResponseState))






