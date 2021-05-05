from task_grounding.task_grounding import TaskGrounding, TaskGroundingReturn, TaskErrorType
from database_handler.database_handler import DatabaseHandler
import unittest
from unittest.mock import Mock
from ner_lib.ner import EntityType
from ner_lib.command_builder import Task, TaskType, ObjectEntity, SpatialType, SpatialDescription
import argparse
from enum import Enum
from find_objects_lib.find_objects import ObjectInfo
from ner_lib.command_builder import CommandBuilder, SpatialType, Task, TaskType, ObjectEntity as ObjectEntityType
from ner_lib.ner import NER, EntityType
from robot_control.robot_control import RobotController
from grounding_lib.grounding import Grounding, GroundingErrorType
from vision_lib.vision_controller import VisionController
from grounding_lib.spatial import SpatialRelation
from database_handler.database_handler import DatabaseHandler
from task_grounding.task_grounding import TaskGrounding, TaskGroundingError, TaskErrorType, TaskGroundingReturn
from ui_interface_lib.ui_interface import UIInterface
from typing import Type, Callable, Any, NewType, List
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