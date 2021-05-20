#!/usr/bin/env python3
import rospy
import argparse
from enum import Enum
from find_objects_lib.find_objects import ObjectInfo
from ner_lib.command_builder import CommandBuilder, SpatialType, Task, TaskType, ObjectEntity as ObjectEntityType, SpatialDescription
from ner_lib.ner import NER, EntityType
from vision_lib.ros_camera_interface import ROSCamera
from robot_control.robot_control import RobotController
from grounding_lib.grounding import Grounding, GroundingErrorType, GroundingReturn
from vision_lib.vision_controller import VisionController
from grounding_lib.spatial import SpatialRelation
from database_handler.database_handler import DatabaseHandler
from task_grounding.task_grounding import TaskGrounding, TaskGroundingError, TaskErrorType, TaskGroundingReturn
from ui_interface_lib.ui_interface import UIInterface
from typing import Type, Callable, Any, NewType, List
import random
from little_helper_interfaces.msg import StringWithTimestamp
from text_to_speech.srv import TextToSpeech


def tasks_to_human_sentence(tasks: List[Task]):
    sentence = task_to_human_sentence(tasks[0])
    if len(tasks) > 1:
        for i, other_task in enumerate(tasks[1:]):
            if i == len(tasks[1:]) - 1:
                sentence += f", finally {task_to_human_sentence(other_task)}"
            else:
                sentence += f", then {task_to_human_sentence(other_task)}"
    return sentence


def task_to_human_sentence(task: Task):
    # example: pick up the blue cover which is next to the black cover, then place it in the top right corner
    sentence = task.name
    if not task.task_type == TaskType.PLACE:
        sentence += f" the {build_object_sentence(task.objects_to_execute_on[0])}"
    else:
        sentence += f" it{build_object_sentence(task.objects_to_execute_on[0], skip_main_object=True)}"

    return sentence


def build_object_sentence(main_object: ObjectEntityType, skip_main_object=False):
    sentence = main_object.name if not skip_main_object else ""
    connection_variants = [
        "that is",
        "which is",
        "and it should be"
    ]
    for spatial_description in main_object.spatial_descriptions:
        if spatial_description.spatial_type == SpatialType.OTHER:
            if "in the" not in spatial_description.object_entity.name:
                sentence += f" in the {spatial_description.object_entity.name}"
            else:
                sentence += f" {spatial_description.object_entity.name}"
        else:
            spatial_type_word = spatial_type_to_human_adjective(spatial_description.spatial_type)
            sentence += f" {random.choice(connection_variants)} {spatial_type_word} the {spatial_description.object_entity.name}"
    return sentence


def spatial_type_to_human_adjective(spatial_type: SpatialType):
    if spatial_type == SpatialType.NEXT_TO:
        return "next to"
    elif spatial_type == SpatialType.TOP_OF:
        return "to the top of"
    elif spatial_type == SpatialType.BOTTOM_OF:
        return "to the bottom of"
    elif spatial_type == SpatialType.LEFT_OF:
        return "to the left of"
    elif spatial_type == SpatialType.RIGHT_OF:
        return "to the right of"
    elif spatial_type == SpatialType.BELOW:
        return "below"
    elif spatial_type == SpatialType.ABOVE:
        return "above"

    return "I should not be saying this"


class DependencyContainer:
    def __init__(self, ner: NER, command_builder: CommandBuilder,
                 grounding: Grounding, speak,
                 send_human_sentence_to_gui,
                 camera: ROSCamera, ui_interface: UIInterface, task_grounding: TaskGrounding,
                 robot: RobotController, vision_controller: VisionController):
        self.ner = ner
        self.command_builder = command_builder
        self.grounding = grounding
        self.speak = speak
        self.send_human_sentence_to_gui = send_human_sentence_to_gui
        self.camera = camera
        self.ui_interface = ui_interface
        self.task_grounding = task_grounding
        self.robot = robot
        self.vision_controller = vision_controller


class StateMachine:
    def __init__(self, container: DependencyContainer, web_socket_is_connected):
        self.state_dict = {
            "last_received_sentence": "",
            "last_received_sentence_timestamp": None,
            "websocket_is_connected": web_socket_is_connected,
            "task_grounding_return": None,
            "base_task": Task,
            "wait_response_called_from": None,
            "tasks_to_perform": None,
            "carrying_object": False,
            "grounding_error": None,
        }

        self.container = container
        self.current_state = DefaultState(self.state_dict, self.container)
        self.state_stack = []

    def got_new_speech_to_text(self, message, timestamp):
        if message is not None or message.strip() != "":
            self.state_dict["last_received_sentence"] = message
            self.state_dict["last_received_sentence_timestamp"] = timestamp

    def run(self):
        while self.current_state is not None:
            new_state = self.current_state.execute()
            if new_state:
                self.current_state = new_state


class State:
    def __init__(self, state_dict, container: DependencyContainer, previous_state=None):
        self.state_dict = state_dict
        self.container = container
        self.previous_state = previous_state

    def execute(self):
        pass


class DefaultState(State):
    def __init__(self, state_dict, container: DependencyContainer):
        super().__init__(state_dict, container)
        self.wait_for_greet = WaitForGreetingState(state_dict, container)

    def execute(self):
        return self.wait_for_greet


class WaitForGreetingState(State): # Not used right now
    def __init__(self, state_dict, container: DependencyContainer, previous_state=None):
        super().__init__(state_dict, container, previous_state)
        self.greet = GreetState(state_dict, container, previous_state)
        self.wait_for_response = WaitForResponseState(state_dict, container, self)
        self.is_first_call = True

    def execute(self):
        if self.is_first_call:
            self.is_first_call = False
            return self.wait_for_response
        else:
            entities = self.container.ner.get_entities(self.state_dict["last_received_sentence"])
            has_greeting = any([x[0] == EntityType.GREETING for x in entities])
            if has_greeting:
                return self.greet
            else:
                self.is_first_call = True
                return self


class GreetState(State):
    def __init__(self, state_dict, container: DependencyContainer, previous_state=None):
        super().__init__(state_dict, container, previous_state)
        self.wait_for_command_state = WaitForResponseState(state_dict, container, self, timeout=7)
        self.is_first_call = True

    def execute(self):
        if self.is_first_call:
            self.is_first_call = False
            spoken_sentence = "G'day mate"
            self.container.speak(spoken_sentence)
            return self.wait_for_command_state
        else:
            if self.state_dict["last_received_sentence"] is not None:
                entities = self.container.ner.get_entities(self.state_dict["last_received_sentence"])
                is_teach = any([x[0] == EntityType.TEACH for x in entities])
                has_object = any(x[0] == EntityType.OBJECT for x in entities)
                has_task = any(x[0] == EntityType.TASK for x in entities)
                teach_task_state = StartTeachState(self.state_dict, self.container, self)
                if is_teach and has_object and has_task:
                    return teach_task_state
                elif is_teach and has_object:
                    return StartTeachObjectState(self.state_dict, self.container, self)
                elif not is_teach and has_task:
                    verify_command_state = VerifyCommandState(self.state_dict, self.container, self)
                    return verify_command_state
                elif is_teach:
                    return teach_task_state
            ask_for_command_state = AskForCommandState(self.state_dict, self.container, self)
            return ask_for_command_state


class AskForCommandState(State):
    def __init__(self, state_dict, container: DependencyContainer, previous_state=None):
        super().__init__(state_dict, container, previous_state)
        self.wait_for_command_state = WaitForResponseState(state_dict, container, self, timeout=7)
        self.is_first_call = True

    def execute(self):
        if self.is_first_call:
            self.is_first_call = False
            if isinstance(self.previous_state, GreetState):
                spoken_sentence = "What would you like me to do?"
                self.container.speak(spoken_sentence)
                return self.wait_for_command_state
            elif isinstance(self.previous_state, PerformTaskState):
                spoken_sentence = "Is there anything else you want me to do?"
                self.container.speak(spoken_sentence)
                return self.wait_for_command_state
            elif isinstance(self.previous_state, VerifyCommandState):
                log_string = "Ok, what would you then like me to do?"
                self.container.speak(log_string)
                return self.wait_for_command_state
        else: # Got a response
            if self.state_dict["last_received_sentence"] is not None:
                entities = self.container.ner.get_entities(self.state_dict["last_received_sentence"])
                is_teach = any([x[0] == EntityType.TEACH for x in entities])
                has_task = any([x[0] == EntityType.TASK for x in entities])
                has_object = any([x[0] == EntityType.OBJECT for x in entities])
                affirmation = any([x[0] == EntityType.AFFIRMATION for x in entities])
                denial = any([x[0] == EntityType.DENIAL for x in entities])
                start_teach_state = StartTeachState(self.state_dict, self.container, self)
                if is_teach and has_task and has_object:
                    return start_teach_state
                elif is_teach and has_object:
                    return StartTeachObjectState(self.state_dict, self.container, self)
                elif not is_teach and has_task:
                    verify_command_state = VerifyCommandState(self.state_dict, self.container, self)
                    return verify_command_state
                elif is_teach:
                    return start_teach_state
                elif affirmation or denial: # Assumes that the user responded to whether the robot should do a task
                    if affirmation:
                        self.container.speak("Okay, what would you like me to do?")
                        return self.wait_for_command_state
                    elif denial:
                        self.container.speak("Okay, goodbye for now master skywalker")
                        wait_for_greet_state = WaitForGreetingState(self.state_dict, self.container, self)
                        return wait_for_greet_state
                else:  # The user did not give a valid input
                    self.container.speak("I'm not sure what you wanted me to do. You have three options. I can "
                                         "perform one of my known tasks or you can teach me a new task or you can "
                                         "teach me how to recognise a new object. What would you like me to do?")
                    return self.wait_for_command_state
            else:
                spoken_sentence = "Sorry I did not get that, what can I do for you?"
                self.container.speak(spoken_sentence)
                return self.wait_for_command_state


class VerifyCommandState(State):
    def __init__(self, state_dict, container: DependencyContainer, previous_state=None):
        super().__init__(state_dict, container, previous_state)
        self.wait_for_response = WaitForResponseState(state_dict, container, self)
        self.is_first_call = True

    def execute(self):
        entities = self.container.ner.get_entities(self.state_dict["last_received_sentence"])
        if self.is_first_call:
            self.is_first_call = False
            is_teach = any([x[0] == EntityType.TEACH for x in entities])
            has_object = any(x[0] == EntityType.OBJECT for x in entities)
            has_task = any(x[0] == EntityType.TASK for x in entities)
            start_teach_state = StartTeachState(self.state_dict, self.container, self)
            if is_teach and has_object and has_task:
                return start_teach_state
            elif is_teach and has_object:
                return StartTeachObjectState(self.state_dict, self.container, self)
            elif not is_teach:
                self.state_dict["base_task"] = self.container.command_builder.get_task(self.state_dict["last_received_sentence"])
                log_string = f"Ok, just to be sure. You want me to execute the task {self.state_dict['base_task'].name}?"
                self.container.speak(log_string)
                return self.wait_for_response
            else:
                return start_teach_state
        else:  # Got response
            affirmation = any([x[0] == EntityType.AFFIRMATION for x in entities])
            denial = any([x[0] == EntityType.DENIAL for x in entities])
            if affirmation:
                extract_task = ExtractTaskState(self.state_dict, self.container, self)
                return extract_task
            elif denial:
                ask_for_command = AskForCommandState(self.state_dict, self.container, self)
                return ask_for_command
            else:
                log_string = f"Sorry, I did not catch that. Did you want me to execute the task {self.state_dict['base_task'].name}?"
                self.container.speak(log_string)
                return self.wait_for_response


class WaitForResponseState(State):
    def __init__(self, state_dict, container: DependencyContainer, previous_state=None, timeout=None):
        super().__init__(state_dict, container, previous_state)
        self.timeout = timeout

    def execute(self):
        got_new_sentence = False
        init_time_stamp = rospy.get_rostime()
        while not got_new_sentence:
            if self.state_dict["last_received_sentence_timestamp"] is not None:
                time_difference = self.state_dict["last_received_sentence_timestamp"] - init_time_stamp
                if time_difference >= rospy.Duration.from_sec(0):
                    got_new_sentence = True
                    break
            rospy.sleep(rospy.Duration.from_sec(0.1))
            if self.timeout and rospy.get_rostime() - init_time_stamp >= rospy.Duration.from_sec(self.timeout):
                self.state_dict["last_received_sentence"] = None
                got_new_sentence = True
        return self.previous_state


class ExtractTaskState(State):
    def __init__(self, state_dict, container: DependencyContainer, previous_state=None):
        super().__init__(state_dict, container, previous_state)

    def execute(self):
        self.state_dict["task_grounding_return"] = self.container.task_grounding.get_specific_task_from_task(self.state_dict['base_task'])
        if self.state_dict["task_grounding_return"].is_success:
            self.state_dict["tasks_to_perform"] = self.state_dict["task_grounding_return"].task_info
            perform_task_state = PerformTaskState(self.state_dict, self.container, self)
            return perform_task_state
        else:
            validate_task_state = ValidateTaskState(self.state_dict, self.container, self)
            return validate_task_state


class ValidateTaskState(State):
    def __init__(self, state_dict, container: DependencyContainer, previous_state=None):
        super().__init__(state_dict, container, previous_state)

    def execute(self):
        error = self.state_dict['task_grounding_return'].error
        if error.error_code == TaskErrorType.UNKNOWN:
            log_string = f"Sorry, I do not know the task {error.error_task_name}"
            self.container.speak(log_string)
        elif error.error_code == TaskErrorType.NO_OBJECT:
            log_string = f"Sorry, I don't know which object to perform the task {error.error_task_name}"
            self.container.speak(log_string)
        elif error.error_code == TaskErrorType.NO_SUBTASKS:
            log_string = f"Sorry, I don't know the sub tasks for the task {error.error_task_name}"
            self.container.speak(log_string)
        elif error.error_code == TaskErrorType.NO_SPATIAL:
            log_string = f"Sorry, I am missing a spatial description of where to perform the task task {error.error_task_name}"
            self.container.speak(log_string)
        ask_for_clarification_state = AskForClarificationState(self.state_dict, self.container, self)
        return ask_for_clarification_state


class AskForClarificationState(State):
    def __init__(self, state_dict, container: DependencyContainer, previous_state=None):
        super().__init__(state_dict, container, previous_state)
        self.is_first_run = True
        self.wait_for_response_state = WaitForResponseState(state_dict, container, self)
        self.error = self.state_dict['task_grounding_return'].error

    def execute(self):
        if self.is_first_run:
            self.is_first_run = False
            if self.error.error_code == TaskErrorType.UNKNOWN:
                log_string = f"Do you want to teach me the task {self.error.error_task_name}?"
                self.container.speak(log_string)
            elif self.error.error_code == TaskErrorType.NO_OBJECT:
                log_string = f"Please specify the object I need to perform the task on."
                self.container.speak(log_string)
            elif self.error.error_code == TaskErrorType.NO_SUBTASKS:
                log_string = f"Do you want to teach me how to perform the task {self.error.error_task_name}?"
                self.container.speak(log_string)
            elif self.error.error_code == TaskErrorType.NO_SPATIAL:
                log_string = f"Please repeat what you want me to do, and remember to specify the spatial description."
                self.container.speak(log_string)
            return self.wait_for_response_state
        else: # Got response
            start_teach_state = StartTeachState(self.state_dict, self.container, self)
            extract_task_state = ExtractTaskState(self.state_dict, self.container, self)
            wait_for_greet_state = WaitForGreetingState(self.state_dict, self.container, self)
            entities = self.container.ner.get_entities(self.state_dict["last_received_sentence"])
            if self.error.error_code == TaskErrorType.UNKNOWN or self.error.error_code == TaskErrorType.NO_SUBTASKS:
                affirmation = any([x[0] == EntityType.AFFIRMATION for x in entities])
                if affirmation:
                    return start_teach_state
                else:
                    self.container.speak("Sorry master, when I don't know a task I can't perform it. I will restart my program")
                    return wait_for_greet_state
            elif self.error.error_code == TaskErrorType.NO_OBJECT or self.error.error_code == TaskErrorType.NO_SPATIAL:
                has_object = any([x[0] == EntityType.OBJECT for x in entities])
                self.container.command_builder.add_entities_to_task(self.state_dict['base_task'], self.state_dict["last_received_sentence"])
                if has_object:
                    return extract_task_state
                else:
                    self.container.speak("Sorry master, seems like you didn't specify an object again. I will restart my program.")
                    return wait_for_greet_state


class PerformTaskState(State):
    def __init__(self, state_dict, container: DependencyContainer, previous_state=None):
        super().__init__(state_dict, container, previous_state)
        self.has_started_performing_task = False

    def execute(self):
        if self.state_dict["task_grounding_return"].task_info:
            task = self.state_dict['task_grounding_return'].task_info[0]

            if not self.has_started_performing_task:
                log_string = f"I will now {task_to_human_sentence(task)}."
                self.container.speak(log_string)
                self.has_started_performing_task = True

            success = self.perform_task(task)
            if not success:
                clarify_objects_state = ClarifyObjects(self.state_dict, self.container, self)
                return clarify_objects_state

            if success:
                del self.state_dict['task_grounding_return'].task_info[0] # Removing the task from the list, we just completed
                return self
            else:
                self.container.speak(f"I failed to perform the task: {task.task_type.value} on the object: {task.objects_to_execute_on[0].name}. I might have done something wrong. I'm sorry master. I will restart my program.")
                wait_for_greet_state = WaitForGreetingState(self.state_dict, self.container,
                                                                 self)  # TODO Should we reset the state dict here?
                return wait_for_greet_state
        else:
            # All tasks have been carried out
            self.container.speak("I have performed the task you requested!")
            ask_for_new_command_state = AskForCommandState(self.state_dict, self.container, self)
            return ask_for_new_command_state

    def perform_task(self, task):
        self.container.robot.move_out_of_view()
        np_rgb = self.container.camera.get_image()
        np_depth = self.container.camera.get_depth()

        grounding_return = None
        if task.task_type != TaskType.PLACE:
            grounding_return = self.container.grounding.find_object(task.objects_to_execute_on[0])
            if not grounding_return.is_success:
                self.state_dict["grounding_error"] = grounding_return.error_code
                return False
            else:
                if self.state_dict["websocket_is_connected"]:
                    self.container.ui_interface.send_images(np_rgb,
                                                            grounding_return.object_infos[0].object_img_cutout_cropped)
        success = False
        if task.task_type == TaskType.PICK:
            success = self.container.robot.pick_up(grounding_return.object_infos[0], np_rgb, np_depth)
            self.state_dict["carrying_object"] = True
        elif task.task_type == TaskType.FIND:
            success = self.container.robot.point_at(grounding_return.object_infos[0], np_rgb, np_depth)
        elif task.task_type == TaskType.PLACE:
            if not self.state_dict["carrying_object"]:
                self.container.speak("The place task could not be accomplished as no object is carried.")
                success = False
            else:
                positions, error = self.container.grounding.get_location(task.objects_to_execute_on[0])
                position = [round(positions[0][0]), round(positions[0][1]), 50]
                success = self.container.robot.place(position, np_rgb)
                self.state_dict["carrying_object"] = False
        return success


class ClarifyObjects(State):
    def __init__(self, state_dict, container: DependencyContainer, previous_state=None):
        super().__init__(state_dict, container, previous_state)
        self.wait_response_state = WaitForResponseState(state_dict, container, self)
        self.is_first_run = True
        self.error = self.state_dict["grounding_error"]
        self.asked_for_restart = False

    def execute(self):
        wait_for_greet_state = WaitForGreetingState(self.state_dict, self.container, self)
        perform_task_state = PerformTaskState(self.state_dict, self.container, self)
        if self.is_first_run:
            self.is_first_run = False
            if self.error == GroundingErrorType.UNKNOWN:
                self.container.speak(f"Sorry mate, I don't know the object: {self.state_dict['task_grounding_return'].task_info[0].objects_to_execute_on[0].name}."
                                     f" Please make sure I know it")
                return StartTeachObjectState(self.state_dict, self.container, self)
            elif self.error == GroundingErrorType.CANT_FIND:
                self.container.speak(f"Sorry my guy, I could not find the object: {self.state_dict['task_grounding_return'].task_info[0].objects_to_execute_on[0].name}."
                                     f" Please make sure it is on the table. If it is, I think I need to get my feature database updated.")
                self.container.speak("Is the object clearly visible on the table now?")
                return self.wait_response_state
            elif self.error == GroundingErrorType.CANT_FIND_RELATION:
                self.container.speak(f"Sorry mate, I couldn't find the object {self.state_dict['task_grounding_return'].task_info[0].objects_to_execute_on[0].name} "
                                     f"that matches the specified spatial relation.")
                self.container.speak("Do you want to retry?")
                return self.wait_response_state
            else:
                self.container.speak("Got unknown error from visual grounding. This is your fault.")
                return wait_for_greet_state
        else:
            # Got response
            entities = self.container.ner.get_entities(self.state_dict["last_received_sentence"])
            denial = any([x[0] == EntityType.DENIAL for x in entities])
            if denial and not self.asked_for_restart:
                self.container.speak("Okay, since you said the object is not visible I can't perform my task. I will now restart.")
                return wait_for_greet_state
            elif (self.error == GroundingErrorType.CANT_FIND or self.error == GroundingErrorType.CANT_FIND_RELATION) and not self.asked_for_restart:
                self.container.speak("I will check if I can find the object now.")
                task = self.state_dict['task_grounding_return'].task_info[0]
                self.container.robot.move_out_of_view()
                grounding_return = self.container.grounding.find_object(task.objects_to_execute_on[0])
                if grounding_return.is_success:
                    self.container.speak("I could find the object now. I will resume my task.")
                    return perform_task_state
                elif grounding_return.error_code == GroundingErrorType.CANT_FIND or self.error == GroundingErrorType.CANT_FIND_RELATION:
                    self.container.speak(
                        f"Sorry matey, I could still not find the object: {self.state_dict['task_grounding_return'].task_info[0].objects_to_execute_on[0].name}."
                        f" I think I need to get my features updated. I will now restart.") # TODO add support for updating features.
                    return wait_for_greet_state
                else:
                    self.state_dict["grounding_error"] = grounding_return.error_code
                    clean_clarify_objects = ClarifyObjects(self.state_dict, self.container, self)
                    return clean_clarify_objects
            elif self.asked_for_restart:
                entities = self.container.ner.get_entities(self.state_dict["last_received_sentence"])
                affirmation = any([x[0] == EntityType.AFFIRMATION for x in entities])
                if affirmation:
                    self.container.speak("Okay master. I'm sorry I failed you. I will now restart.")
                    return wait_for_greet_state
                else:
                    self.container.speak("Okay master. I will try to find the objects again from scratch.")
                    return perform_task_state
            else:
                self.container.speak("Sorry master, it seems like I'm having troubles. Should I restart my program?")
                self.asked_for_restart = True
                return self.wait_response_state


class StartTeachState(State):
    def __init__(self, state_dict, container: DependencyContainer, previous_state=None):
        super().__init__(state_dict, container, previous_state)
        self.verify_task_name_state = VerifyTaskNameState(state_dict, container, self)

    def execute(self):
        self.container.speak("What is the name of the task you want to teach me?")
        return self.verify_task_name_state


class VerifyTaskNameState(State):
    def __init__(self, state_dict, container: DependencyContainer, previous_state=None):
        super().__init__(state_dict, container, previous_state)
        self.wait_for_state = WaitForResponseState(state_dict, container, self)
        self.is_first_call = True

    def execute(self):
        if self.is_first_call:
            self.is_first_call = False
            return self.wait_for_state
        else:
            entities = self.container.ner.get_entities(self.state_dict["last_received_sentence"])
            task_words = [x[1] for x in entities if x[0] == EntityType.TASK]
            if len(task_words) == 0:
                self.container.speak("No task names found in what you said, please try again")
                self.is_first_call = True
                return self
            elif len(task_words) > 1:
                self.container.speak(f"I recognised multiple task names. The words i recognised are: {' and '.join(task_words)}. Please try again")
                self.is_first_call = True
                return self
            task_name = task_words[0]
            self.container.speak(f"I recognised the word, {task_name}, which will be associated with the name of the task")
            ask_for_task_words_state = AskForTaskWordsState(self.state_dict, self.container, task_name, self)
            return ask_for_task_words_state


class AskForTaskWordsState(State):
    def __init__(self, state_dict, container: DependencyContainer, task_name, previous_state=None):
        super().__init__(state_dict, container, previous_state)
        self.wait_for_state = WaitForResponseState(state_dict, container, self)
        self.task_name = task_name
        self.is_first_call = True

    def execute(self):
        if self.is_first_call:
            self.is_first_call = False
            self.container.speak("Which other words do you want to associate with this task?")
            return self.wait_for_state
        else:
            entities = self.container.ner.get_entities(self.state_dict["last_received_sentence"])
            task_words = [x[1] for x in entities if x[0] == EntityType.TASK]
            if len(task_words) > 0:
                self.container.speak(f"I recognised the following words: {', '.join(task_words)}")
            else:
                self.container.speak("I did not recognise any words. Let's try again")
                self.is_first_call = True
                return self
            task_sequence_state = AskForTaskSequenceState(self.state_dict, self.container, self.task_name, task_words, self)
            return task_sequence_state


class AskForTaskSequenceState(State):
    def __init__(self, state_dict, container: DependencyContainer, task_name, task_words, previous_state=None):
        super().__init__(state_dict, container, previous_state)
        self.task_name = task_name
        self.wait_for_response_state = WaitForResponseState(state_dict, container, self)
        self.task_words = task_words
        self.is_first_call = True

    def execute(self):
        if self.is_first_call:
            self.is_first_call = False
            self.container.speak("Tell me how I should carry out this task")
            return self.wait_for_response_state
        else:
            if "new_task_sequence" not in self.state_dict.keys() or self.state_dict["new_task_sequence"] is None:
                self.state_dict["new_task_sequence"] = []

            extract_task_state = ExtractTeachTaskState(self.state_dict, self.container, self.task_name, self.task_words, self)
            return extract_task_state


class ExtractTeachTaskState(State):
    def __init__(self, state_dict, container: DependencyContainer, task_name, task_words, previous_state=None):
        super().__init__(state_dict, container, previous_state)
        self.task_name = task_name
        self.task_words = task_words

    def execute(self):
        sentence = self.state_dict["last_received_sentence"]
        if sentence is None or sentence.strip() == "":
            self.container.speak("I did not get what you said")
            ask_for_task_sequence_state = AskForTaskSequenceState(self.state_dict, self.container, self.task_name, self.task_words, self)
            return ask_for_task_sequence_state
        task = self.container.command_builder.get_task(sentence)
        validate_task_state = ValidateTeachTaskState(self.state_dict, self.container, self.task_name, self.task_words, task, self)
        return validate_task_state


class ValidateTeachTaskState(State):
    def __init__(self, state_dict, container: DependencyContainer, task_name, task_words, task: Task, previous_state=None):
        super().__init__(state_dict, container, previous_state)
        self.task_name = task_name
        self.task_words = task_words
        self.task = task

    def execute(self):
        task_return_info = self.container.task_grounding.get_specific_task_from_task(self.task)
        if task_return_info.is_success:
            tasks = task_return_info.task_info
            ask_for_clarification_state = AskForClarificationTeachState(self.state_dict, self.container,
                                                                        self.task_name, self.task_words, tasks, self)
            return ask_for_clarification_state
        else:
            error = task_return_info.error
            if error.error_code == TaskErrorType.UNKNOWN:
                self.container.speak(f"Sorry, I do not know the task {error.error_task_name}")
            elif error.error_code == TaskErrorType.NO_OBJECT:
                self.container.speak(f"Sorry, I don't know on which object to perform the task {error.error_task_name}")
            elif error.error_code == TaskErrorType.NO_SUBTASKS:
                self.container.speak(f"Sorry, I don't know the sub tasks for the task {error.error_task_name}")
            elif error.error_code == TaskErrorType.NO_SPATIAL:
                self.container.speak(f"Sorry, I am missing a spatial description of where to perform the task {error.error_task_name}")
            ask_for_task_sequence_state = AskForTaskSequenceState(self.state_dict, self.container, self.task_name,
                                                                  self.task_words, self)
            return ask_for_task_sequence_state


class AskForClarificationTeachState(State):
    def __init__(self, state_dict, container: DependencyContainer, task_name, task_words, tasks: List[Task], previous_state=None):
        super().__init__(state_dict, container, previous_state)
        self.task_name = task_name
        self.task_words = task_words
        self.tasks = tasks
        self.is_first_run = True
        self.wait_for_response_state = WaitForResponseState(state_dict, container, self)

    def execute(self):
        if self.is_first_run:
            human_task_text = tasks_to_human_sentence(self.tasks)
            self.is_first_run = False
            self.container.speak(f"I have constructed the task sequence you told me to perform: {human_task_text}. Is this correct?")
            return self.wait_for_response_state
        else:
            entities = self.container.ner.get_entities(self.state_dict["last_received_sentence"])
            if any([x[0] == EntityType.AFFIRMATION for x in entities]):
                self.state_dict["new_task_sequence"].extend(self.tasks)
                ask_if_more_steps_state = AskIfMoreStepsState(self.state_dict, self.container, self.task_name, self.task_words, self)
                return ask_if_more_steps_state
            elif any([x[0] == EntityType.DENIAL for x in entities]):
                self.container.speak("I heard you said no. Let's try again.")
                ask_for_task_sequence_state = AskForTaskSequenceState(self.state_dict, self.container, self.task_name,
                                                                      self.task_words, self)
                return ask_for_task_sequence_state
            else:
                self.container.speak("I did not get whether you said yes or no.")
                self.is_first_run = True
                return self


class AskIfMoreStepsState(State):
    def __init__(self, state_dict, container: DependencyContainer, task_name, task_words, previous_state=None):
        super().__init__(state_dict, container, previous_state)
        self.task_name = task_name
        self.task_words = task_words
        self.is_first_run = True
        self.wait_for_response_state = WaitForResponseState(state_dict, container, self)

    def execute(self):
        if self.is_first_run:
            self.container.speak("Are there more steps you want to add to this task?")
            self.is_first_run = False
            return self.wait_for_response_state
        else:
            entities = self.container.ner.get_entities(self.state_dict["last_received_sentence"])
            if any([x[0] == EntityType.AFFIRMATION for x in entities]):
                ask_for_task_sequence_state = AskForTaskSequenceState(self.state_dict, self.container, self.task_name,
                                                                      self.task_words, self)
                return ask_for_task_sequence_state
            elif any([x[0] == EntityType.DENIAL for x in entities]):
                task_return = self.container.task_grounding.teach_new_task(self.task_name, self.state_dict["new_task_sequence"], self.task_words)
                self.state_dict["new_task_sequence"] = []
                if task_return.is_success:
                    self.container.speak(f"I have now learned the task {self.task_name} and associated the words {', '.join(self.task_words)} with it.")
                    ask_for_command_state = AskForCommandState(self.state_dict, self.container, self)
                    return ask_for_command_state
                else:
                    error = task_return.error
                    if error.error_code == TaskErrorType.ALREADY_KNOWN_TASK:
                        self.container.speak(f"The task {error.error_task_name} is already known and cannot be teached again")
                    elif error.error_code == TaskErrorType.ALREADY_USED_WORD:
                        self.container.speak(f"The word {error.error_task_name} has already been associated with a task")
                    elif error.error_code == TaskErrorType.UNKNOWN:
                        self.container.speak(f"The task {error.error_task_name} is not an already known task, and can therefore not be used to create a new task")
                    ask_for_command_state = AskForCommandState(self.state_dict, self.container, self)
                    return ask_for_command_state
            else:
                self.container.speak("I did not get whether you said yes or no.")
                self.is_first_run = True
                return self


class StartTeachObjectState(State):
    def __init__(self, state_dict, container: DependencyContainer, previous_state=None):
        super().__init__(state_dict, container, previous_state)
        self.ask_for_object_name_state = AskForObjectNameState(state_dict, container, self)
        self.is_first_call = True

    def execute(self):
        entities = self.container.ner.get_entities(self.state_dict["last_received_sentence"])
        object_name_entities = [x[1] for x in entities if x[0] == EntityType.OBJECT or x[0] == EntityType.COLOUR]
        if len(object_name_entities) > 0:
            object_name = " ".join(object_name_entities)
            self.container.speak(f"Do you want me to learn the object {object_name}?")
            return VerifyObjectNameState(self.state_dict, self.container, object_name)
        else:
            self.container.speak("What is the name of the new object you want to teach me?")
            return self.ask_for_object_name_state


class AskForObjectNameState(State):
    def __init__(self, state_dict, container: DependencyContainer, previous_state=None):
        super().__init__(state_dict, container, previous_state)
        self.wait_for_state = WaitForResponseState(state_dict, container, self)
        self.is_first_call = True

    def execute(self):
        if self.is_first_call:
            self.is_first_call = False
            return self.wait_for_state
        else:
            entities = self.container.ner.get_entities(self.state_dict["last_received_sentence"])
            object_words = [x[1] for x in entities if x[0] == EntityType.OBJECT or x[0] == EntityType.COLOUR]
            if len(object_words) == 0:
                self.container.speak("No object words found in what you said, please try again")
                self.is_first_call = True
                return self
            object_name = " ".join([x.lower() for x in object_words])
            object_exists = self.container.grounding.db.object_exists(object_name)
            if object_exists:
                self.container.speak(f"That object name, {object_name}, is already associated with an object. Please try a new one")
                self.is_first_call = True
                return self
            self.container.speak(f"I recognised, {object_name} which will be the name of the new object. Is this correct?")
            return VerifyObjectNameState(self.state_dict, self.container, object_name, self)


class VerifyObjectNameState(State):
    def __init__(self, state_dict, container: DependencyContainer, object_name, previous_state=None):
        super().__init__(state_dict, container, previous_state)
        self.wait_for_state = WaitForResponseState(state_dict, container, self)
        self.is_first_call = True
        self.object_name = object_name

    def execute(self):
        if self.is_first_call:
            self.is_first_call = False
            return self.wait_for_state
        else:
            entities = self.container.ner.get_entities(self.state_dict["last_received_sentence"])
            affirmation = any([x[0] == EntityType.AFFIRMATION for x in entities])
            denial = any([x[0] == EntityType.DENIAL for x in entities])
            if denial:
                self.container.speak(f"Okay, please say the name of the object again")
                return AskForObjectNameState(self.state_dict, self.container, self)
            elif affirmation:
                return AskClearTableState(self.state_dict, self.container, self.object_name, self)
            else:
                self.container.speak(f"I didn't get whether you said yes or no. Please confirm if {self.object_name} is correct.")
                return self.wait_for_state


class AskClearTableState(State):
    def __init__(self, state_dict, container: DependencyContainer, object_name, previous_state=None):
        super().__init__(state_dict, container, previous_state)
        self.wait_for_state = WaitForResponseState(state_dict, container, self)
        self.object_name = object_name
        self.is_first_call = True

    def execute(self):
        if self.is_first_call:
            self.is_first_call = False
            self.container.speak(f"Is the table clear of all objects except from one {self.object_name}?")
            return self.wait_for_state
        else:
            entities = self.container.ner.get_entities(self.state_dict["last_received_sentence"])
            affirmation = any([x[0] == EntityType.AFFIRMATION for x in entities])
            denial = any([x[0] == EntityType.DENIAL for x in entities])
            if denial:
                self.container.speak(f"Okay, please clear the table of all objects except for one {self.object_name}. Let me know when you have done that")
                self.is_first_call = True
                return self.wait_for_state
            elif affirmation:
                return VerifyCorrectObjectOnTableState(self.state_dict, self.container, self.object_name, self)
            else:
                self.container.speak(f"I didn't understand. Please clear the table of all objects except for one {self.object_name}, then reply yes if you have done so.")
                return self.wait_for_state


class VerifyCorrectObjectOnTableState(State):
    def __init__(self, state_dict, container: DependencyContainer, object_name, previous_state=None):
        super().__init__(state_dict, container, previous_state)
        self.wait_for_response_state = WaitForResponseState(state_dict, container, self)
        self.ask_clear_table_state = AskClearTableState(self.state_dict, self.container, object_name, self)
        self.object_name = object_name
        self.is_first_call = True

    def execute(self):
        if self.is_first_call:
            self.is_first_call = False
            self.container.robot.move_out_of_view()
            objects = self.container.vision_controller.get_masks_with_features()
            if len(objects) > 1:
                self.container.speak("I found more than one object on the table, please remove all other items")
                return self.ask_clear_table_state
            elif len(objects) == 0:
                self.container.speak("I found no objects")
                return self.ask_clear_table_state
            self.container.speak("I will now point to the object you want to teach me")
            object_info = objects[0]
            rgb_np = self.container.vision_controller.get_rgb()
            depth_np = self.container.vision_controller.get_depth()
            self.container.robot.point_at(object_info, rgb_np, depth_np)
            self.container.speak("Is this the correct item?")
            return self.wait_for_response_state
        else:
            entities = self.container.ner.get_entities(self.state_dict["last_received_sentence"])
            affirmation = any([x[0] == EntityType.AFFIRMATION for x in entities])
            denial = any([x[0] == EntityType.DENIAL for x in entities])
            if affirmation:
                self.container.speak(f"Okay, I will try to remember how this {self.object_name} looks for next time")
                self.container.robot.move_out_of_view()
                response = self.container.grounding.learn_new_object(self.object_name)
                if response.is_success:
                    self.container.speak(f"Object learned successfully. I should be able to start recognising {self.object_name}")
                    ask_for_command_state = AskForCommandState(self.state_dict, self.container, self)
                    return ask_for_command_state
                else:
                    if response.error_code == GroundingErrorType.ALREADY_KNOWN:
                        self.container.speak(f"Object is already known. Please try the teach object again from beginning.")
                        return StartTeachObjectState(self.state_dict, self.container, self)
                    elif response.error_code == GroundingErrorType.MULTIPLE_REF:
                        self.container.speak(f"Multiple objects detected. Have you cleared the table?")
                        return self.ask_clear_table_state
            elif denial:
                self.container.speak("Please try to move the item and make sure that no other objects are on the table")
                self.is_first_call = True
                return self
            else:
                self.container.speak("You did not say yes or no. Please confirm if it is the correct item")
                return self.wait_for_response_state


class DialogFlow:
    def __init__(self, ner_model_path, ner_tag_path, feature_weights_path, db_path, background_image_file, websocket_uri, debug=False):
        rospy.init_node('dialog_controller', log_level=(rospy.DEBUG if debug else rospy.INFO))
        self.database_handler = DatabaseHandler(db_path=db_path)
        self.vision_controller = VisionController(background_image_file=background_image_file,
                                                                      weights_path=feature_weights_path)
        self.grounding = Grounding(db=self.database_handler,
                                   vision_controller=self.vision_controller,
                                   spatial=SpatialRelation(database_handler=self.database_handler))
        self.task_grounding = TaskGrounding(self.database_handler)
        self.sentence = ""
        self.object_info = ObjectInfo()
        self.robot = RobotController()
        self.camera = ROSCamera()
        self.ner = NER(ner_model_path, ner_tag_path)
        self.command_builder = CommandBuilder(self.ner)
        self.last_received_sentence = None
        self.last_received_sentence_timestamp = None
        rospy.loginfo("Waiting for TTS service to come online")
        rospy.wait_for_service("tts")
        self.tts = rospy.ServiceProxy("tts", TextToSpeech)
        self.speech_to_text_subscriber = rospy.Subscriber("speech_to_text", StringWithTimestamp, callback=self.speech_to_text_callback, queue_size=1)
        self.ui_interface = UIInterface(websocket_uri)
        self.websocket_is_connected = self.ui_interface.connect()
        self.container = DependencyContainer(self.ner, self.command_builder, self.grounding, self.speak,
                                             self.send_human_sentence_to_GUI, self.camera, self.ui_interface,
                                             self.task_grounding, self.robot, self.vision_controller)
        self.state_machine = StateMachine(self.container, self.websocket_is_connected)

    def start(self):
        self.state_machine.run()

    def send_human_sentence_to_GUI(self, sentence):
        if self.websocket_is_connected:
            self.ui_interface.send_as_user(sentence)

    def send_robot_sentence_to_GUI(self, sentence):
        if self.websocket_is_connected:
            self.ui_interface.send_as_robot(sentence)

    def speak(self, sentence):
        self.tts(sentence)
        rospy.loginfo(sentence)
        self.send_robot_sentence_to_GUI(sentence)

    def speech_to_text_callback(self, data):
        rospy.logdebug(f"Got STT:\n{data}")
        self.last_received_sentence_timestamp = data.timestamp
        self.last_received_sentence = data.data
        self.send_human_sentence_to_GUI(data.data)
        self.state_machine.got_new_speech_to_text(self.last_received_sentence, self.last_received_sentence_timestamp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--ner_model", dest="ner_model_path", default="ner_pytorch_model.bin",
                        help="The path to the weight file for the NER model")
    parser.add_argument("-t", "--tag_file", dest="tags_path", default="tags.txt",
                        help="The path to the NER tags file")
    parser.add_argument("-f", "--feature_model", dest="feature_model", default="feature_extraction.pth",
                        help="The path to the feature extraction weight file")
    parser.add_argument("-d", "--db", dest="grounding_database", default="grounding.db",
                        help="The path to the grounding SQLite database")
    parser.add_argument("-b", "--background", dest="background_image", default="background.png",
                        help="The path to the background image")
    parser.add_argument("-w", "--ws_uri", dest="websocket_uri", default="ws://localhost:8765",
                        help="The URI to the websocket for the UI interface")
    args = parser.parse_args(rospy.myargv()[1:])
    try:
        dialog = DialogFlow(ner_model_path=args.ner_model_path, ner_tag_path=args.tags_path,
                            feature_weights_path=args.feature_model, db_path=args.grounding_database,
                            background_image_file=args.background_image, websocket_uri=args.websocket_uri, debug=True)
        dialog.start()
    except rospy.ROSInterruptException:
        pass

