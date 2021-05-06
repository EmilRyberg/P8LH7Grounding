#!/usr/bin/env python3
import rospy
import argparse
from enum import Enum
from find_objects_lib.find_objects import ObjectInfo
from ner_lib.command_builder import CommandBuilder, SpatialType, Task, TaskType, ObjectEntity as ObjectEntityType
from ner_lib.ner import NER, EntityType
from vision_lib.ros_camera_interface import ROSCamera
from robot_control.robot_control import RobotController
from grounding_lib.grounding import Grounding, GroundingErrorType
from vision_lib.vision_controller import VisionController
from grounding_lib.spatial import SpatialRelation
from database_handler.database_handler import DatabaseHandler
from task_grounding.task_grounding import TaskGrounding, TaskGroundingError, TaskErrorType, TaskGroundingReturn
from ui_interface_lib.ui_interface import UIInterface
from typing import Type, Callable, Any, NewType, List
import random
from little_helper_interfaces.msg import StringWithTimestamp
from text_to_speech.srv import TextToSpeech




class DependencyContainer:
    def __init__(self, ner: NER, command_builder: CommandBuilder,
                 grounding: Grounding, speak,
                 send_human_sentence_to_gui,
                 camera: ROSCamera, ui_interface: UIInterface, task_grounding: TaskGrounding,
                 robot: RobotController):
        self.ner = ner
        self.command_builder = command_builder
        self.grounding = grounding
        self.speak = speak
        self.send_human_sentence_to_gui = send_human_sentence_to_gui
        self.camera = camera
        self.ui_interface = ui_interface
        self.task_grounding = task_grounding
        self.robot = robot


class StateMachine:
    def __init__(self, container: DependencyContainer):
        self.state_dict = {
            "last_received_sentence": "",
            "last_received_sentence_timestamp": None,
            "websocket_is_connected": False,
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

    def execute(self):
        got_greeting = False
        if got_greeting:
            return self.greet
        else:
            return self.greet


class GreetState(State):
    def __init__(self, state_dict, container: DependencyContainer, previous_state=None):
        super().__init__(state_dict, container, previous_state)
        self.wait_for_command_state = WaitForResponseState(state_dict, container, self)
        self.is_first_call = True

    def execute(self):
        if self.is_first_call:
            self.is_first_call = False
            spoken_sentence = "Hello."
            self.container.speak(spoken_sentence)
            return self.wait_for_command_state
        else:
            if self.state_dict["last_received_sentence"] is not None:
                entities = self.container.ner.get_entities(self.state_dict["last_received_sentence"])
                is_teach = any([x[0] == EntityType.TEACH for x in entities])
                has_task = any([x[0] == EntityType.TASK for x in entities])
                if not is_teach and has_task:
                    verify_command_state = VerifyCommandState(self.state_dict, self.container, self)
                    return verify_command_state
                elif is_teach:
                    teach_state = StartTeachState(self.state_dict, self.container, self)
                    return teach_state
            ask_for_command_state = AskForCommandState(self.state_dict, self.container, self)
            return ask_for_command_state

class AskForCommandState(State):
    def __init__(self, state_dict, container: DependencyContainer, previous_state=None):
        super().__init__(state_dict, container, previous_state)
        self.wait_for_command_state = WaitForResponseState(state_dict, container, self)
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
                has_task =  any([x[0] == EntityType.TASK for x in entities])
                if not is_teach and has_task:
                    verify_command_state = VerifyCommandState(self.state_dict, self.container, self)
                    return verify_command_state
                elif not is_teach: # Assumes that the user responded to whether the robot should do a task
                    affirmation = any([x[0] == EntityType.AFFIRMATION for x in entities])
                    denial = any([x[0] == EntityType.DENIAL for x in entities])
                    if affirmation:
                        self.container.speak("Okay, what would you like me to do?")
                        return self.wait_for_command_state
                    else:
                        self.container.speak("Okay, goodbye for now master skywalker")
                        wait_for_greet_state = WaitForGreetingState(self.state_dict, self.container, self)
                        return wait_for_greet_state

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
            is_teach = any([x[0] == EntityType.TEACH for x in entities]) # TODO make states for teaching
            if not is_teach:
                self.state_dict["base_task"] = self.container.command_builder.get_task(self.state_dict["last_received_sentence"])
                log_string = f"Ok, just to be sure. You want me to execute the task {self.state_dict['base_task'].name}?"
                self.container.speak(log_string)
                return self.wait_for_response
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
    def __init__(self, state_dict, container: DependencyContainer, previous_state=None):
        super().__init__(state_dict, container, previous_state)

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
            if rospy.get_rostime() - init_time_stamp >= rospy.Duration.from_sec(3):
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
            log_string = f"Sorry, I do not know the task {error.error_task}"
            self.container.speak(log_string)
        elif error.error_code == TaskErrorType.NO_OBJECT:
            log_string = f"Sorry, I don't know which object to perform the task {error.error_task.task_type.value}"
            self.container.speak(log_string)
        elif error.error_code == TaskErrorType.NO_SUBTASKS:
            log_string = f"Sorry, I don't know the sub tasks for the task {error.error_task}"
            self.container.speak(log_string)
        elif error.error_code == TaskErrorType.NO_SPATIAL:
            log_string = f"Sorry, I am missing a spatial description of where to perform the task task {error.error_task}"
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
                log_string = f"Do you want to teach me the task {self.error.error_task}?"
                self.container.speak(log_string)
            elif self.error.error_code == TaskErrorType.NO_OBJECT:
                # TODO make logic for adding object to the task instead of having to re-do the task.
                log_string = f"Please repeat what you want me to do, and remember to specify the object I need to perform the task on."
                self.container.speak(log_string)
            elif self.error.error_code == TaskErrorType.NO_SUBTASKS:
                log_string = f"Do you want to teach me how to perform the task {self.error.error_task}?"
                self.container.speak(log_string)
            elif self.error.error_code == TaskErrorType.NO_SPATIAL:
                # TODO make logic for adding object to the task instead of having to re-do the task.
                log_string = f"Please repeat what you want me to do, and remember to specify the spatial description."
                self.container.speak(log_string)
            return self.wait_for_response_state
        else: # Got response
            start_teach = StartTeachState(self.state_dict, self.container, self)
            extract_task = ExtractTaskState(self.state_dict, self.container, self)
            if self.error.error_code == TaskErrorType.UNKNOWN:
                return start_teach
            elif self.error.error_code == TaskErrorType.NO_OBJECT:
                return extract_task
            elif self.error.error_code == TaskErrorType.NO_SUBTASKS:
                return start_teach
            elif self.error.error_code == TaskErrorType.NO_SPATIAL:
                return extract_task


class PerformTaskState(State):
    def __init__(self, state_dict, container: DependencyContainer, previous_state=None):
        super().__init__(state_dict, container, previous_state)

    def execute(self):
        log_string = f"I will now execute the task {self.state_dict['base_task'].name}."
        self.container.speak(log_string)

        if self.state_dict["task_grounding_return"].task_info:
            task = self.state_dict['task_grounding_return'].task_info[0]
            self.container.robot.move_out_of_view()

            np_rgb = self.container.camera.get_image()
            np_depth = self.container.camera.get_depth()

            grounding_return = None
            if task.task_type != TaskType.PLACE:
                grounding_return = self.container.grounding.find_object(task.objects_to_execute_on[0])
                if not grounding_return.is_success:
                    self.state_dict["grounding_error"] = grounding_return.error_code
                    clarify_objects_state = ClarifyObjects(self.state_dict, self.container, self)
                    return clarify_objects_state
                else:
                    if self.state_dict["websocket_is_connected"]:
                        self.container.ui_interface.send_images(np_rgb, grounding_return.object_info[0].object_img_cutout_cropped)
            success = False
            if task.task_type == TaskType.PICK:
                success = self.container.robot.pick_up(grounding_return.object_info[0], np_rgb, np_depth)
                self.state_dict["carrying_object"] = True

            elif task.task_type == TaskType.FIND:
                success = self.container.robot.point_at(grounding_return.object_info[0], np_rgb, np_depth)

            elif task.task_type == TaskType.PLACE:
                #if not self.state_dict["carrying_object"]:
                #    self.container.speak("The place task could not be accomplished as no object is carried.")
                #    success = False
                #else:
                x,y = self.container.grounding.get_location(task.objects_to_execute_on[0])
                position = [x, y, 50]
                success = self.container.robot.place(position, np_rgb)
                self.state_dict["carrying_object"] = False

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
                self.container.speak(f"Sorry master, I don't know the object: {self.state_dict['task_grounding_return'].task_info[0].objects_to_execute_on[0].name}."
                                     f" Please make sure I know it") # TODO add support for teaching objects.
                return wait_for_greet_state
            elif self.error == GroundingErrorType.CANT_FIND:
                self.container.speak(f"Sorry master, I could not find the object: {self.state_dict['task_grounding_return'].task_info[0].objects_to_execute_on[0].name}."
                                     f" Please make sure it is on the table. If it is, I think I need to get my feature database updated.")
                self.container.speak("Is the object clearly visible on the table now?")
                return self.wait_response_state
            elif self.error == GroundingErrorType.ALREADY_KNOWN:
                pass # TODO use this when adding support for teaching objects.
            else:
                self.container.speak("Got unknown error from visual grounding. Please fix my code master.")
                return wait_for_greet_state
        else:
            # Got response
            entities = self.container.ner.get_entities(self.state_dict["last_received_sentence"])
            denial = any([x[0] == EntityType.DENIAL for x in entities])
            if denial and not self.asked_for_restart:
                self.container.speak("Okay, since you said the object is not visible I can't perform my task. I will now restart.")
                return wait_for_greet_state
            elif self.error == GroundingErrorType.CANT_FIND and not self.asked_for_restart:
                self.container.speak("I will check if I can find the object now.")
                task = self.state_dict['task_grounding_return'].task_info[0]
                self.container.robot.move_out_of_view()
                grounding_return = self.container.grounding.find_object(task.objects_to_execute_on[0])
                if grounding_return.is_success:
                    self.container.speak("I could find the object now. I will resume my task.")
                    return perform_task_state
                elif grounding_return.error_code == GroundingErrorType.CANT_FIND:
                    self.container.speak(
                        f"Sorry master, I could still not find the object: {self.state_dict['task_grounding_return'].task_info[0].objects_to_execute_on[0].name}."
                        f" I think I need to get my features updated. I will now restart.") # TODO add support for updating features.
                    return wait_for_greet_state
                else:
                    self.state_dict["grounding_error"] = grounding_return.error_code
                    clean_clarify_objects = ClarifyObjects(self.state_dict, self.container, self)
                    return clean_clarify_objects
            elif self.asked_for_restart:
                entities = self.container.ner.get_entities(self.state_dict["last_received_sentence"])
                affirmation = any([x[0] == EntityType.AFFIRMATION for x in entities])
                denial = any([x[0] == EntityType.DENIAL for x in entities])
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
            task_name = self.state_dict["last_received_sentence"]  # assumed whole sentence is task name
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
            entites = self.container.ner.get_entities(self.state_dict["last_received_sentence"])
            task_words = [x[1] for x in entites if x[0] == EntityType.TASK]
            self.container.speak(f"I recognised the following words: {', '.join(task_words) if len(task_words) > 0 else 'none'}")
            task_sequence_state = AskForTaskSequenceState(self.state_dict, self.container, self.task_name, task_words, self)
            return task_sequence_state


class AskForTaskSequenceState(State):
    def __init__(self, state_dict, container: DependencyContainer, task_name, task_words, previous_state=None, future_error=None):
        super().__init__(state_dict, container, previous_state)
        self.task_name = task_name
        self.wait_for_response_state = WaitForResponseState(state_dict, container, self)
        self.task_words = task_words
        self.is_first_call = True
        self.future_error = future_error

    def execute(self):
        if self.is_first_call:
            self.is_first_call = False
            if self.future_error is not None:
                self.container.speak("Something went wrong, you might have specified a task that does not exist")
            else:
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
            ask_for_task_sequence_state = AskForTaskSequenceState(self.state_dict, self.container, self.task_name, self.task_words, self, future_error="fail")
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
            ask_for_task_sequence_state = AskForTaskSequenceState(self.state_dict, self.container, self.task_name,
                                                                  self.task_words, self, task_return_info.error)
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
            human_task_text = None
            self.is_first_run = False
            for task in self.tasks:
                if human_task_text is None:
                    human_task_text = task.task_type.value
                else:
                    human_task_text += f"then {task.task_type.value}"
            self.container.speak(f"I have constructed the task sequence you told me to perform {human_task_text}. Is this correct?")
            return self.wait_for_response_state
        else:
            entities = self.container.ner.get_entities(self.state_dict["last_received_sentence"])
            if len([x for x in entities if x[0] == EntityType.AFFIRMATION]) > 0:
                self.state_dict["new_task_sequence"].extend(self.tasks)
                ask_if_more_steps_state = AskIfMoreStepsState(self.state_dict, self.container, self.task_name, self.task_words, self)
                return ask_if_more_steps_state
            elif len([x for x in entities if x[0] == EntityType.DENIAL]) > 0:
                self.container.speak("I heard you said no. Let's try again.")
                ask_for_task_sequence_state = AskForTaskSequenceState(self.state_dict, self.container, self.task_name,
                                                                      self.task_words, self, "dummy error")
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
            self.container.speak("Is there more steps you want to add to this task?")
            self.is_first_run = False
            return self.wait_for_response_state
        else:
            entities = self.container.ner.get_entities(self.state_dict["last_received_sentence"])
            if len([x for x in entities if x[0] == EntityType.AFFIRMATION]) > 0:
                ask_for_task_sequence_state = AskForTaskSequenceState(self.state_dict, self.container, self.task_name,
                                                                      self.task_words, self)
                return ask_for_task_sequence_state
            elif len([x for x in entities if x[0] == EntityType.DENIAL]) > 0:
                task_return = self.container.task_grounding.teach_new_task(self.task_name, self.state_dict["new_task_sequence"], self.task_words)
                self.state_dict["new_task_sequence"] = []
                if task_return.is_success:
                    self.container.speak(f"I have now learned the task {self.task_name} and associated the words {', '.join(self.task_words)} with it.")
                    ask_for_command_state = AskForCommandState(self.state_dict, self.container, self)
                    return ask_for_command_state
                else:
                    rospy.logerr(f"Something went wrong when teaching task: {task_return.error.error_code}")
                    self.container.speak("Something went wrong.") # TODO: Handle different errors
                    ask_for_command_state = AskForCommandState(self.state_dict, self.container, self)
                    return ask_for_command_state
            else:
                self.container.speak("I did not get whether you said yes or no.")
                self.is_first_run = True
                return self


class DialogFlow:
    def __init__(self, ner_model_path, ner_tag_path, feature_weights_path, db_path, background_image_file, websocket_uri):
        rospy.init_node('dialog_controller')
        self.first_convo_flag = True
        self.database_handler = DatabaseHandler(db_path=db_path)
        self.grounding = Grounding(db=self.database_handler,
                                   vision_controller=VisionController(background_image_file=background_image_file,
                                                                      weights_path=feature_weights_path),
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
        self.carrying_object = False
        self.base_task = None
        self.tasks_to_perform = None
        self.task_grounding_return = None
        #self.current_state = DialogState.INITIALISE
        self.container = DependencyContainer(self.ner, self.command_builder, self.grounding, self.speak,
                                             self.send_human_sentence_to_GUI, self.camera, self.ui_interface, self.task_grounding, self.robot)
        self.state_machine = StateMachine(self.container)

    def start(self):
        self.state_machine.run()

    def send_human_sentence_to_GUI(self, sentence):
        rospy.loginfo(f"Got sentence: {sentence}")
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
        rospy.logdebug(f"Got STT: {data}")
        self.last_received_sentence_timestamp = data.timestamp
        self.last_received_sentence = data.data
        self.send_human_sentence_to_GUI(data.data)
        self.state_machine.got_new_speech_to_text(self.last_received_sentence, self.last_received_sentence_timestamp)

    def pick_control(self, object_info, rgb, depth):
        self.tts(f"Okay, I will try to pick up the {object_info.name}") # might need to rework if we take the name out of objectinfo
        self.robot.pick_up(object_info, rgb, depth) # placeholder

    def build_object_sentence(self, main_object: ObjectEntityType):
        sentence = main_object.name
        connection_variants = [
            "that is",
            "which is",
            "and it should be"
        ]
        for spatial_description in main_object.spatial_descriptions:
            spatial_type_word = self.spatial_type_to_human_adjective(spatial_description.spatial_type)
            sentence += f" {random.choice(connection_variants)} {spatial_type_word} the {spatial_description.object_entity.name}"
        return sentence

    def spatial_type_to_human_adjective(self, spatial_type: SpatialType):
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
                            background_image_file=args.background_image, websocket_uri=args.websocket_uri)
        dialog.start()
    except rospy.ROSInterruptException:
        pass

