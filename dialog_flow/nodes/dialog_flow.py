#!/usr/bin/env python3
import argparse
import rospy
from enum import Enum
from find_objects.find_objects import ObjectInfo
from ner_lib.command_builder import CommandBuilder, SpatialType, Task, TaskType, ObjectEntity as ObjectEntityType
from ner_lib.ner import NER, EntityType
from little_helper_interfaces.msg import StringWithTimestamp
from vision_lib.ros_camera_interface import ROSCamera
from robot_control.robot_control import RobotController
from grounding_lib.grounding import Grounding
from vision_lib.vision_controller import VisionController
from grounding_lib.spatial import SpatialRelation
from database_handler.database_handler import DatabaseHandler
from text_to_speech.srv import TextToSpeech
from task_grounding.task_grounding import TaskGrounding, TaskGroundingError, TaskErrorType, TaskGroundingReturn
from ui_interface_lib.ui_interface import UIInterface
from typing import Type, Callable, Any, NewType, List
import random


class DialogState(Enum):
    INITIALISE = 0
    WAIT_FOR_GREETING = 1
    GREET = 2
    ASK_FOR_COMMAND = 3
    WAIT_FOR_COMMAND = 4
    VERIFY_COMMAND = 5
    WAIT_FOR_VERIFICATION = 6
    EXTRACT_TASK = 7
    CHECK_FOR_MISSING_CLARIFICATION = 8
    ASK_FOR_CLARIFICATION = 9
    WAIT_FOR_CLARIFICATION = 10
    PROCESS_CLARIFICATION = 11
    PERFORM_TASK = 12
    ASK_FOR_FURTHER_INSTRUCTION = 13
    WAIT_FURTHER_INSTRUCTION = 14
    PROCESS_FURTHER_INSTRUCTION = 15


class DependencyContainer:
    def __init__(self, ner: NER, command_builder: CommandBuilder,
                 grounding: Grounding, speak,
                 send_human_sentence_to_gui,
                 camera: ROSCamera, ui_interface: UIInterface, task_grounding: TaskGrounding):
        self.ner = ner
        self.command_builder = command_builder
        self.grounding = grounding
        self.speak = speak
        self.send_human_sentence_to_gui = send_human_sentence_to_gui
        self.camera = camera
        self.ui_interface = ui_interface
        self.task_grounding = task_grounding


class StateMachine:
    def __init__(self, container: DependencyContainer):
        self.state_dict = {
            "last_received_sentence": "",
            "last_received_sentence_timestamp": None,
            "websocket_is_connected": False,
            "task_grounding_return": None,
            "base_task": Task,
            "wait_response_called_from": None
        }

        self.container = container
        self.current_state = DefaultState(self.state_dict, self.container)
        self.state_stack = []

    def got_new_speech_to_text(self, message, timestamp):
        self.state_dict["last_received_sentence"] = message
        self.state_dict["last_received_sentence_timestamp"] = timestamp

    def run(self):
        while True:
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


class WaitForGreetingState(State):
    def __init__(self, state_dict, container: DependencyContainer, previous_state=None):
        super().__init__(state_dict, container, previous_state)
        self.greet = GreetState(state_dict, container, previous_state)

    def execute(self):
        got_greeting = False
        if got_greeting:
            return self.greet
        else:
            return self.previous_state


class GreetState(State):
    def __init__(self, state_dict, container: DependencyContainer, previous_state=None):
        super().__init__(state_dict, container, previous_state)
        self.wait_for_command_state = WaitForResponseState(state_dict, container, self)

    def execute(self):
        spoken_sentence = "Hello."
        rospy.loginfo(spoken_sentence)
        self.container.speak(spoken_sentence)
        return self.wait_for_command_state


class AskForCommandState(State):
    def __init__(self, state_dict, container: DependencyContainer, previous_state=None):
        super().__init__(state_dict, container, previous_state)
        self.wait_for_command_state = WaitForResponseState(state_dict, container, self)

    def execute(self):
        if isinstance(self.previous_state.previous_state, GreetState):
            spoken_sentence = "What would you like me to do?"
            rospy.loginfo(spoken_sentence)
            self.container.speak(spoken_sentence)
            self.state_dict["wait_response_called_from"] = self
            return self.wait_for_command_state
        elif isinstance(self.previous_state, PerformTaskState):
            spoken_sentence = "I have now performed the task you requested. Is there anything else you want me to do?"
            rospy.loginfo(spoken_sentence)
            self.container.speak(spoken_sentence)
            self.state_dict["wait_response_called_from"] = self
            return self.wait_for_command_state


class VerifyCommandState(State):
    def __init__(self, state_dict, container: DependencyContainer, previous_state=None):
        super().__init__(state_dict, container, previous_state)
        self.wait_for_response = WaitForResponseState(state_dict, container, self)

    def execute(self):
        self.container.send_human_sentence_to_gui(self.state_dict["last_received_sentence"])
        entities = self.container.ner.get_entities(self.state_dict["last_received_sentence"])
        is_teach = any([x[0] == EntityType.TEACH for x in entities]) # TODO make states for teaching
        if not is_teach:
            self.state_dict["base_task"] = self.container.command_builder.get_task(self.state_dict["last_received_sentence"])
            log_string = f"Ok, just to be sure. You want me to execute the task {self.state_dict['base_task'].name}?"
            rospy.loginfo(log_string)
            self.container.speak(log_string)
            self.state_dict["wait_response_called_from"] = self
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
        return self.previous_state


class WaitForResponseStateOld(State):
    def __init__(self, state_dict, container: DependencyContainer, previous_state=None):
        super().__init__(state_dict, container, previous_state)
        self.previous_state = previous_state
        self.ask_for_command = AskForCommandState(state_dict, container, self)
        self.verify_command = VerifyCommandState(state_dict, container, self)
        self.extract_task = ExtractTaskState(state_dict, container, self)

    def execute(self):
        start_timestamp = rospy.get_rostime()
        got_new_sentence = False
        while not got_new_sentence:
            if self.state_dict["last_received_sentence_timestamp"] is not None:
                time_difference = self.state_dict["last_received_sentence_timestamp"] - start_timestamp
                #rospy.loginfo(f"Time diff: {time_difference}")
                if time_difference >= rospy.Duration.from_sec(0):
                    got_new_sentence = True
                    break
            rospy.sleep(rospy.Duration.from_sec(0.1))
        self.container.send_human_sentence_to_gui(self.state_dict["last_received_sentence"])
        entities = self.container.ner.get_entities(self.state_dict["last_received_sentence"])
        if isinstance(self.previous_state, GreetState) or isinstance(self.previous_state, AskForCommandState):
            return self.verify_command
        elif isinstance(self.previous_state, VerifyCommandState):
            affirmation = any([x[0] == EntityType.AFFIRMATION for x in entities])
            denial = any([x[0] == EntityType.DENIAL for x in entities])
            if affirmation:
                return self.extract_task
            elif denial:
                log_string = "Ok, what would you then like me to do?"
                rospy.loginfo(log_string)
                self.container.speak(log_string)
                return self
            else:
                log_string = f"Sorry, I did not catch that. Did you want me to execute the task {self.state_dict['base_task'].name}?"
                rospy.loginfo(log_string)
                self.container.speak(log_string)
                return self

class ExtractTaskState(State):
    def __init__(self, state_dict, container: DependencyContainer, previous_state=None):
        super().__init__(state_dict, container, previous_state)

    def execute(self):
        self.task_grounding_return = self.container.task_grounding.get_specific_task_from_task(self.state_dict['base_task'])
        if self.task_grounding_return.is_success:
            self.current_state = DialogState.PERFORM_TASK
            self.tasks_to_perform = self.task_grounding_return.task_info
        else:
            self.current_state = DialogState.CHECK_FOR_MISSING_CLARIFICATION

class ValidateTaskState(State):
    def __init__(self, state_dict, container: DependencyContainer, previous_state=None):
        super().__init__(state_dict, container, previous_state)

    def execute(self):
        error = self.state_dict['task_grounding_return'].error
        if error.error_code == TaskErrorType.UNKNOWN:
            log_string = f"Sorry, I do not know the task {error.error_task}"
            rospy.loginfo(log_string)
            self.container.speak(log_string)
        elif error.error_code == TaskErrorType.NO_OBJECT:
            log_string = f"Sorry, I don't know which object to perform the task {error.error_task.task_type.value}" # TODO ask Emil if this will work
            rospy.loginfo(log_string)
            self.container.speak(log_string)
        elif error.error_code == TaskErrorType.NO_SUBTASKS:
            log_string = f"Sorry, I don't know the sub tasks for the task {error.error_task}"
            rospy.loginfo(log_string)
            self.container.speak(log_string)
        elif error.error_code == TaskErrorType.NO_SPATIAL:
            log_string = f"Sorry, I am missing a spatial description of where to perform the task task {error.error_task}"
            rospy.loginfo(log_string)
            self.container.speak(log_string)

class AskForClarificationState(State):
    def __init__(self, state_dict, container: DependencyContainer, previous_state=None):
        super().__init__(state_dict, container, previous_state)

    def execute(self):
        # TODO ask the user for various specifications depending on the error type
        self.current_state=DialogState.WAIT_FOR_CLARIFICATION


class ProcessClarification(State):
    def __init__(self, state_dict, container: DependencyContainer, previous_state=None):
        super().__init__(state_dict, container, previous_state)

    def execute(self):
        # TODO check that the clarification seems sufficient and go to state 'extract task'
        self.current_state=DialogState.EXTRACT_TASK


class PerformTaskState(State):
    def __init__(self, state_dict, container: DependencyContainer, previous_state=None):
        super().__init__(state_dict, container, previous_state)

    def execute(self):
        log_string = f"I will now execute the task {self.state_dict['base_task'].name}."
        rospy.loginfo(log_string)
        self.container.speak(log_string)

        for task in self.state_dict['task_grounding_return'].task_info:
            # To make sure robot is out of view, might be unecesarry
            # while not self.robot.is_out_of_view():
            self.robot.move_out_of_view()

            np_rgb = self.container.camera.get_image()
            np_depth = self.container.camera.get_depth()

            grounding_return = self.grounding.find_object(task.objects_to_execute_on[0])
            if not grounding_return.is_success:
                # TODO: Handle unknown object here
                sentence = "I could not find the object you were looking for. Restarting."
                self.speak(sentence)
                return
            else:
                if self.state_dict["websocket_is_connected"]:
                    self.container.ui_interface.send_images(np_rgb, grounding_return.object_info.object_img_cutout_cropped)

            if task.task_type == TaskType.PICK:
                success = self.robot.pick_up(grounding_return.object_info, np_rgb, np_depth)
                self.carrying_object = True

            elif task.task_type== TaskType.FIND:
                success = self.robot.point_at(grounding_return.object_info, np_rgb, np_depth)

            elif task.task_type==TaskType.PLACE:
                if not self.carrying_object:
                    self.container.ui_interface.send_as_robot("The place task could not be accomplished as no object is carried.")
                    success = 0
                else:
                    position = [200, -250, 100]
                    success = self.robot.place(position)
                    self.carrying_object = False

            if success:
                self.tts("Done!")
                rospy.loginfo("Done!")
                if self.state_dict["websocket_is_connected"]:
                    self.container.ui_interface.send_as_robot(f"Done!")
            else:
                self.speak("The task failed. I might have done something wrong. I'm sorry master.")

class FindObjects(State):
    def __init__(self, state_dict, container: DependencyContainer, previous_state=None):
        super().__init__(state_dict, container, previous_state)

    def execute(self):
        got_greeting = False
        if got_greeting:
            return self.greet
        else:
            return self.previous_state

class ClarifyObjects(State):
    def __init__(self, state_dict, container: DependencyContainer, previous_state=None):
        super().__init__(state_dict, container, previous_state)

    def execute(self):
        got_greeting = False
        if got_greeting:
            return self.greet
        else:
            return self.previous_state


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
                # TODO: Next step
                pass
            elif len([x for x in entities if x[0] == EntityType.DENIAL]) > 0:
                self.container.speak("I heard you said no. Let's try again.")
                ask_for_task_sequence_state = AskForTaskSequenceState(self.state_dict, self.container, self.task_name,
                                                                      self.task_words, self, "dummy error")
                return ask_for_task_sequence_state
            else:
                #Neither affirmation or denial
                pass



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
        self.current_state = DialogState.INITIALISE
        self.container = DependencyContainer(self.ner, self.command_builder, self.grounding, self.speak,
                                             self.send_human_sentence_to_GUI, self.camera, self.ui_interface, self.task_grounding)
        self.state_machine = StateMachine(self.container)

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
        self.state_machine.got_new_speech_to_text(self.last_received_sentence, self.last_received_sentence_timestamp)

    def learn_control(self, name, image):
        self.tts("I will try and learn the new object")
        self.grounding.learn_new_object(name, image) # placeholder

    def pick_control(self, object_info, rgb, depth):
        self.tts(f"Okay, I will try to pick up the {object_info.name}") # might need to rework if we take the name out of objectinfo
        self.robot.pick_up(object_info, rgb, depth) # placeholder

    def find_control(self, object_info, rgb):
        self.tts(f"Okay, I will try to find the {object_info.name}")
        self.robot.find(object_info, rgb) # placeholder

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
        while True:
            dialog.controller()
    except rospy.ROSInterruptException:
        pass

