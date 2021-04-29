#!/usr/bin/env python3
import argparse
import rospy
from find_objects.find_objects import ObjectInfo
from ner_lib.command_builder import CommandBuilder, PickUpTask, FindTask, MoveTask, PlaceTask, SpatialType, ObjectEntity as ObjectEntityType
from little_helper_interfaces.msg import StringWithTimestamp
from vision_lib.ros_camera_interface import ROSCamera
from robot_control.robot_control import RobotController
from grounding_lib.grounding import Grounding
from vision_lib.vision_controller import VisionController
from grounding_lib.spatial import SpatialRelation
from database_handler.database_handler import DatabaseHandler
from text_to_speech.srv import TextToSpeech
from ui_interface_lib.ui_interface import UIInterface
import random


class DialogFlow:
    def __init__(self, ner_model_path, ner_tag_path, feature_weights_path, db_path, background_image_file, websocket_uri):
        rospy.init_node('dialog_controller')
        self.first_convo_flag = True
        self.database_handler = DatabaseHandler(db_path=db_path)
        self.grounding = Grounding(db=self.database_handler,
                                   vision_controller=VisionController(background_image_file=background_image_file,
                                                                      weights_path=feature_weights_path),
                                   spatial=SpatialRelation(database_handler=self.database_handler))
        self.sentence = ""
        self.object_info = ObjectInfo()
        self.robot = RobotController()
        self.camera = ROSCamera()
        self.command_builder = CommandBuilder(ner_model_path, ner_tag_path)
        self.last_received_sentence = None
        self.last_received_sentence_timestamp = None
        rospy.loginfo("Waiting for TTS service to come online")
        rospy.wait_for_service("tts")
        self.tts = rospy.ServiceProxy("tts", TextToSpeech)
        self.speech_to_text_subscriber = rospy.Subscriber("speech_to_text", StringWithTimestamp, callback=self.speech_to_text_callback, queue_size=1)
        self.ui_interface = UIInterface(websocket_uri)
        self.websocket_is_connected = self.ui_interface.connect()
        self.carrying_object = False

    def controller(self):
        #check to see if initialising conversation and gets sentence
        if self.first_convo_flag == True:
            self.first_conversation()
            self.first_convo_flag = False
        else:
            self.continuing_conversation()

        rospy.loginfo(f"Got sentence: {self.last_received_sentence}")
        if self.websocket_is_connected:
            self.ui_interface.send_as_user(self.last_received_sentence)
        task = self.command_builder.get_task(self.last_received_sentence)

        if not isinstance(task, (PickUpTask, FindTask, MoveTask, PlaceTask)):
            sentence = "Only default tasks are supported at the moment"
            rospy.loginfo(sentence)
            self.tts(sentence)
            if self.websocket_is_connected:
                self.ui_interface.send_as_robot(sentence)
            return

        log_string = f"Ok, just to be sure. You want me to {task.plaintext_name} the {self.build_object_sentence(task.objects_to_execute_on[0])}"
        rospy.loginfo(log_string)
        self.tts(log_string)
        if self.websocket_is_connected:
            self.ui_interface.send_as_robot(log_string)

        attempts = 0
        while attempts < 5:
            self.spin_until_new_sentence()
            if self.websocket_is_connected:
                self.ui_interface.send_as_user(self.last_received_sentence)
            if self.last_received_sentence.lower() == "no" or self.last_received_sentence.lower() == "no.":
                self.tts("Okay, I will restart my program")
                rospy.loginfo("Okay, I will restart my program")
                if self.websocket_is_connected:
                    self.ui_interface.send_as_robot("Okay, I will restart my program")
                return
            elif self.last_received_sentence.lower() == "yes" or self.last_received_sentence.lower() == "yes.":
                break
            else:
                self.tts("Sorry, I did not understand what you just said. Please say yes or no.")
                rospy.loginfo("Sorry, I did not understand what you just said. Please say yes or no.")
                if self.websocket_is_connected:
                    self.ui_interface.send_as_robot("Sorry, I did not understand what you just said. Please say yes or no.")
                attempts += 1

        if isinstance(task, PlaceTask):
            sentence = f"Okay, I will now try to {task.plaintext_name} this next to the {task.objects_to_execute_on[0].name}"
        else:
            sentence = f"Okay, I will now try to {task.plaintext_name} the {task.objects_to_execute_on[0].name}"
        self.tts(sentence)
        rospy.loginfo(sentence)
        if self.websocket_is_connected:
            self.ui_interface.send_as_robot(sentence)

        #To make sure robot is out of view, might be unecesarry
        #while not self.robot.is_out_of_view():
        self.robot.move_out_of_view()

        np_rgb = self.camera.get_image()
        np_depth = self.camera.get_depth()

        grounding_return = self.grounding.find_object(task.objects_to_execute_on[0])
        if not grounding_return.is_success:
            # TODO: Handle unknown object here
            sentence = "I could not find the object you were looking for. Restarting."
            self.tts(sentence)
            rospy.loginfo(sentence)
            if self.websocket_is_connected:
                self.ui_interface.send_as_robot(sentence)
            return
        else:
            if self.websocket_is_connected:
                self.ui_interface.send_images(np_rgb, grounding_return.object_info.object_img_cutout_cropped)

        success = True
        if isinstance(task, PickUpTask):
            success = self.robot.pick_up(grounding_return.object_info, np_rgb, np_depth)
            self.carrying_object = True

        elif isinstance(task, FindTask):
            success = self.robot.point_at(grounding_return.object_info, np_rgb, np_depth)

        elif isinstance(task, PlaceTask):
            if not self.carrying_object:
                self.ui_interface.send_as_robot("The place task could not be accomplished as no object is carried.")
                success = 0
            else:
                position = [200, -250, 100]
                success = self.robot.place(position)
                self.carrying_object = False

        elif isinstance(task, MoveTask):
            success = self.robot.pick_up(grounding_return.object_info, np_rgb, np_depth)
            if success:
                position = [200, -250, 100]
                success = self.robot.place(position)

        if success:
            self.tts("Done!")
            rospy.loginfo("Done!")
            if self.websocket_is_connected:
                self.ui_interface.send_as_robot(f"Done!")
        else:
            self.tts("The pick up task failed. I might have done something wrong. I'm sorry master.")
            rospy.loginfo("Finished program, with failure")
            if self.websocket_is_connected:
                self.ui_interface.send_as_robot("The pick up task failed. I might have done something wrong. I'm sorry master.")

    def first_conversation(self):
        spoken_sentence = "Hello. What would you like me to do?"
        rospy.loginfo(spoken_sentence)
        self.tts(spoken_sentence)
        if self.websocket_is_connected:
            self.ui_interface.send_as_robot(spoken_sentence)
        self.spin_until_new_sentence()

    def continuing_conversation(self):
        spoken_sentence = "Is there anything else you want me to do?"
        rospy.loginfo(spoken_sentence)
        self.tts(spoken_sentence)
        if self.websocket_is_connected:
            self.ui_interface.send_as_robot(spoken_sentence)
        self.spin_until_new_sentence()

    def spin_until_new_sentence(self):
        start_timestamp = rospy.get_rostime()
        got_new_sentence = False
        while not got_new_sentence:
            if self.last_received_sentence_timestamp is not None:
                time_difference = self.last_received_sentence_timestamp - start_timestamp
                #rospy.loginfo(f"Time diff: {time_difference}")
                if time_difference >= rospy.Duration.from_sec(0):
                    got_new_sentence = True
                    break
            rospy.sleep(rospy.Duration.from_sec(0.1))

    def speech_to_text_callback(self, data):
        rospy.logdebug(f"Got STT: {data}")
        self.last_received_sentence_timestamp = data.timestamp
        self.last_received_sentence = data.data

    def learn_control(self, name, image):
        self.tts("I will try and learn the new object")
        self.grounding.learn_new_object(name, image) #placeholder

    def pick_control(self, object_info, rgb, depth):
        self.tts(f"Okay, I will try to pick up the {object_info.name}") #might need to rework if we take the name out of objectinfo
        self.robot.pick_up(object_info, rgb, depth) #placeholder

    def find_control(self, object_info, rgb):
        self.tts(f"Okay, I will try to find the {object_info.name}")
        self.robot.find(object_info, rgb) #placeholder

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

