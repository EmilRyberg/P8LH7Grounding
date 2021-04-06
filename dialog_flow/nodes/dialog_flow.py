#!/usr/bin/env python3

import sys
import rospy
from find_objects.find_objects import ObjectInfo
from ner_lib.command_builder import CommandBuilder, FindTask, PickUpTask, SpatialType
from ner.srv import NER
from little_helper_interfaces.msg import Task, ObjectEntity, OuterObjectEntity, SpatialDescription, OuterTask, StringWithTimestamp
from vision_lib.ros_camera_interface import ROSCamera
from speech_to_text.speech_to_text import SpeechToText
from cv_bridge import CvBridge
import numpy as np
from robot_control.robot_control import RobotController
from grounding.grounding import Grounding
from vision_lib.vision_controller import VisionController
from grounding.spatial import SpatialRelation
from grounding.database_handler import DatabaseHandler
from text_to_speech.srv import TextToSpeech, TextToSpeechRequest
from std_msgs.msg import String


"""
Man i wish we followed the uml better
"""
class DialogFlow:
    def __init__(self, ner_model_path, ner_tag_path, feature_weights_path, db_path, background_image_file):
        self.first_convo_flag = True
        self.grounding = Grounding(db=DatabaseHandler(db_path=db_path),
                                   vision_controller=VisionController(background_image_file=background_image_file,
                                                                      weights_path=feature_weights_path),
                                   spatial=SpatialRelation())
        self.sentence = ""
        self.object_info = ObjectInfo()
        self.robot = RobotController()
        self.camera = ROSCamera()
        self.command_builder = CommandBuilder(ner_model_path, ner_tag_path)
        self.last_received_sentence = None
        self.last_received_sentence_timestamp = None

        #self.np_depth
        #self.np_rgb

        #ROS Publishers
        #self.learn_pub = rospy.Publisher('MainLearn', String, queue_size=10)
        rospy.wait_for_service("tts")
        self.tts = rospy.ServiceProxy("tts", TextToSpeech)
        self.speech_to_text_subscriber = rospy.Subscriber("speech_to_text", StringWithTimestamp, callback=self.speech_to_text_callback, queue_size=1)

    def controller(self):
        #check to see if initialising conversation and gets sentence
        if self.first_convo_flag == True:
            self.first_conversation()
            self.first_convo_flag = False
        else:
            self.continuing_conversation()

        #subscribe to the ner service
        # rospy.wait_for_service('ner')
        # try:
        #     ner_service = rospy.ServiceProxy('ner', NER)
        #     rospy.wait_for_service(ner_service)
        #     task = ner_service(sentence)
        # except rospy.ServiceException as e:
        #         print(f"Service call failed: {e}")

        rospy.loginfo(f"Got sentence: {self.last_received_sentence}")
        task = self.command_builder.get_task(self.last_received_sentence)

        #self.tts_pub.publish(f"Ok, just to be sure. You want me to: {task.type} the {task.object1.name} which is located {task.object1.spatial_descirption[0].spatial_type}")
        task_type = "pick up" if isinstance(task, PickUpTask) else "find" # TODO: Replace this later
        log_string = f"Ok, just to be sure. You want me to: {task_type} the {task.object_to_pick_up.name}"
        rospy.loginfo(log_string)
        self.tts(log_string)

        attempts = 0
        while attempts < 5:
            self.spin_until_new_sentence()
            if self.last_received_sentence.lower() == "no" or self.last_received_sentence.lower() == "no.":
                self.tts("Okay, I will restart my program")
                rospy.loginfo("Okay, I will restart my program")
                return
            elif self.last_received_sentence.lower() == "yes" or self.last_received_sentence.lower() == "yes.":
                break
            else:
                self.tts("Sorry, I did not understand what you just said. Please say yes or no.")
                rospy.loginfo("Sorry, I did not understand what you just said. Please say yes or no.")
                attempts += 1

        # TODO: Switch based on task type
        self.tts(f"Okay, I will now look for the {task.object_to_pick_up}")
        rospy.loginfo(f"Okay, I will now look for the {task.object_to_pick_up}")

        #To make sure robot is out of view, might be unecesarry
        #while not self.robot.is_out_of_view():
        self.robot.move_out_of_view()

        np_rgb = self.camera.get_image()
        np_depth = self.camera.get_depth()

        #TODO decide if we want the grounding node to be a service or just to use at as class normally
        # try:
        #     grounding_service = rospy.ServiceProxy('grounding', Grounding)
        #     self.object_info = grounding_service(task.object1, np_rgb_image)
        # except rospy.ServiceException as e:
        #     print(f"Service call failed: {e}")
        grounding_return = self.grounding.find_object(task.object_to_pick_up)

        # #TODO add known flag in the return from grounding
        # if object_info.known == 0:
        #     self.tts_pub.publish("Sorry, I could not find the object you wanted.")
        #     self.tts_pub.publish("I am able to learn objects")
        #     return #TODO maybe just look for new sentence here, instead of looping back to the beginning

        #TODO update when robot_controller is made
        #elif object_info.known == 1:
        if isinstance(task, PickUpTask):
            self.robot.pick_up(grounding_return.object_info, np_rgb, np_depth)
        #if isinstance(task, FindTask):
        #    find_control(object_info)
        # if task.type == "learn":
        #     learn_control(task.object1.name, np_rgb_image)

        self.tts("Done!")
        rospy.loginfo("Done!")

    def first_conversation(self):
        spoken_sentence = "Hello. What would you like me to do?"
        rospy.loginfo(spoken_sentence)
        self.tts(spoken_sentence)
        self.spin_until_new_sentence()

    def continuing_conversation(self):
        spoken_sentence = "Is there anything else you want me to do?"
        rospy.loginfo(spoken_sentence)
        self.tts(spoken_sentence)
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


if __name__ == '__main__':
    try:
        rospy.init_node('dialog_controller')
        dialog = DialogFlow(ner_model_path="ner_pytorch_model.bin", ner_tag_path="tags.txt",
                            feature_weights_path="triplet-epoch-9-loss-0.16331.pth", db_path="/home/emilryberg/Documents/Projects/p8_catkin_ws/src/p8/grounding/grounding.db",
                            background_image_file="background.png")
        while True:
            dialog.controller()
    except rospy.ROSInterruptException:
        pass

