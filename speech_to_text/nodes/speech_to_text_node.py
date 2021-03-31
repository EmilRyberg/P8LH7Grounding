#!/usr/bin/env python3
import rospy
import argparse
import azure.cognitiveservices.speech as speechsdk
from speech_to_text.speech_to_text import SpeechToText
from little_helper_interfaces.msg import StringWithTimestamp


class SpeechToTextNode:
    def __init__(self, api_key):
        rospy.init_node("speech_to_text")
        self.speech_to_text = SpeechToText(key=api_key)
        self.publisher = rospy.Publisher("speech_to_text", StringWithTimestamp, queue_size=1)

    def run(self):
        while not rospy.is_shutdown():
            result = self.speech_to_text.wait_for_speech_and_get_text()
            if result.reason == speechsdk.ResultReason.RecognizedSpeech:
                text = result.text
                rospy.loginfo(f"Text: {text}")
                time = rospy.get_rostime()
                message = StringWithTimestamp(data=text, timestamp=time)
                self.publisher.publish(message)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-k", "--key", dest="api_key", help="The Azure API Key")
    args = parser.parse_args(rospy.myargv()[1:])
    node = SpeechToTextNode(args.api_key)
    node.run()