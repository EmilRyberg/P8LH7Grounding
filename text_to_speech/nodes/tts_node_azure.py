#!/usr/bin/env python3
import rospy
from text_to_speech.srv import TextToSpeech, TextToSpeechRequest, TextToSpeechResponse
import azure.cognitiveservices.speech as speechsdk
import argparse


class TextToSpeechNode:

    def __init__(self, api_key):
        rospy.init_node("text_to_speech_server")
        self.service = rospy.Service("tts", TextToSpeech, self.handle_request)
        speech_config = speechsdk.SpeechConfig(subscription=api_key, region="northeurope")
        speech_config.speech_synthesis_voice_name = "Microsoft Server Speech Text to Speech Voice (en-AU, WilliamNeural)"
        self.speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config)
        self.service.spin()

    def handle_request(self, request: TextToSpeechRequest):
        message = request.message
        try:
            result = self.speech_synthesizer.speak_text_async(message).get()
            # Check result
            if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                rospy.logdebug("voice synthesised successfully")
            elif result.reason == speechsdk.ResultReason.Canceled:
                cancellation_details = result.cancellation_details
                rospy.logerr("Speech synthesis canceled: {}".format(cancellation_details.reason))
                if cancellation_details.reason == speechsdk.CancellationReason.Error:
                    rospy.logerr("Error details: {}".format(cancellation_details.error_details))
                raise Exception()
            return TextToSpeechResponse(success=True)
        except Exception as e:
            rospy.logerr(e)
            return TextToSpeechResponse(success=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-k", "--key", dest="api_key", help="The Azure API Key")
    args = parser.parse_args(rospy.myargv()[1:])
    tts_node = TextToSpeechNode(api_key=args.api_key)

