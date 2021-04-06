#!/usr/bin/env python3
import os
import pathlib
import rospy
import uuid
from pydub import AudioSegment
from pydub.playback import play
from gtts import gTTS
from std_msgs.msg import String
from tts_output.srv import TextToSpeech, TextToSpeechRequest, TextToSpeechResponse


class TextToSpeechNode:
    AUDIO_TEMP_DIR = "/tmp/tts_audio_temp"

    def __init__(self):
        rospy.init_node("text_to_speech_server")
        self.service = rospy.Service("tts", TextToSpeech, self.handle_request)
        self.audio_files = {}
        if not os.path.exists(self.AUDIO_TEMP_DIR):
            os.mkdir(self.AUDIO_TEMP_DIR)
        else:
            path = pathlib.Path(self.AUDIO_TEMP_DIR)
            for item in path.iterdir():
                item.unlink()
        self.service.spin()

    def handle_request(self, request: TextToSpeechRequest):
        message = request.message
        audio_file = None
        try:
            if message.lower() in self.audio_files.keys():
                audio_file = self.audio_files[message.lower()]
            else:
                tts = gTTS(message)
                id = str(uuid.uuid4())
                audio_file = self.AUDIO_TEMP_DIR + f"/{id}.mp3"
                self.audio_files[message.lower()] = audio_file
                tts.save(audio_file)
            rospy.loginfo(f"Playing file: {audio_file}")
            audio = AudioSegment.from_mp3(audio_file)
            play(audio)
            return TextToSpeechResponse(success=True)
        except Exception as e:
            rospy.logerr(e)
            return TextToSpeechResponse(success=False)


if __name__ == '__main__':
    tts_node = TextToSpeechNode()

