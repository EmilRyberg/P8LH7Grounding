import azure.cognitiveservices.speech as speechsdk


class SpeechToText:
    def __init__(self, key=None):
        self.speech_config = speechsdk.SpeechConfig(subscription=key,
                                               region="westeurope")
        self.speech_recognizer = speechsdk.SpeechRecognizer(speech_config=self.speech_config)

    def wait_for_speech_and_get_text(self):
        return self.speech_recognizer.recognize_once_async().get()


if __name__ == "__main__":
    speech_recog = SpeechToText()
    while True:
        word = speech_recog.wait_for_speech_and_get_text()