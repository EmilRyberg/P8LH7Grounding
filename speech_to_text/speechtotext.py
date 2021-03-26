import azure.cognitiveservices.speech as speechsdk


class SpeechToText:
    def __init__(self, key=None):
        self.speech_config = speechsdk.SpeechConfig(subscription=key,
                                               region="westeurope")
        self.speech_recognizer = speechsdk.SpeechRecognizer(speech_config=self.speech_config)

    def run(self):
        print("Speak into your microphone.")
        result = self.speech_recognizer.recognize_once_async().get()
        print(result.text)
        return result


if __name__ == "__main__":
    speech_recog = SpeechToText()
    while True:
        word = speech_recog.run()