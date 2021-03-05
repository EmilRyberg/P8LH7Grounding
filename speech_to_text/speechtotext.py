import speech_recognition as sr


class Speech_to_text:
    def __init__(self):
        self.r = sr.Recognizer()
        self.microphone = sr.Microphone()

    def Listener(self):
        with self.microphone as source: #used default microphone
            print("Listening for speech")
            audio = self.r.listen(source)
        return audio

    def Recognizer(self, audio):
        #using the SPhinx audio recognizer, there are others but require keys to use their services
        try:
            print("I think you said " + self.r.recognize_sphinx(audio))
            word = self.r.recognize_sphinx(audio)
        except sr.UnknownValueError:
            print("I could not understand what you said")
        except sr.RequestError as e:
            print("Sphinx error; {0}".format(e))
        
        return word

if __name__ == "__main__":
    speech_recog = Speech_recog()
    while True:
        audio = speech_recog.Listener()
        word = speech_recog.Recognizer(audio)