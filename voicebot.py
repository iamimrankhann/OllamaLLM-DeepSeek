import whisper
import speech_recognition as sr
import os
import os
import speech_recognition as sr
from gtts import gTTS
import pygame

def record(name="voice.wav", path='.'):
    try:
        recognizer = sr.Recognizer()
        # print("Listening... Please speak into the microphone.")

        # Use the microphone as source
        with sr.Microphone() as source:
            # Calibrate for ambient noise for 1 second
            recognizer.adjust_for_ambient_noise(source, duration=1)
            print("Calibration complete. Please start speaking.****")
            
            # Listen for speech with a timeout and a phrase_time_limit
            # 'timeout' waits for speech to start, 'phrase_time_limit' sets max recording length
            audio = recognizer.listen(source, timeout=7, phrase_time_limit=10)

        # Define the full file path and save the audio data
        audio_file_path = os.path.join(path, name)
        with open(audio_file_path, "wb") as f:
            f.write(audio.get_wav_data())

        print(f"Recording saved to: {audio_file_path}")
        return audio_file_path

    except sr.WaitTimeoutError:
        print("No speech detected within the timeout period.")
        return None

    except Exception as e:
        print("An error occurred during recording:", str(e))
        return None

def asr_model(model_size='base'):
    try:
        model = whisper.load_model(model_size)

        return model
    except Exception as e:
        print("error as ",str(e))
        return None
def text_to_speech(text, filename="output.mp3", lang="en"):
    try:
        tts = gTTS(text=text, lang=lang, slow=False)
        tts.save(filename)
    except Exception as e:
        print("Error in TTS:", str(e))
def play_audio(audio_path):
    pygame.init()
    pygame.mixer.init()
    pygame.mixer.music.load(audio_path)
    pygame.mixer.music.play()

    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10) 
if __name__ == "__main__":
    record()
    print("recordinf Done now decodind it ******")
    play_audio('voice.wav')
    model = asr_model()
    text = model.transcribe('voice.wav', fp16=False)
    text_to_speech(text['text'])
    play_audio('output.mp3')
    print(text['text'])
