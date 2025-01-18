# #main.py
# from openai import OpenAI

# FILE_LOCATION = "./audio/ENG AND NEPALI 1.m4a"

# audio_file = open(FILE_LOCATION, "rb")
# client = OpenAI()
# transcription = client.audio.translations.create(
#     model="whisper-1", 
#     file=audio_file,
#     response_format="verbose_json",
#     language="en"


# )

# # print(transcription.text)
# print(transcription)

# import subprocess

# try:
#     result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, check=True)
#     print("FFmpeg is available:")
#     print(result.stdout)
# except FileNotFoundError:
#     print("FFmpeg is not found. Ensure it is installed and added to PATH.")
# except subprocess.CalledProcessError as e:
#     print("FFmpeg returned an error:", e)

import whisper
# FILE_LOCATION = "./audio/ENG AND NEPALI 1.m4a"

model = whisper.load_model("turbo")
# result = model.transcribe(r"C:/Documents/projects/audio to script/audio/test.m4a")

task = "transcribe" # Default is "transcribe"
source_language = "Nepali"

audio_file = "C:/Documents/projects/audio to script/Record (online-voice-recorder.com).mp3"
# Simply pass "translate" into your task parameter.
result = model.transcribe(audio_file, task=task, fp16=False, language=source_language)

print(result["text"])

# def translate(audio_file):
#     options = dict(beam_size=5, best_of=5)
#     translate_options = dict(task="translate",fp16=False,**options)
#     result = model.transcribe(audio_file,**translate_options)
#     return result

# result = translate(audio_file)
# print(result["text"])

