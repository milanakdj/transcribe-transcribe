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
FILE_LOCATION = "./audio/ENG AND NEPALI 1.m4a"

model = whisper.load_model("turbo")
# result = model.transcribe(r"C:/Documents/projects/audio to script/audio/test.m4a")

task = "translate" # Default is "transcribe"
source_language = "Nepali"

# Simply pass "translate" into your task parameter.
result = model.transcribe("C:/Documents/projects/audio to script/Record (online-voice-recorder.com) (1).mp3", task=task, fp16=False, language=source_language)

print(result["text"])
