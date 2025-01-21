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
import os
# FILE_LOCATION = "./audio/ENG AND NEPALI 1.m4a"

model = whisper.load_model("large")
audio_file = "./record_one.mp3"
# result = model.transcribe(r"C:/Documents/projects/audio to script/audio/test.m4a")

task = "transcribe" # Default is "transcribe"
source_language = "Nepali"

# Simply pass "translate" into your task parameter.
# result = model.transcribe(audio_file, task=task, fp16=False, language=source_language)

# print(result["text"])

def translate(audio_file):
    # options = dict(beam_size=5, best_of=5)
    translate_options = dict(task="translate",fp16=False, language=source_language)
    result = model.transcribe(audio_file,**translate_options)
    return result

results = translate(audio_file)
print(results["text"])
print(results["segments"])

import uuid

from whisper.utils import WriteVTT

output_dir = "./content"
audio_path = f"convert_{uuid.uuid4()}.vtt"
wt = WriteVTT(output_dir="./content")

with open(os.path.join(output_dir,audio_path), "w") as vtt:
    wt.write_result(result = results, file=vtt)

subtilte = audio_path + ".vtt"

