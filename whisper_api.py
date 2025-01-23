from math import ceil
import glob
import os
from openai import OpenAI
from pydub import AudioSegment

FILE_LOCATION = "./0122.MP3"

audio_file = open(FILE_LOCATION, "rb")

audio_name = (FILE_LOCATION.split(".m")[0]).split("/")[1]
client = OpenAI()
# transcription = client.audio.translations.create(
#     model="whisper-1",
#     file=audio_file,
#     response_format="vtt"

# )

# print(transcription)
# # print(transcription.text)

# convert the mp4 to mp3
if not FILE_LOCATION.lower().endswith(".mp3"):
    AudioSegment.from_file(FILE_LOCATION).export(audio_name + ".mp3", format="mp3")


audio = AudioSegment.from_mp3(f"./{audio_name}.mp3")
print(audio_name)

# PyDub handles time in milliseconds
start_time = 0
ten_minutes = 10 * 60 * 1000
audio_length = len(audio)
print(audio_length)

for i in range(ceil(audio_length / ten_minutes)):
    slice_10_minutes = audio[start_time : start_time + ten_minutes]
    slice_10_minutes.export(
        f"./export_slice/audio_slice/{audio_name}_{i}.mp3", format="mp3"
    )
    start_time = start_time + ten_minutes

slice_directory = "./export_slice"
os.chdir(slice_directory)

complete_translation = ""

for audio in glob.glob("./audio_slice/*.mp3"):
    mp3_filename = os.path.splitext(os.path.basename(audio))[0] + ".mp3"
    if audio_name in mp3_filename:
        audio_file = open(f"./audio_slice/{mp3_filename}", "rb")
        transcription = client.audio.translations.create(
            model="whisper-1", file=audio_file, response_format="vtt"
        )

    # print(transcription)
    complete_translation = complete_translation + transcription

print(complete_translation)
with open(f"./text_slice/complete_timestamp_{audio_name.split('.')[0]}.txt", "w") as f:
        f.write(str(complete_translation))
with open(f"./text_slice/complete_timestamp_{audio_name.split('.')[0]}.vtt", "w") as f:
        f.write(complete_translation)
