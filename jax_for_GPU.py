from whisper_jax import FlaxWhisperPipline
import jax.numpy as jnp
import os

# import ipdb;ipdb.set_trace()
pipeline = FlaxWhisperPipline("openai/whisper-large-v2", dtype=jnp.bfloat16, batch_size=16)

from jax.experimental.compilation_cache import compilation_cache as cc

cc.set_cache_dir("./jax_cache")

audio ="./Record (online-voice-recorder.com).mp3"
# JIT compile the forward call - slow, but we only do once
text = pipeline(audio,task="translate",return_timestamps=True, language="Nepali")

print(text)
# used cached function thereafter - super fast!
text = pipeline(audio)

# let's check our transcription - looks spot on!
print(text)

audio_length_in_mins = len(audio["array"]) / audio["sampling_rate"] / 60
print(f"Audio is {audio_length_in_mins} mins.")


# transcribe using cached function
text = pipeline(audio)

print(text)

# compile the forward call with timestamps - slow but we only do once
outputs = pipeline(audio, return_timestamps=True)
text = outputs["text"]  # transcription
chunks = outputs["chunks"]  # transcription + timestamps

# use cached timestamps function - super fast!
outputs = pipeline(audio, return_timestamps=True)
text = outputs["text"] 
chunks = outputs["chunks"]


from whisper.utils import WriteVTT

output_dir = "./content"
audio_path = "test1234.vtt"
wt = WriteVTT(output_dir="./content")

with open(os.path.join(output_dir,audio_path), "w") as vtt:
    wt.write_result(result = outputs, file=vtt)
