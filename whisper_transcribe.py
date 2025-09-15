import os
import argparse
import jax.numpy as jnp
from pathlib import Path
from whisper_jax import FlaxWhisperPipline
from jax.experimental.compilation_cache import compilation_cache as cc
# from whisper.utils import WriteVTT
from pydub import AudioSegment


def convert_to_mp3(file_path: str, output_name: str) -> str:
    """
    Convert input audio or video file to MP3 if not already in MP3 format.

    Args:
        file_path (str): Path to the input audio or video file.
        output_name (str): Base name for the output MP3 file.

    Returns:
        str: Path to the resulting MP3 file.
    """
    file_path = Path(file_path)

    # If it's already an MP3, just return the original path
    if file_path.suffix.lower() == ".mp3":
        return str(file_path)

    output_path = Path(f"{output_name}.mp3")

    try:
        # Pydub uses FFmpeg internally to handle both audio and video
        audio = AudioSegment.from_file(file_path)
        audio.export(output_path, format="mp3")
        print(f"Converted {file_path} to {output_path}")
    except Exception as e:
        raise RuntimeError(
            f"Failed to process file '{file_path}'. "
            f"Ensure FFmpeg is installed and accessible.\nError: {e}"
        )

    return str(output_path)


def setup_pipeline(model_name: str = "openai/whisper-tiny", cache_dir: str = "./jax_cache"):
    """
    Initialize the Whisper JAX pipeline with caching enabled.
    """
    cc.set_cache_dir(cache_dir)
    return FlaxWhisperPipline(model_name, dtype=jnp.bfloat16, batch_size=16)


def transcribe_audio(pipeline, audio_path: str, translate: bool = False):
    """
    Transcribe or translate audio using Whisper JAX.

    Args:
        pipeline: Whisper JAX pipeline.
        audio_path (str): Path to the audio file.
        translate (bool): If True, perform translation instead of transcription.

    Returns:
        dict: Transcription results including text and timestamps.
    """
    task = "translate" if translate else "transcribe"
    outputs = pipeline(audio_path, task=task, return_timestamps=True)
    return outputs

class WriteVTT2:
    def write_result(self, result: dict, file):
        file.write("WEBVTT\n\n")
        for segment in result.get("chunks", []):
            start = self._format_time(segment['timestamp'][0])
            end = self._format_time(segment['timestamp'][1])
            file.write(f"{start} --> {end}\n{segment['text'].strip()}\n\n")

    def _format_time(self, seconds: float) -> str:
        millis = int((seconds % 1) * 1000)
        seconds = int(seconds)
        mins, secs = divmod(seconds, 60)
        hours, mins = divmod(mins, 60)
        return f"{hours:02d}:{mins:02d}:{secs:02d}.{millis:03d}"


def save_as_vtt(outputs: dict, output_path: str):
    """
    Save transcription results to a VTT file.
    """
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    wt = WriteVTT2()
    with open(output_path, "w", encoding="utf-8") as vtt:
        wt.write_result(result=outputs, file=vtt)


def main():
    parser = argparse.ArgumentParser(description="Transcribe or translate audio/video using Whisper JAX.")
    parser.add_argument("--audio", required=True, help="Path to the input audio or video file (e.g., input.mp4, input.wav, input.mp3)")
    parser.add_argument("--output", required=True, help="Path to save the output VTT file (e.g., output.vtt)")
    parser.add_argument("--translate", action="store_true", help="Translate instead of transcribe")
    args = parser.parse_args()

    # Prepare file paths
    input_path = Path(args.audio)
    audio_name = input_path.stem

    # Convert to MP3 if necessary
    print("Converting input file to MP3 if needed...")
    mp3_file = convert_to_mp3(str(input_path), audio_name)

    # Initialize Whisper JAX pipeline
    print("Initializing Whisper JAX pipeline...")
    pipeline = setup_pipeline()

    # First run: JIT compile (slow, but only once)
    print("Running initial compilation and transcription...")
    outputs = transcribe_audio(pipeline, mp3_file, translate=args.translate)
    print("\nInitial transcription with timestamps:")
    print(outputs["text"])

    # Save subtitles as VTT
    save_as_vtt(outputs, args.output)
    print(f"VTT file saved at: {args.output}")


if __name__ == "__main__":
    main()

"""
Usage Examples:
---------------
Transcribe:
python whisper_transcribe.py --audio myfile.mp4 --output subtitles.vtt

Translate to English:
python whisper_transcribe.py --audio myfile.mp4 --output subtitles.vtt --translate
"""
