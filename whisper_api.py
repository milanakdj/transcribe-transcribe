import os
import argparse
import tempfile
import shutil
from math import ceil
from pathlib import Path
from openai import OpenAI
from pydub import AudioSegment


def convert_to_mp3(file_path: str, output_name: str) -> str:
    """
    Convert input file to MP3 if not already in MP3 format.
    """
    if not file_path.lower().endswith(".mp3"):
        output_path = f"{output_name}.mp3"
        AudioSegment.from_file(file_path).export(output_path, format="mp3")
        return output_path
    return file_path


def slice_audio(file_path: str, output_dir: str, slice_minutes: int = 10) -> list[str]:
    """
    Slice audio into chunks of given minutes and export as MP3.
    """
    os.makedirs(output_dir, exist_ok=True)

    audio = AudioSegment.from_mp3(file_path)
    slice_ms = slice_minutes * 60 * 1000
    audio_length = len(audio)

    slice_paths = []
    start_time = 0

    for i in range(ceil(audio_length / slice_ms)):
        slice_file = Path(output_dir) / f"{Path(file_path).stem}_{i}.mp3"
        audio[start_time:start_time + slice_ms].export(slice_file, format="mp3")
        slice_paths.append(str(slice_file))
        start_time += slice_ms

    return slice_paths


def transcribe_slices(client, slice_paths: list[str], response_format: str = "vtt") -> str:
    """
    Transcribe audio slices with Whisper API.
    """
    translations = []
    for path in slice_paths:
        print(f"Transcribing slice: {path}")
        with open(path, "rb") as audio_file:
            transcription = client.audio.translations.create(
                model="whisper-1", file=audio_file, response_format=response_format
            )
            translations.append(str(transcription))
    return "\n".join(translations)


def save_transcriptions(base_name: str, text: str, output_dir: str = "./export_slice/text_slice"):
    """
    Save transcription in .txt, .srt, and .vtt formats.
    """
    os.makedirs(output_dir, exist_ok=True)
    for ext in ("txt", "srt", "vtt"):
        out_file = Path(output_dir) / f"complete_timestamp_{base_name}.{ext}"
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(text)


def parse_arguments():
    """
    Parse command-line arguments for dynamic configuration.
    """
    parser = argparse.ArgumentParser(description="Audio transcription using OpenAI Whisper.")
    parser.add_argument("file", help="Path to the audio file to process.")
    parser.add_argument("--name", help="Base name for output files. Defaults to input filename.", default=None)
    parser.add_argument("--slice_minutes", type=int, default=10, help="Duration of each audio slice in minutes. Default is 10.")
    parser.add_argument("--text_dir", default="./export_slice/text_slice", help="Directory to store transcription output.")
    parser.add_argument("--format", choices=["vtt", "srt", "text"], default="vtt", help="Output transcription format.")
    return parser.parse_args()


def main():
    args = parse_arguments()
    client = OpenAI()

    # Extract the base name if not provided
    audio_name = args.name if args.name else Path(args.file).stem

    # Create a temporary directory for audio slices
    temp_dir = tempfile.mkdtemp(prefix="audio_slices_")
    print(f"Temporary directory created: {temp_dir}")

    try:
        # Step 1: Convert to MP3 if needed
        mp3_file = convert_to_mp3(args.file, audio_name)

        # Step 2: Slice audio into temporary directory
        slice_paths = slice_audio(mp3_file, temp_dir, args.slice_minutes)
        print(f"Created {len(slice_paths)} audio slices.")

        # Step 3: Transcribe slices
        complete_translation = transcribe_slices(client, slice_paths, response_format=args.format)

        # Step 4: Save results
        save_transcriptions(audio_name, complete_translation, args.text_dir)
        print("Transcription complete and saved to ", args.text_dir)

    finally:
        # Cleanup temporary directory after processing
        shutil.rmtree(temp_dir)
        print(f"Temporary directory deleted: {temp_dir}")


if __name__ == "__main__":
    main()

    #usage:vtt
"""
Usage Examples:
---------------
Transcribe:
python whisper_api.py "/path/to/your/audiofile.wav" \
  --name "meeting_notes" \
  --slice_minutes 15 \
  --text_dir "./output/text_slices" \
  --format 

"""
