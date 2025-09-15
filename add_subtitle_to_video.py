import subprocess
import argparse
from pathlib import Path


def add_subtitles(video_file: str, subtitle_file: str, output_file: str) -> None:
    """
    Add subtitles to a video using FFmpeg with custom styling.

    Args:
        video_file (str): Path to the input video file.
        subtitle_file (str): Path to the subtitle (.srt) file.
        output_file (str): Path to save the output video.
    """
    vf_arg = (
        f"subtitles={subtitle_file}:"
        "force_style='OutlineColour=&H100000000,BorderStyle=3,Outline=1,Shadow=0,Fontsize=12'"
    )

    command = [
        "ffmpeg",
        "-i", video_file,
        "-vf", vf_arg,
        "-c:a", "copy",  # copy audio without re-encoding
        output_file
    ]

    subprocess.run(command, check=True)


def main():
    parser = argparse.ArgumentParser(description="Embed subtitles into a video using FFmpeg.")
    parser.add_argument("video", help="Path to the input video file (e.g., input.mp4)")
    parser.add_argument("subtitle", help="Path to the subtitle file (.srt)")
    parser.add_argument("output", help="Path to save the output video")

    args = parser.parse_args()

    video_file = Path(args.video)
    subtitle_file = Path(args.subtitle)
    output_file = Path(args.output)

    if not video_file.exists():
        raise FileNotFoundError(f"Video file not found: {video_file}")
    if not subtitle_file.exists():
        raise FileNotFoundError(f"Subtitle file not found: {subtitle_file}")

    add_subtitles(str(video_file), str(subtitle_file), str(output_file))


if __name__ == "__main__":
    main()
    #usage: python add_subtitles.py "input.mp4" "subtitles.srt" "output.mp4"

