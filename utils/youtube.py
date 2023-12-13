"""Helper functions for downloading Youtube videos as WAV files."""

from pydub import AudioSegment
from pytube import YouTube
import os


def cut_audio(input_path, output_path, start_sec, end_sec):
    """
    Clips audo from start_sec to end_sec and saves it to output_path.

    Args:
        input_path (str): Path to input audio file.
        output_path (str): Path to save the output audio file.
        start_sec (float): Start time in seconds.
        end_sec (float): End time in seconds.
    """
    audio = AudioSegment.from_file(input_path)
    clip = audio[start_sec * 1000 : end_sec * 1000]
    clip.export(output_path, format="wav")


def download_video_as_wav(yt_id, index, start_sec, end_sec, output_dir, verbose=True):
    """
    Download a YouTube video as a WAV file.

    Args:
        id (str): YouTube video ID.
        start_sec (float): Start time in seconds.
        end_sec (float): End time in seconds.
        output_dir (str): Directory to save the WAV file.
    """
    if verbose:
        print("Downloading audio: ", yt_id)
        print("Index: ", index)
        print("\n")
    yt = YouTube(f"https://www.youtube.com/watch?v={yt_id}", use_oauth=True)
    yt.streams.get_audio_only(subtype="mp4").download(
        output_path=output_dir,
        filename=f"{index}.mp4",
    )

    cut_audio(
        f"{output_dir}/{index}.mp4", f"{output_dir}/{index}.wav", start_sec, end_sec
    )

    os.system(f"rm {output_dir}/{index}.mp4")
