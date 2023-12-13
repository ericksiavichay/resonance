import os
from utils.youtube import download_video_as_wav

"""Data URls"""
ESC50_URL = "https://github.com/karoldvl/ESC-50/archive/master.zip"

if __name__ == "__main__":
    print("\nInstalling requirements...")
    os.system("pip install -r requirements.txt")
    print("Requirements installed.\n")

    if not os.path.exists("./ESC-50-master"):
        response = input(
            "Path to ESC-50 not found. Would you like to download ESC-50? (y/[n])"
        )
        if response == "y":
            print(f"Downloading ESC-50 from: {ESC50_URL}")
            os.system(f"wget {ESC50_URL}")
            os.system("python -m zipfile -e master.zip ESC-50-master")
            os.system("rm master.zip")
            print("Downloaded ESC-50.")

    if not os.path.exists("./MusicCaps") or len(os.listdir("./MusicCaps")) == 0:
        response = input(
            "Path to MusicCaps not found. Would you like to download MusicCaps? (y/[n])"
        )
        if response == "y":
            print("Downloading MusicCaps...")
            if not os.path.exists("MusicCaps"):
                os.mkdir("MusicCaps")
            import pandas as pd

            music_csv = pd.read_csv("utils/musiccaps-public.csv")
            music_csv.apply(
                lambda row: download_video_as_wav(
                    row["ytid"], row.name, row["start_s"], row["end_s"], "MusicCaps"
                ),
                axis=1,
            )
            print("Downloaded MusicCaps.")
