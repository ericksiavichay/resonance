import os

"""Data URls"""
ESC50_URL = "https://github.com/karoldvl/ESC-50/archive/master.zip"

if __name__ == "__main__":
    if not os.path.exists("./ESC-50-master"):
        response = input(
            "Path to data not found. Would you like to download ESC-50? (y/[n])"
        )
        if response == "y":
            print(f"Downloading ESC-50 from: {ESC50_URL}")
            os.system(f"wget {ESC50_URL}")
            os.system("python -m zipfile -e master.zip ESC-50-master")
            print("Download complete.")

    print("\nInstalling requirements...")
    os.system("pip install -r requirements.txt")
    print("Requirements installed.\n")
