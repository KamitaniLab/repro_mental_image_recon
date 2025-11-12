#!/usr/bin/env python3
"""
download_brain_features.py

A script to download and extract brain decoding feature data (.tar.gz) from Google Drive.
Based on https://colab.research.google.com/drive/1gaMoae0ntiT94-rQUMymkZboNc-imTzl?usp=drive_link
"""

import gdown
import tarfile
from pathlib import Path

def main():
    # Google Drive のファイルID
    file_id = "1Q7TVsVbASMqnDYfFjFzo2SV6njExu8qq"
    download_url = f"https://drive.google.com/uc?id={file_id}"

    # 保存パス
    output_dir = Path("./mental_img_recon/content/mental_img_recon")
    output_dir.mkdir(parents=True, exist_ok=True)

    download_path = output_dir / "downloaded_file.tar.gz"

    print(f"Downloading from: {download_url}")
    print(f"Saving to: {download_path}")

    # Download the file
    gdown.download(download_url, str(download_path), quiet=False)

    # Extract the files
    print("Extracting files...")
    with tarfile.open(download_path, "r:gz") as tar:
        tar.extractall(path=output_dir)

    print(f"✅ Done! Files extracted to: {output_dir}")

if __name__ == "__main__":
    main()