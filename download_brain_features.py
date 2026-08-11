#!/usr/bin/env python3
"""Download the decoded brain features of Koide-Majima et al. (2024) from Google Drive.

The archive holds the decoded VGG19/CLIP features and the mean DNN features that the
reconstruction reads, and extracts to the layout config_recon.yaml points at:

    lib/mental_img_recon/content/mental_img_recon/features/
        decoded_features/{S01,S02,S03}/{VGG19,CLIP_ViT-B_32}/<layer>/
        meanDNNfeature/{VGG19,CLIP_ViT-B_32,...}/

Source: https://colab.research.google.com/drive/1gaMoae0ntiT94-rQUMymkZboNc-imTzl

The download is 1.7 GB and extracts to about 3.6 GB. It is checked against the
sha256 of the copy these analyses ran on, and an archive that already verifies is
not fetched again.
"""

import hashlib
import tarfile
from pathlib import Path

import gdown

# Google Drive file id, from the Colab notebook published with the original paper.
FILE_ID = "1Q7TVsVbASMqnDYfFjFzo2SV6njExu8qq"
SHA256 = "a12b7e124302e417de88c8160a77cf29d8cd7acc72493e1843f3b35ccc8b8b6f"
SIZE_BYTES = 1_784_162_805

OUTPUT_DIR = Path("./lib/mental_img_recon/content/mental_img_recon")
ARCHIVE_NAME = "downloaded_file.tar.gz"


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    archive = output_dir / ARCHIVE_NAME

    if archive.exists() and sha256_of(archive) == SHA256:
        print(f"Archive already present and verified: {archive}")
    else:
        url = f"https://drive.google.com/uc?id={FILE_ID}"
        print(f"Downloading from: {url}")
        print(f"Saving to: {archive} ({SIZE_BYTES / 1e9:.1f} GB)")
        gdown.download(url, str(archive), quiet=False)

        digest = sha256_of(archive)
        if digest != SHA256:
            raise SystemExit(
                f"sha256 mismatch for {archive}\n"
                f"  expected {SHA256}\n"
                f"  got      {digest}\n"
                "Google Drive serves an HTML quota page instead of the file when the "
                "daily download limit is hit; delete the file and retry later."
            )
        print("Checksum OK.")

    print("Extracting files...")
    with tarfile.open(archive, "r:gz") as tar:
        # filter='data' refuses absolute paths and links pointing outside the
        # destination. It is the default from Python 3.14; set it explicitly so the
        # behaviour does not depend on the interpreter version.
        tar.extractall(path=output_dir, filter="data")

    print(f"Done. Files extracted to: {output_dir}")


if __name__ == "__main__":
    main()
