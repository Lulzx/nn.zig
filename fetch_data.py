#!/usr/bin/env python3
"""
DailyDialog Dataset Downloader
Downloads training and validation data for nn.zig
"""

import os
import zipfile

BASE_URL = "https://huggingface.co/datasets/roskoN/dailydialog/resolve/main"

FILES = [
    ("train.zip", "1.94 MB"),
    ("validation.zip", "180 KB"),
    ("test.zip", "179 KB"),
]


def download_file(url, filename, size_hint):
    """Download a file with progress display."""
    try:
        import requests
        return download_with_requests(url, filename, size_hint)
    except ImportError:
        print("Note: Install 'requests' for better progress: pip install requests")
        return download_with_urllib(url, filename, size_hint)


def download_with_requests(url, filename, size_hint):
    import requests

    print(f"Downloading {filename} (~{size_hint})...")
    print(f"URL: {url}")

    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get('content-length', 0))
    downloaded = 0

    with open(filename, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            downloaded += len(chunk)
            if total_size > 0:
                pct = (downloaded / total_size) * 100
                bar = "=" * int(pct // 2) + ">" + " " * (50 - int(pct // 2))
                print(f"\r[{bar}] {pct:.1f}%", end="", flush=True)

    print(f"\nDownloaded: {filename}")
    return True


def download_with_urllib(url, filename, size_hint):
    from urllib.request import urlopen

    print(f"Downloading {filename} (~{size_hint})...")
    print(f"URL: {url}")

    with urlopen(url) as response:
        total_size = int(response.headers.get('Content-Length', 0))
        downloaded = 0

        with open(filename, "wb") as f:
            while True:
                chunk = response.read(8192)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    pct = (downloaded / total_size) * 100
                    print(f"\rDownloaded: {downloaded / 1_000_000:.1f}MB ({pct:.1f}%)", end="", flush=True)

    print(f"\nDownloaded: {filename}")
    return True


def extract_zip(zip_path, extract_dir):
    """Extract zip archive."""
    print(f"Extracting {zip_path}...")

    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(extract_dir)

    print(f"Extracted to: {extract_dir}/")
    return True


def process_dialogs_to_txt(data_dir="data", output_file="dailydialog.txt"):
    """Process dialog files into a single text file for training."""
    print(f"\nProcessing dialogs to {output_file}...")

    total_dialogs = 0

    with open(output_file, "w", encoding="utf-8") as out:
        # Process train, validation, and test splits
        for split in ["train", "validation", "test"]:
            # Try various path patterns (handles nested extraction)
            possible_paths = [
                os.path.join(data_dir, split, split, f"dialogues_{split}.txt"),
                os.path.join(data_dir, split, f"dialogues_{split}.txt"),
                os.path.join(data_dir, split, "dialogues_text.txt"),
            ]

            dialog_file = None
            for path in possible_paths:
                if os.path.exists(path):
                    dialog_file = path
                    break

            if dialog_file is None:
                print(f"Warning: Could not find dialog file for {split}")
                continue

            print(f"Processing {split} split from {dialog_file}...")

            with open(dialog_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    # DailyDialog format: utterances separated by __eou__
                    utterances = line.split("__eou__")
                    utterances = [u.strip() for u in utterances if u.strip()]

                    if utterances:
                        # Write dialog with speaker turns
                        for i, utterance in enumerate(utterances):
                            speaker = "A" if i % 2 == 0 else "B"
                            out.write(f"{speaker}: {utterance}\n")
                        out.write("\n")  # Blank line between dialogs
                        total_dialogs += 1

    size = os.path.getsize(output_file) if os.path.exists(output_file) else 0
    print(f"Processed {total_dialogs} dialogs to {output_file} ({size / 1_000_000:.2f}MB)")
    return total_dialogs > 0


def main():
    os.makedirs("data", exist_ok=True)

    # Download all zip files
    for filename, size_hint in FILES:
        filepath = os.path.join("data", filename)
        if os.path.exists(filepath):
            size = os.path.getsize(filepath)
            print(f"{filename} already exists ({size / 1_000:.1f}KB)")
        else:
            url = f"{BASE_URL}/{filename}"
            download_file(url, filepath, size_hint)

    # Extract zip files
    for filename, _ in FILES:
        filepath = os.path.join("data", filename)
        split_name = filename.replace(".zip", "")
        extract_dir = os.path.join("data", split_name)

        if os.path.exists(extract_dir):
            print(f"{extract_dir}/ already exists, skipping extraction.")
        elif os.path.exists(filepath):
            extract_zip(filepath, extract_dir)

    # Process dialogs to text file
    if not os.path.exists("dailydialog.txt"):
        process_dialogs_to_txt()
    else:
        size = os.path.getsize("dailydialog.txt")
        print(f"\ndailydialog.txt already exists ({size / 1_000_000:.2f}MB)")

    # Summary
    print("\n" + "=" * 50)
    print("Download complete!")
    print("=" * 50)

    if os.path.exists("dailydialog.txt"):
        size = os.path.getsize("dailydialog.txt")
        lines = sum(1 for _ in open("dailydialog.txt", encoding="utf-8"))
        print(f"  dailydialog.txt: {size / 1_000_000:.2f}MB ({lines} lines)")

    print("\nReady to train! Run:")
    print("  zig build run -Doptimize=ReleaseFast")


if __name__ == "__main__":
    main()
