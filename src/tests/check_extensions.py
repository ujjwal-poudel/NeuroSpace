"""
This script scans the dataset directory and prints every file extension found,
along with the number of occurrences. This is useful for verifying what image
formats are present before writing your dataset loader.
"""

import os
from collections import Counter

DATASET_DIR = "/Users/ujjwalpoudel/Documents/insane_projects/NeuroSpace/data/living_room"   # Change this to your dataset path

def get_extensions(directory):
    exts = Counter()
    for root, dirs, files in os.walk(directory):
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            exts[ext] += 1
    return exts

if __name__ == "__main__":
    extensions = get_extensions(DATASET_DIR)
    print("Found file extensions:")
    for ext, count in extensions.items():
        print(f"{ext}: {count}")
