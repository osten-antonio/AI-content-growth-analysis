import csv
from datasets import load_dataset
import os
from tqdm import tqdm
from collections import defaultdict


year_counter = defaultdict(int)
LIMIT = 50000

dataset = load_dataset('Rijgersberg/YouTube-Commons', split="train", streaming=True, columns=['text','date','transcription_language'])

def extract_year(example):
    try:
        return int(str(example['date'])[:4])
    except:
        return None

first_write = not os.path.exists('youtube.csv')
with open('youtube.csv', "a", newline="", encoding="utf-8") as f:
    writer = None

    for row in tqdm(dataset, unit="rows"):

        year = extract_year(row)
        if year is None:
            continue

        if not (2020 <= year <= 2025):
            continue

        if year_counter[year] >= LIMIT:
            continue

        if row['transcription_language'] != 'en':
            continue

        if writer is None:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if first_write:
                writer.writeheader()

        writer.writerow(row)
        year_counter[year] += 1

        if all(year_counter[y] >= LIMIT for y in range(2020, 2026)):
            break
