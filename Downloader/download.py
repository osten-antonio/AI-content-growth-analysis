import csv
from datasets import load_dataset
import datasets
import argparse
from tqdm import tqdm
import os
from collections import defaultdict

datasets.config.STREAMING_READ_MAX_RETRIES = 50
datasets.config.STREAMING_READ_RETRY_INTERVAL = 5

parser = argparse.ArgumentParser("download")
parser.add_argument("link", type=str)
parser.add_argument("out", type=str)
args = parser.parse_args()

year_counter = defaultdict(int)
month_counter = defaultdict(int)

MONTH_LIMIT = 5000 

dataset = load_dataset(args.link, split="train", streaming=True)

def extract_year_month(example):
    try:
        s = str(example["date"])
        return int(s[:4]), int(s[5:7])
    except:
        return None, None


first_write = not os.path.exists(args.out)

with open(args.out, "a", newline="", encoding="utf-8") as f:
    writer = None

    for row in tqdm(dataset, unit="rows"):

        year, month = extract_year_month(row)
        if year is None:
            continue

        if month is None:
            continue

        if not (2020 <= year <= 2025):
            continue

        if month_counter[(year, month)] >= MONTH_LIMIT:
            continue

        if writer is None:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if first_write:
                writer.writeheader()


        writer.writerow(row)

        year_counter[year] += 1
        month_counter[(year, month)] += 1
