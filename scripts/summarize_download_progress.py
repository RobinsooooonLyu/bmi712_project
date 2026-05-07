#!/usr/bin/env python3
import argparse
import csv
import os
from collections import Counter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--progress-log")
    args = parser.parse_args()

    manifest_rows = []
    with open(args.manifest, encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        manifest_rows.extend(reader)

    complete = 0
    partial = 0
    missing = 0
    for row in manifest_rows:
        target = os.path.join(args.output_dir, row["filename"])
        expected_size = row.get("size", "")
        if not os.path.exists(target):
            missing += 1
            continue
        actual_size = str(os.path.getsize(target))
        if expected_size and actual_size == expected_size:
            complete += 1
        else:
            partial += 1

    print(f"Manifest files: {len(manifest_rows)}")
    print(f"Complete:       {complete}")
    print(f"Partial:        {partial}")
    print(f"Missing:        {missing}")

    if args.progress_log and os.path.exists(args.progress_log):
        counter = Counter()
        with open(args.progress_log, encoding="utf-8") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for row in reader:
                counter[row["status"]] += 1
        print("Progress log status counts:")
        for key, value in counter.most_common():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
