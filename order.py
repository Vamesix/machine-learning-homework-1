import os
import csv
import shutil

labels = []
with open("data/training_labels.csv") as file:
    csv_file = csv.reader(file)
    next(csv_file)  # skip header
    for row in csv_file:
        index = int(row[0])
        label = row[1].replace("/", "_")  # escape slash, needed for directory

        # Make a new directory if a new label is encountered
        if label not in labels:
            labels.append(label)
            os.mkdir(f"data/training_data/{label}")

        # Move file
        source_file = f"data/training_data/{index:>06}.jpg"
        target_path = f"data/training_data/{label}"
        shutil.move(source_file, target_path)