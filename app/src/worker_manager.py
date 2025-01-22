import csv
import os
from fuzzywuzzy import process, fuzz


class WorkerManager:
    def __init__(self, csv_dir):
        self.csv_dir = csv_dir

    def find_worker_by_receiver(self, receiver_name):
        best_match = None
        highest_score = 0

        for filename in os.listdir(self.csv_dir):
            if filename.endswith('.csv'):
                with open(os.path.join(self.csv_dir, filename), newline='') as csvfile:
                    csv_reader = csv.reader(csvfile)
                    for row in csv_reader:
                        if row:
                            match, score = process.extractOne(receiver_name, row, scorer=fuzz.ratio)
                            if score > highest_score and score > 70:
                                best_match = (filename.replace(".csv", ""), filename, match)
                                highest_score = score

        if best_match:
            return best_match[0], best_match[1], best_match[2]

        return None, None, None