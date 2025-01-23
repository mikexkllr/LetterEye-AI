from multiprocessing import process
import os
from fuzzywuzzy import process, fuzz

class FolderManager:
    def __init__(self, output_dir):
        self.output_dir = output_dir

    def find_or_create_folder(self, worker_name, receiver_name):
        main_folder = os.path.join(self.output_dir, worker_name)
        os.makedirs(main_folder, exist_ok=True)

        subdirectories = os.listdir(main_folder)
        best_match = process.extractOne(receiver_name, subdirectories, scorer=fuzz.ratio)

        if best_match:
            existing_folder, score = best_match
            if score > 55:
                return os.path.join(main_folder, existing_folder)

        receiver_folder = os.path.join(main_folder, receiver_name)
        os.makedirs(receiver_folder, exist_ok=True)
        return receiver_folder
