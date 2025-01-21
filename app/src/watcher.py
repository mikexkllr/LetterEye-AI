import os
import time
from tkinter import Text
from dotenv import load_dotenv
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from app.src.core import CoreApplication

class PDFHandler(FileSystemEventHandler):
    def __init__(self, app):
        self.app = app

    def on_created(self, event):
        if event.is_directory:
            return
        if event.src_path.endswith('.pdf'):
            print(f"New PDF file detected: {event.src_path}")
            if self._wait_for_file(event.src_path):
                self.app.run(event.src_path)

    def _wait_for_file(self, file_path, wait_time=1, max_attempts=10):
        attempts = 0
        while attempts < max_attempts:
            initial_size = os.path.getsize(file_path)
            time.sleep(wait_time)
            new_size = os.path.getsize(file_path)
            if initial_size == new_size:
                return True
            attempts += 1
        print(f"Failed to fully copy the file: {file_path}")
        return False
    
class Watcher:
    def __init__(self, openai_api_key, language, csv_dir):
        self.openai_api_key = openai_api_key
        self.language = language
        self.csv_dir = csv_dir
        self.observer = Observer()

    def start(self, output_dir: str, folder_to_watch: str, log: Text):
        print(f"Watching folder: {folder_to_watch}")
        self.app = CoreApplication(self.openai_api_key, self.language, self.csv_dir, output_dir)
        self.event_handler = PDFHandler(self.app)
        self.observer.schedule(self.event_handler, folder_to_watch, recursive=False)
        self.observer.start()
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            self.observer.stop()
        self.observer.join()

    def stop(self):
        self.observer.stop()
        self.observer.join()
    

