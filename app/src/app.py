import os
from threading import Event
from tkinter import END, Text
from dotenv import load_dotenv
from app.src.watcher import Watcher
from pubsub import pub


class Application:
    def __init__(self):
        load_dotenv()
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.language = os.getenv('LANGUAGE')
        self.csv_dir = os.getenv('CSV_FILES')
        self.output_dir = os.getenv('OUTPUT_DIR')
        self.folder_to_watch = os.getenv('PDF_FOLDER_PATH')

        self.watcher = Watcher(self.openai_api_key, self.language, self.csv_dir)

    def start_watching(self, folder_to_watch, output_path, stop_event: Event):
        # gui logic to start watching the folder and process files
        # Implement file watching logic here
        
        self.folder_to_watch = folder_to_watch or self.folder_to_watch
        self.output_dir = output_path or self.output_dir

        if not self.output_dir:
            pub.sendMessage('log_event', message="Please provide an output path.\n")
            return

        if not self.folder_to_watch:
            pub.sendMessage('log_event', message="Please provide a folder to watch.\n")
            return

        self.watcher.start(self.output_dir, self.folder_to_watch, stop_event)