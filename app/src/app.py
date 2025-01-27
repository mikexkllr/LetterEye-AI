import os
from threading import Event
from tkinter import END, Text
from dotenv import load_dotenv
from app.src.watcher import Watcher
from pubsub import pub

from config import Config


class Application:

    def __init__(self):
        self.openai_api_key = Config.settings.openai_api_key
        self.language = Config.settings.language
        self.csv_dir = Config.settings.csv_files
        self.output_dir = Config.settings.output_dir
        self.folder_to_watch = Config.settings.pdf_folder_path

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