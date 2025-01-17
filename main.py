import os
import time
from dotenv import load_dotenv
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from application import Application

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


if __name__ == "__main__":
    load_dotenv()
    folder_to_watch = os.getenv('PDF_FOLDER_PATH')
    app = Application()
    event_handler = PDFHandler(app)
    observer = Observer()
    observer.schedule(event_handler, folder_to_watch, recursive=False)
    observer.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()