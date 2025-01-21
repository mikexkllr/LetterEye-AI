import threading
from tkinter import END, Button, Entry, Label, Text

from app.src.app import Application

class GUI:
    def __init__(self, master):
        self.master = master
        self.app: Application = Application(self)
        master.title("File Watcher")

        self.label_folder = Label(master, text="Folder to Watch:")
        self.label_folder.pack()

        self.entry_folder = Entry(master, width=50)
        self.entry_folder.pack()

        self.label_output = Label(master, text="Output Path:")
        self.label_output.pack()

        self.entry_output = Entry(master, width=50)
        self.entry_output.pack()

        self.start_button = Button(master, text="Start", command=self.start_process)
        self.start_button.pack()

        self.log_text = Text(master, height=15, width=60)
        self.log_text.pack()

        self.stop_event = threading.Event()

    def start_process(self):
        folder_to_watch = self.entry_folder.get()
        output_path = self.entry_output.get()
        self.log_text.insert(END, f"Watching folder: {folder_to_watch}\n")
        self.log_text.insert(END, f"Output path: {output_path}\n")

        self.stop_event.clear()
        self.start_button.config(text="Stop", command=self.stop_process)
        thread = threading.Thread(target=self.trigger_start_watching, args=(folder_to_watch, output_path))
        thread.start()

    def trigger_start_watching(self, folder_to_watch, output_path):
        self.app.start_watching(folder_to_watch, output_path)

    def stop_process(self):
        self.log_text.insert(END, "Stopping the process...\n")
        self.stop_event.set()
        self.log_text.see(END)
        self.start_button.config(text="Start", command=self.start_process)

    def log_message(self, message):
        self.log_text.insert(END, message)
        self.log_text.see(END)