import threading
from tkinter import END, LEFT, X, Button, Entry, Frame, Label, Text, filedialog
from pubsub import pub
from app.src.app import Application

class GUI:
    def __init__(self, master):
        self.master = master
        self.thread = None
        self.stop_event = threading.Event()

        pub.subscribe(self.log_message, 'log_event')
        pub.subscribe(self.print_message, 'print_event')


        self.label_folder = Label(master, text="Folder to Watch:")
        self.label_folder.pack()

        self.frame_folder = Frame(master)
        self.frame_folder.pack(fill=X)

        self.entry_folder = Entry(self.frame_folder, width=50)
        self.entry_folder.pack(side=LEFT, fill=X, expand=True)

        self.button_browse_folder = Button(self.frame_folder, text="Browse", command=self.browse_folder)
        self.button_browse_folder.pack(side=LEFT)

        self.label_output = Label(master, text="Output Path:")
        self.label_output.pack()

        self.frame_output = Frame(master)
        self.frame_output.pack(fill=X)

        self.entry_output = Entry(self.frame_output, width=50)
        self.entry_output.pack(side=LEFT, fill=X, expand=True)

        self.button_browse_output = Button(self.frame_output, text="Browse", command=self.browse_output)
        self.button_browse_output.pack(side=LEFT)

        self.start_button = Button(master, text="Start", command=self.start_process)
        self.start_button.pack()

        self.log_text = Text(master, height=15, width=60)
        self.log_text.pack()

    def browse_folder(self):
        folder_selected = filedialog.askdirectory()
        if folder_selected:
            self.entry_folder.delete(0, END)
            self.entry_folder.insert(0, folder_selected)

    def browse_output(self):
        output_selected = filedialog.askdirectory()
        if output_selected:
            self.entry_output.delete(0, END)
            self.entry_output.insert(0, output_selected)

    def start_process(self):
        folder_to_watch = self.entry_folder.get()
        output_path = self.entry_output.get()
        self.log_text.insert(END, f"Watching folder: {folder_to_watch}\n")
        self.log_text.insert(END, f"Output path: {output_path}\n")
        self.log_text.see(END)

        if self.thread is None or not self.thread.is_alive():
            self.thread = threading.Thread(target=self.trigger_start_watching, args=(folder_to_watch, output_path))
            self.thread.start()
        
        self.start_button.config(text="Stop", command=self.stop_process)

    def trigger_start_watching(self, folder_to_watch, output_path):
        self.stop_event.clear()
        app = Application()
        app.start_watching(folder_to_watch, output_path, self.stop_event)

    def print_message(self, message):
        print(message)

    def stop_process(self):
        self.stop_event.set()
        self.log_text.insert(END, "Stopping the process...\n")
        self.log_text.see(END)
        self.start_button.config(text="Start", command=self.start_process)

    def log_message(self, message):
        print(message)
        print("_________________________________")
        self.log_text.insert(END, message + "\n")
        self.log_text.see(END)