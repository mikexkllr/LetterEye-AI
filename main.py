from app.src.gui import GUI
import tkinter as tk

if __name__ == "__main__":
    root = tk.Tk()
    root.title("File Watcher")
    gui = GUI(root)
    root.mainloop()