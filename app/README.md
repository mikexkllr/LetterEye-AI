# My Tkinter App

This project is a simple Tkinter application that allows users to set a folder to watch for changes and specify an output path. It features a user-friendly interface with logging capabilities to monitor the application's activity.

## Project Structure

```
my-tkinter-app
├── src
│   ├── main.py        # Entry point of the application
│   ├── gui.py         # Contains the GUI class
│   ├── app.py         # Main application logic
│   └── utils.py       # Utility functions for file operations
├── requirements.txt    # Lists project dependencies
└── README.md           # Project documentation
```

## Requirements

To run this application, you need to install the following dependencies:

- Tkinter (usually included with Python)
- watchdog (for file watching)

You can install the required packages using pip:

```
pip install -r requirements.txt
```

## Usage

1. Clone the repository or download the project files.
2. Navigate to the project directory.
3. Install the required dependencies.
4. Run the application:

```
python src/main.py
```

5. Set the folder to watch and the output path in the GUI.
6. Click the "Start" button to begin monitoring the specified folder.
7. Logs will be displayed in the text field for your reference.

## Contributing

Feel free to submit issues or pull requests if you have suggestions or improvements for the project.