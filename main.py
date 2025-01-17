import os
from application import Application


if __name__ == "__main__":
    app = Application()
    app.run(os.getenv('PDF_FOLDER_PATH'))