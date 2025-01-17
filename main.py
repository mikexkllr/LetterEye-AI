import os
from application import Application


if __name__ == "__main__":
    app = Application(os.getenv('PDF_PATH'))
    app.run()