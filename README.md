
# PDF Text Extraction and Analysis Script

This project provides a Python script that extracts text from scanned PDF files using Optical Character Recognition (OCR) and analyzes the extracted text using the OpenAI API. The script is specifically designed for processing letters and aims to extract key information such as the sender, recipient, date, type of letter, and a short summary.


## Features

- Converts PDF files to text using Tesseract OCR.
- Analyzes text to identify key information using OpenAI's GPT model.
- Provides a concise summary of the contents of a bill.

## Prerequisites

- Python 3.7 or later
- Tesseract OCR
- OpenAI API Access

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-repo/pdf-text-extraction.git
   cd pdf-text-extraction
   ```

2. **Set Up a Virtual Environment**:
   Create and activate a virtual environment to manage dependencies:
   
   - **On Windows**:
     ```bash
     python -m venv venv
     .\venv\Scripts\activate
     ```

   - **On macOS and Linux**:
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```

3. **Install Python Dependencies**:
   Once the virtual environment is activated, use pip to install the necessary packages:
   ```bash
   pip install python-dotenv pytesseract pdf2image openai
   ```

4. **Install Tesseract OCR**:
   Follow the instructions on [Tesseract's GitHub page](https://github.com/tesseract-ocr/tesseract) to install Tesseract on your system.

5. **Prepare the Environment File**:
   Create a `.env` file in the root directory of the project with the following content:
   ```
   PDF_PATH=path/to/your/file.pdf
   OPENAI_API_KEY=your_openai_api_key
   ```
   - Replace `path/to/your/file.pdf` with the path to your input PDF file.
   - Replace `your_openai_api_key` with your actual OpenAI API key.

6. **Install Poppler (if required by pdf2image)**:
   - **Windows**: Download from [Poppler for Windows](http://blog.alivate.com.au/poppler-windows/), and make sure it is added to your PATH.
   - **macOS**: Install via Homebrew `brew install poppler`.
   - **Linux**: Install via your package manager, e.g., `sudo apt-get install poppler-utils`.

## Usage

1. **Run the Script**:
   Ensure your virtual environment is activated, then execute the script from your command line:
   ```bash
   python main.py
   ```

2. **Output**:
   - The script will convert the PDF to text, analyze it, and output the extracted details such as who issued the bill, the recipient, the date, and a short summary.

## Error Handling

- The script includes basic error handling to catch and display errors during file processing, API calls, or issues with the OCR.
- Ensure that Tesseract and Poppler are correctly configured and accessible to avoid OCR errors.

## Notes

- The script assumes that the PDF contains clear and linear text typical of bills and that necessary OCR configurations are set.
- The OpenAI API usage may incur costs; ensure that you monitor and manage your API usage within your account's limits.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome. Please fork the repository and submit a pull request for any improvements or additional features.

---

This updated README includes instructions for setting up a Python virtual environment, ensuring a clean and isolated environment for your project dependencies. Adjust any placeholder information, such as repository URLs, as needed.