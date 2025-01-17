import os
from dotenv import load_dotenv
import pytesseract
from pdf2image import convert_from_path
from langchain import LLMChain, PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
import csv
from fuzzywuzzy import process, fuzz

# Define a Pydantic model for structured data extraction
class LetterDetails(BaseModel):
    sender: str = Field(..., description="The sender of the letter")
    receiver: str = Field(..., description="The receiver of the letter. NAME ONLY (no other details)")
    organisation: str = Field(..., description="The organisation/ company/name of the sender - empty if not present or private person")
    date_of_writing: str = Field(..., description="The date the letter was written")
    type_of_letter: str = Field(..., description="The type of the letter. Maximal 4 words which describes what the letter is about")
    short_summary: str = Field(..., description="A summary of the letter")

# Manager class to handle worker information in CSV files
class WorkerManager:
    def __init__(self, csv_dir):
        self.csv_dir = csv_dir

    def find_worker_by_receiver(self, receiver_name):
        best_match = None
        highest_score = 0

        for filename in os.listdir(self.csv_dir):
            if filename.endswith('.csv'):
                with open(os.path.join(self.csv_dir, filename), newline='') as csvfile:
                    csv_reader = csv.reader(csvfile)
                    for row in csv_reader:
                        if row:
                            # Use fuzzy matching to find the best candidate
                            match, score = process.extractOne(receiver_name, row, scorer=fuzz.ratio)
                            if score > highest_score and score > 70:  # Threshold can be adjusted
                                best_match = (filename.replace(".csv", ""), filename, match)
                                highest_score = score

        if best_match:
            return best_match[0], best_match[1], best_match[2]

        return None, None, None

# Manager class to handle folder management
class FolderManager:
    def __init__(self, output_dir):
        self.output_dir = output_dir

    def find_or_create_folder(self, worker_name, receiver_name):
        main_folder = os.path.join(self.output_dir, worker_name)
        os.makedirs(main_folder, exist_ok=True)

        # Check for existing folders with similar names
        subdirectories = os.listdir(main_folder)
        best_match = process.extractOne(receiver_name, subdirectories, scorer=fuzz.ratio)

        if best_match:
            existing_folder, score = best_match
            if score > 70:  # Using a fuzziness threshold
                return os.path.join(main_folder, existing_folder)

        # Create new folder since no similar one exists
        receiver_folder = os.path.join(main_folder, receiver_name)
        os.makedirs(receiver_folder, exist_ok=True)
        return receiver_folder

# Processor class for handling PDF and OCR operations
class PDFProcessor:
    def __init__(self, pdf_path, language, api_key):
        self.pdf_path = pdf_path
        self.language = language
        self.llm = ChatOpenAI(api_key=api_key, model="gpt-4o-mini")

    def convert_pdf_to_images(self):
        return convert_from_path(self.pdf_path)

    def perform_ocr(self, images):
        ocr_text = ""
        for image in images:
            ocr_text += pytesseract.image_to_string(image)
        return ocr_text

    # Update the save_pdf method to create a structured filename
    def save_pdf(self, images, receiver_folder, letter_details: LetterDetails, worker_name):
        # Construct the filename using the specified format: date, sender organization, worker name, and letter type
        date_received = letter_details.date_of_writing
        organization = letter_details.organisation if letter_details.organisation else "Private"
        worker = worker_name
        letter_type = letter_details.type_of_letter  # A lol

        # Create a safe filename by removing any unwanted characters
        filename = f"{date_received}_{organization}_{worker}_{letter_type}.pdf"
        filename = filename.replace(' ', '_').replace('/', '-').replace('\\', '-')

        pdf_output_path = os.path.join(receiver_folder, filename)
        images[0].save(pdf_output_path, save_all=True, append_images=images[1:])
        return pdf_output_path

    def analyze_text(self, ocr_text):
        prompt_template = (
            "Extract the sender, receiver with ONLY the name of the person who is receiving this letter, date of writing, type of letter (like a bill, offer, letter of application or anything else you can think of), and provide a summary in this language {language} of the following letter text: {ocr_text}"
        )
        llm_with_structured_output = self.llm.with_structured_output(LetterDetails)
        prompt = PromptTemplate(template=prompt_template, input_variables=["ocr_text", "language"])
        chain = prompt | llm_with_structured_output

        response = chain.invoke({"ocr_text": ocr_text, "language": self.language})
        return response

# Application class to coordinate the entire process
class Application:
    def __init__(self):
        load_dotenv()
        self.pdf_path = os.getenv('PDF_PATH')
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.language = os.getenv('LANGUAGE')
        self.csv_dir = os.getenv('CSV_FILES')  # Ensure this path is set in the .env
        self.output_dir = 'output'

    def run(self):
        pdf_processor = PDFProcessor(self.pdf_path, self.language, self.openai_api_key)
        images = pdf_processor.convert_pdf_to_images()
        ocr_text = pdf_processor.perform_ocr(images)
        print("OCR Text:", ocr_text)

        letter_details = pdf_processor.analyze_text(ocr_text)
        print("Extracted Details:")
        print(letter_details)

        receiver_name = letter_details.receiver
        worker_manager = WorkerManager(self.csv_dir)
        worker_name, csv_filename, matched_receiver = worker_manager.find_worker_by_receiver(receiver_name)

        print(f"Receiver: {receiver_name}")
        print(f"Matched Receiver: {matched_receiver}")
        print(f"Worker Name: {worker_name}")

        if not worker_name:
            print(f"Receiver {receiver_name} not found in any CSV files.")
            return

        folder_manager = FolderManager(self.output_dir)
        receiver_folder = folder_manager.find_or_create_folder(worker_name, matched_receiver)

        pdf_processor.save_pdf(images, receiver_folder, letter_details, worker_name)

# Entry point for the application
if __name__ == "__main__":
    app = Application()
    app.run()