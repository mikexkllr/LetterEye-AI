import os
from dotenv import load_dotenv
import pytesseract
from pdf2image import convert_from_path
from langchain import LLMChain, PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
import csv

# Define a Pydantic model for structured data extraction
class LetterDetails(BaseModel):
    # Define fields for letter details
    sender: str = Field(..., description="The sender of the letter")
    receiver: str = Field(..., description="The receiver of the letter")
    date_of_writing: str = Field(..., description="The date the letter was written")
    type_of_letter: str = Field(..., description="The type of the letter")
    summary: str = Field(..., description="A summary of the letter")

# Manager class to handle worker information in CSV files
class WorkerManager:
    def __init__(self, csv_dir):
        # Directory where CSV files are stored
        self.csv_dir = csv_dir

    def find_worker_by_receiver(self, receiver_name):
        # Iterate over files in the CSV directory
        for filename in os.listdir(self.csv_dir):
            # Check that the file is a CSV
            if filename.endswith('.csv'):
                # Open the CSV file
                with open(os.path.join(self.csv_dir, filename), newline='') as csvfile:
                    csv_reader = csv.reader(csvfile)
                    # Iterate over each row
                    for row in csv_reader:
                        # Check if the receiver name is in any row
                        if row and receiver_name in row:
                            # Return the worker's name and filename
                            return filename.replace(".csv", ""), filename
        # Return None if receiver is not found
        return None, None

# Manager class to handle folder creation
class FolderManager:
    def __init__(self, output_dir):
        # Directory where output folders are created
        self.output_dir = output_dir

    def create_folder_structure(self, worker_name, receiver_name):
        # Create main folder for worker
        main_folder = os.path.join(self.output_dir, worker_name)
        os.makedirs(main_folder, exist_ok=True)
        # Create subfolder for receiver
        receiver_folder = os.path.join(main_folder, receiver_name)
        os.makedirs(receiver_folder, exist_ok=True)
        # Return the path of the receiver's folder
        return receiver_folder

# Processor class for handling PDF and OCR operations
class PDFProcessor:
    def __init__(self, pdf_path, language, api_key):
        # Path to the PDF file
        self.pdf_path = pdf_path
        # Language for text generation
        self.language = language
        # LLM instance with given API key
        self.llm = ChatOpenAI(api_key=api_key, model="gpt-4o-mini")

    def convert_pdf_to_images(self):
        # Convert the PDF into a list of images
        return convert_from_path(self.pdf_path)

    def perform_ocr(self, images):
        # Perform OCR on image list
        ocr_text = ""
        for image in images:
            # Extract text from image using Tesseract
            ocr_text += pytesseract.image_to_string(image)
        # Return the combined text
        return ocr_text

    def save_pdf(self, images, receiver_folder):
        # Save the PDF images into a single combined PDF file
        pdf_output_path = os.path.join(receiver_folder, 'output.pdf')
        images[0].save(pdf_output_path, save_all=True, append_images=images[1:])
        # Return the path where the PDF was saved
        return pdf_output_path

    def analyze_text(self, ocr_text):
        # Template for prompt to the language model
        prompt_template = (
            "Extract the sender, receiver, date of writing, type of letter (like a bill, offer, letter of application or anything else you can think of), and provide a summary in this language {language} of the following letter text: {ocr_text}"
        )
        # Set up the LLM with the structured output format
        llm_with_structured_output = self.llm.with_structured_output(LetterDetails)
        prompt = PromptTemplate(template=prompt_template, input_variables=["ocr_text", "language"])
        chain = prompt | llm_with_structured_output
        
        # Invoke the model to get the structured output
        response = chain.invoke({"ocr_text": ocr_text, "language": self.language})
        # Return the LetterDetails object
        return response

# Application class to coordinate the entire process
class Application:
    def __init__(self):
        # Load environment variables
        load_dotenv()
        # Set parameters from environment variables
        self.pdf_path = os.getenv('PDF_PATH')
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.language = os.getenv('LANGUAGE')
        # Directory paths
        self.csv_dir = os.getenv('CSV_FILES')  # Update this with actual path
        self.output_dir = 'output'  # Output directory

    def run(self):
        # Initialize PDFProcessor with required parameters
        pdf_processor = PDFProcessor(self.pdf_path, self.language, self.openai_api_key)
        # Convert the PDF to images
        images = pdf_processor.convert_pdf_to_images()
        
        # Perform OCR on the images to extract text
        ocr_text = pdf_processor.perform_ocr(images)
        print("OCR Text:", ocr_text)

        # Analyze the extracted text to structure it
        letter_details = pdf_processor.analyze_text(ocr_text)
        print("Extracted Details:")
        print(letter_details)

        # Extract the receiver name from the analyzed details
        receiver_name = letter_details.receiver

        # Initialize WorkerManager for the csv directory
        worker_manager = WorkerManager(self.csv_dir)
        # Find the corresponding worker using the receiver's name
        worker_name, csv_filename = worker_manager.find_worker_by_receiver(receiver_name)

        # Check if the worker is found in any CSV
        if not worker_name:
            print(f"Receiver {receiver_name} not found in any CSV files.")
            return

        # Initialize FolderManager for the output directory
        folder_manager = FolderManager(self.output_dir)
        # Create folder structure for worker and receiver
        receiver_folder = folder_manager.create_folder_structure(worker_name, receiver_name)

        # Save the PDF into the created receiver's folder
        pdf_processor.save_pdf(images, receiver_folder)

# Entry point for the application
if __name__ == "__main__":
    # Create an Application instance
    app = Application()
    # Run the application
    app.run()