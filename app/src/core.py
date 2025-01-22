import os
import shutil
from pubsub import pub

from app.src.folder_application import FolderManager
from app.src.pdf_processor import PDFProcessor
from app.src.worker_manager import WorkerManager


class CoreApplication:
    def __init__(self, openai_api_key, language, csv_dir, output_dir):
        self.openai_api_key = openai_api_key
        self.language = language
        self.csv_dir = csv_dir
        self.output_dir = output_dir

    def run(self, pdf_path):
        images = self.convert_pdf_to_images(pdf_path)
        if not images:
            return

        ocr_text = self.perform_ocr(images)
        if not ocr_text:
            return

        letter_details = self.analyze_text(ocr_text)
        if not letter_details:
            return

        worker_name, csv_filename, matched_receiver = self.find_worker(letter_details)
        if not worker_name:
            return

        self.save_pdf_to_folder(pdf_path, letter_details, worker_name, matched_receiver)

    def convert_pdf_to_images(self, pdf_path):
        pdf_processor = PDFProcessor(self.language, self.openai_api_key, self.csv_dir)
        failed_folder = os.path.join(self.output_dir, "unrecognized")
        os.makedirs(failed_folder, exist_ok=True)
        original_pdf_name = os.path.basename(pdf_path)

        try:
            return pdf_processor.convert_pdf_to_images(pdf_path)
        except Exception as e:
            pub.sendMessage('log_event', message=f"Failed to convert PDF to images: {e}")
            failed_path = os.path.join(failed_folder, original_pdf_name)
            pdf_processor.save_pdf([], failed_path)
            return None

    def perform_ocr(self, images):
        pdf_processor = PDFProcessor(self.language, self.openai_api_key, self.csv_dir)
        try:
            ocr_text = pdf_processor.perform_ocr(images)
            pub.sendMessage('print_event', message="OCR Text: " + ocr_text)
            return ocr_text
        except Exception as e:
            pub.sendMessage('log_event', message=f"OCR failed: {e}")
            return None

    def analyze_text(self, ocr_text):
        pdf_processor = PDFProcessor(self.language, self.openai_api_key, self.csv_dir)
        try:
            letter_details = pdf_processor.analyze_text(ocr_text)
            pub.sendMessage('log_event', message="Extracted Details:")
            pub.sendMessage('log_event', message=str(letter_details))
            return letter_details
        except Exception as e:
            pub.sendMessage('log_event', message=f"Failed to extract letter details: {e}")
            return None

    def find_worker(self, letter_details):
        receiver_name = letter_details.receiver
        worker_manager = WorkerManager(self.csv_dir)
        worker_name, csv_filename, matched_receiver = worker_manager.find_worker_by_receiver(receiver_name)

        pub.sendMessage('log_event', message=f"Receiver: {receiver_name}")
        pub.sendMessage('log_event', message=f"Matched Receiver: {matched_receiver}")
        pub.sendMessage('log_event', message=f"Worker Name: {worker_name}")

        if not worker_name:
            if letter_details.responsible_person:
                worker_name = letter_details.responsible_person
                matched_receiver = letter_details.receiver
                csv_filename = f"{worker_name}.csv"
            else:
                pub.sendMessage('log_event', message=f"Receiver {receiver_name} not found in any CSV files.")
                return "unrecognized", None, letter_details.receiver or "unknown"

        return worker_name, csv_filename, matched_receiver

    def save_pdf_to_folder(self, pdf_path, letter_details, worker_name, matched_receiver):
        pdf_processor = PDFProcessor(self.language, self.openai_api_key, self.csv_dir)
        failed_folder = os.path.join(self.output_dir, "failed")
        os.makedirs(failed_folder, exist_ok=True)
        original_pdf_name = os.path.basename(pdf_path)

        try:
            folder_manager = FolderManager(self.output_dir)
            receiver_folder = folder_manager.find_or_create_folder(worker_name, matched_receiver)
            date_received = letter_details.date_of_writing
            organization = letter_details.organisation or "Private"
            worker = worker_name
            letter_type = letter_details.type_of_letter

            filename = f"{date_received}_{organization}_{worker}_{letter_type}.pdf".replace(' ', '_').replace('/', '-').replace('\\', '-')
            pdf_output_path = os.path.join(receiver_folder, filename)
            
            # Copy the original PDF to the new location
            shutil.copy2(pdf_path, pdf_output_path)
            pub.sendMessage('log_event', message=f"PDF saved to: {pdf_output_path}")


            # Delete the original PDF if saved successfully
            os.remove(pdf_path)
        except Exception as e:
            pub.sendMessage('log_event', message=f"Failed to save PDF to structured folder: {e}")
            failed_path = os.path.join(failed_folder, original_pdf_name)
            shutil.copy2(pdf_path, failed_path)