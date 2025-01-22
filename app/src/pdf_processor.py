from langchain_openai import ChatOpenAI
from pdf2image import convert_from_path
import pytesseract
from langchain import LLMChain, PromptTemplate
from app.src.models import LetterDetails
from pubsub import pub
import os

class PDFProcessor:
    def __init__(self, language, api_key, csv_dir):
        self.language = language
        self.llm = ChatOpenAI(api_key=api_key, model="gpt-4o-mini")
        self.csv_dir = csv_dir

    def convert_pdf_to_images(self, pdf_path):
        return convert_from_path(pdf_path)

    def perform_ocr(self, images):
        ocr_text = ""
        for image in images:
            ocr_text += pytesseract.image_to_string(image)
        return ocr_text

    def save_pdf(self, images, save_path):
        images[0].save(save_path, save_all=True, append_images=images[1:])
        pub.sendMessage('log_event', message=f"PDF saved to: {save_path}")

    def analyze_text(self, ocr_text):
        try:
            prompt_template = (
                """
                    From the text provided, extract the names of the sender and recipient, including only the recipient's name. Identify the date of writing and provide a type of letter classified as an ultra-short summary in a maximum of 5 words in {language}. If a responsible person, whose name maybe appears in {responsible_persons_names}, is associated with the recipient, include their name; otherwise, leave that field empty. Text: {ocr_text}
                """
            )

            llm_with_structured_output = self.llm.with_structured_output(LetterDetails)
            prompt = PromptTemplate(template=prompt_template, input_variables=["ocr_text", "language"])
            chain = prompt | llm_with_structured_output
            response = chain.invoke({"ocr_text": ocr_text, "language": self.language, "responsible_persons_names": str(os.listdir(self.csv_dir))})
            return response
        except Exception as e:
            pub.sendMessage('log_event', message=f"Failed to analyze text: {e}")
            raise