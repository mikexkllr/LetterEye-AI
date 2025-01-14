import os
from dotenv import load_dotenv
import pytesseract
from pdf2image import convert_from_path
from langchain import LLMChain, PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

class LetterDetails(BaseModel):
    sender: str = Field(..., description="The sender of the letter")
    receiver: str = Field(..., description="The receiver of the letter")
    date_of_writing: str = Field(..., description="The date the letter was written")
    type_of_letter: str = Field(..., description="The type of the letter")
    summary: str = Field(..., description="A summary of the letter")

def main():
    # Load environment variables
    load_dotenv()
    pdf_path = os.getenv('PDF_PATH')
    openai_api_key = os.getenv('OPENAI_API_KEY')
    language = os.getenv('LANGUAGE')
    
    # Set OpenAI API key
    llm = ChatOpenAI(api_key=openai_api_key, model="gpt-4o-mini")

    try:
        # Convert PDF to images
        print("Converting PDF to images...")
        images = convert_from_path(pdf_path)
        
        # Extract text from images using Tesseract OCR
        print("Performing OCR...")
        ocr_text = ""
        for image in images:
            # Use Tesseract to convert image to string
            ocr_text += pytesseract.image_to_string(image)

        print("OCR Text:" + ocr_text)
        
        # Analyze the OCR output using LangChain
        print("Analyzing text using LangChain...")
        prompt_template = (
            "Extract the sender, receiver, date of writing, type of letter (like a bill, offer, letter of application or anything else you can think of), and provide a summary in this language {language} of the following letter text: {ocr_text}"
        )
        llm_with_structered_output = llm.with_structured_output(LetterDetails)
        prompt = PromptTemplate(template=prompt_template, input_variables=["ocr_text", "language"])
        chain = prompt | llm_with_structered_output
        
        response = chain.invoke({"ocr_text": ocr_text, "language": language})
        
        # Extract and print the results
        letter_details = response
        print("Extracted Details:")
        print(letter_details)

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()