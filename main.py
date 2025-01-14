import os
from dotenv import load_dotenv
import pytesseract
from pdf2image import convert_from_path
from langchain import LLMChain, PromptTemplate
from langchain.llms import OpenAI

def main():


    
    # Load environment variables
    load_dotenv()
    pdf_path = os.getenv('PDF_PATH')
    openai_api_key = os.getenv('OPENAI_API_KEY')
    language = os.getenv('LANGUAGE')
    
    # Set OpenAI API key
    llm = OpenAI(api_key=openai_api_key)

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
            "Extract the sender, receiver, date of writing, type of letter (like a bill, offer, letter of application or anything else you can think of), and provide a summary of the following letter text: {ocr_text}"
        )
        prompt = PromptTemplate(template=prompt_template, input_variables=["ocr_text"])
        chain = LLMChain(llm=llm, prompt=prompt)
        
        response = chain.run(ocr_text=ocr_text)
        
        # Extract and print the results
        print("Extracted Details:")
        print(response)

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()