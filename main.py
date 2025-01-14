import os
from dotenv import load_dotenv
import pytesseract
from pdf2image import convert_from_path
import openai

def main():
    # Load environment variables
    load_dotenv()
    pdf_path = os.getenv('PDF_PATH')
    openai_api_key = os.getenv('OPENAI_API_KEY')
    language = os.getenv('LANGUAGE')
    
    # Set OpenAI API key
    client = openai.OpenAI(api_key= openai_api_key)

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
        
        # Analyze the OCR output using OpenAI API
        print("Analyzing text using OpenAI API...")
        prompt = (
            f"Extract the sender, receiver, date of writing, type of letter (like a bill, offer, letter of application or anything else you can think of), and provide a summary of the following letter text: {ocr_text}"
        )
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": f"You are a helpful assistant to read letters. You get a text provided from a scanner which has a letter. Always try to extract the sender, recipient, date of writing, type of letter and provide a summary of the letter. Always answer in {language}"},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200,
            temperature=0.5
        )
        
        # Extract and print the results
        print("Extracted Details:")
        print(response)
        print(response.choices[0].message["content"])

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()