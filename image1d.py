import os
import json
from dotenv import load_dotenv
from io import BytesIO
from PIL import Image
import google.generativeai as genai
import requests

load_dotenv()

# Load prompts from JSON file
def load_prompts():
    with open('prompts.json', 'r', encoding='utf-8') as f:
        return json.load(f)

def location_based_analysis(image_path, latitude="", longitude="", language="english"):
    """Use latitude and longitude to help identify the image more accurately"""
    
    # Configure Gemini
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    model = genai.GenerativeModel('gemini-2.5-flash')
    
    # Load image
    if image_path.startswith('http'):
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(image_path, headers=headers, timeout=10)
        image = Image.open(BytesIO(response.content))
    else:
        image = Image.open(image_path)
    
    # Load prompts
    prompts = load_prompts()
    
    # Enhanced prompt with coordinates context
    if latitude and longitude:
        prompt_template = prompts.get(language, prompts["english"]).get("with_coordinates")
        prompt = prompt_template.format(latitude=latitude, longitude=longitude)
    else:
        prompt = prompts.get(language, prompts["english"]).get("basic")

    try:
        response = model.generate_content(
            [prompt, image],
            generation_config=genai.types.GenerationConfig(
                temperature=0
            )
        )
        return response.text
    except Exception as e:
        return f"Error in analysis: {e}"

# Example usage
if __name__ == "__main__":
    # Fixed image path
    image_path = "bcd.jpg"
    
    # Hardcoded coordinates
    latitude = "24.7460° N"  
    longitude = "90.4179° E "
    
    # Hardcoded language selection
    language = "english"  # Change to: "english", "chinese", or "traditional_chinese"
    
    print(f"\nAnalyzing image: {image_path}")
    print(f"Coordinates: {latitude}, {longitude}")
    print(f"Language: {language}")
    
    result = location_based_analysis(image_path, latitude, longitude, language)
    
    print("\nANALYSIS RESULT:")
    print(result)