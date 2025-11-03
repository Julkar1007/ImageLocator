import os
from dotenv import load_dotenv
from io import BytesIO
from PIL import Image
import google.generativeai as genai
import requests

load_dotenv()

def location_based_analysis(image_path, location=""):
    """Use location hint to help identify the image more accurately"""
    
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
    
    # Enhanced prompt with location context
    if location:
        prompt = f"""Analyze this image with the context that it might be from: {location}

Please identify:
1. What specific place, landmark, or location is this?
2. If it matches known places in {location}, provide the exact name and location
3. If it's a common object/scene, describe it accurately with paragraphs
4. Your confidence level in this identification ,and no description of the image.
5. If the image content is not found in {location}, reply "not found in {location}"

Be specific and factual. If unsure, say so."""
    else:
        prompt = """Analyze this image and identify what it shows:

1. Is this a specific landmark, building, or location?
2. If it's a famous place, provide its exact name and location
3. If it's a common object/scene, describe it accurately with paragraphs
4. Your confidence level in this identification ,and no description of the image.

Be specific and factual."""

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
    image_path = "M.jpg"
    location_hint = input("Enter location (city/country/region) or press Enter to skip: ").strip()
    
    print(f"\nAnalyzing image: {image_path}")
    print(f"Location: '{location_hint}'...")
    result = location_based_analysis(image_path, location_hint)
    
    print("ANALYSIS RESULT:")
    print(result)