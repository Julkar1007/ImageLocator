import os
import json
import time
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
    
    start_time = time.time()
    
    # Configure Gemini - using a faster model for quicker response
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    model = genai.GenerativeModel('gemini-2.0-flash')
    
    # Load image
    if image_path.startswith('http'):
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(image_path, headers=headers, timeout=10)
        image = Image.open(BytesIO(response.content))
    else:
        image = Image.open(image_path)
    
    # Optimize image for faster processing
    if image.mode != "RGB":
        image = image.convert("RGB")
    image.thumbnail((800, 800), Image.LANCZOS)  # Resize to max 800x800 to reduce data
    
    image_load_time = time.time() - start_time
    print(f"Image loaded in: {image_load_time:.2f}s")
    
    # Load prompts
    prompts = load_prompts()
    
    # Use your existing prompts from prompts.json
    if latitude and longitude:
        prompt_template = prompts.get(language, prompts["english"]).get("with_coordinates")
        prompt = f"The image is taken at the exact coordinates {latitude}, {longitude} in Dhaka, Bangladesh. " + prompt_template.format(latitude=latitude, longitude=longitude) + " Describe key architectural features like number of domes, minarets, presence of a tank or water body, and any unique elements to help identify the specific mosque at these coordinates. Provide a detailed historical overview including construction date, builder, architectural style details, historical context, and significance."
    else:
        prompt = prompts.get(language, prompts["english"]).get("basic") + " Describe key architectural features like number of domes, minarets, presence of a tank or water body, and any unique elements."

    try:
        gen_start_time = time.time()
        # Retry logic for rate limits
        max_retries = 3
        response = None
        for attempt in range(max_retries):
            try:
                response = model.generate_content(
                    [prompt, image],
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.0,
                        max_output_tokens=1000  # Increased to allow more detailed response
                    )
                )
                break  # Success
            except Exception as e:
                error_str = str(e).lower()
                if ("429" in error_str or "quota" in error_str or "rate limit" in error_str) and attempt < max_retries - 1:
                    # Try to parse retry_delay from error
                    import re
                    match = re.search(r'retry_delay \{\s*seconds:\s*(\d+)', str(e))
                    delay = int(match.group(1)) if match else 2 ** attempt
                    print(f"Rate limit hit, retrying in {delay}s...")
                    time.sleep(delay)
                else:
                    return f"Error in analysis: {e}"
        gen_time = time.time() - gen_start_time
        print(f"AI generation time: {gen_time:.2f}s")
        
        # Direct response access for high-quality output
        text = None
        try:
            text = response.text
        except Exception:
            pass
        if not text and hasattr(response, 'candidates') and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, 'safety_ratings'):
                for rating in candidate.safety_ratings:
                    if getattr(rating, 'blocked', False):
                        return "Response was blocked by safety filters."
            if hasattr(candidate, 'content') and candidate.content.parts:
                text = candidate.content.parts[0].text
        if text:
            total_time = time.time() - start_time
            print(f"Total function time: {total_time:.2f}s")
            return text
        else:
            return "No content generated"
            
    except Exception as e:
        return f"Error in analysis: {e}"

# Example usage
if __name__ == "__main__":
    total_start = time.time()
    
    # Fixed image path
    image_path = "abc.jpg"
    
    # Hardcoded coordinates - for Sat Gambuj Mosque in Dhaka
    latitude = "23.7578° N"  
    longitude = "90.3592° E"
    
    # Hardcoded language selection
    language = "english"
    
    print(f"\nAnalyzing image: {image_path}")
    print(f"Coordinates: {latitude}, {longitude}")
    print(f"Language: {language}")
    
    result = location_based_analysis(image_path, latitude, longitude, language)
    
    total_execution = time.time() - total_start
    print(f"\nTotal execution time: {total_execution:.2f}s")
    print("\nANALYSIS RESULT:")
    print(result)