# Import necessary libraries
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import os
from textblob import TextBlob
import re
import base64
from dotenv import load_dotenv  # Import dotenv
import assemblyai as aai

# Load environment variables from .env file
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Configure CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to specific origins in production.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define schemas
class RawComplaint(BaseModel):
    description: str

class ImageURL(BaseModel):
    url: str

class AudioURL(BaseModel):
    url: str

# Set API keys from environment variables
aai.settings.api_key = os.getenv("ASSEMBLYAI_API_KEY")  # AssemblyAI API key
if not aai.settings.api_key:
    raise Exception("ASSEMBLYAI_API_KEY is not set in the environment variables.")

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")  # Mistral API key
if not MISTRAL_API_KEY:
    raise Exception("MISTRAL_API_KEY is not set in the environment variables.")

# API call functions
def transcribe_audio(audio_source, is_url: bool = True):
    """
    Transcribes audio using AssemblyAI.
    
    Args:
        audio_source: Either URL to audio file or path to temporary saved file
        is_url: Boolean indicating if audio_source is a URL
    
    Returns:
        Transcription text
    """
    try:
        # Initialize the transcriber
        transcriber = aai.Transcriber()
        
        # Start transcription
        transcript = transcriber.transcribe(audio_source)
        
        # Check for errors
        if transcript.status == aai.TranscriptStatus.error:
            raise Exception(f"Transcription error: {transcript.error}")
            
        # Return the transcription text
        return transcript.text
        
    except Exception as e:
        print(f"Error during transcription: {str(e)}")
        raise Exception(f"Failed to transcribe audio: {str(e)}")

# Utility functions
def process_text(text: str) -> str:
    """
    Process the raw text by performing advanced cleaning, normalization,
    spelling correction, and grammar improvements.
    """
    # Remove extra whitespace and unwanted characters.
    text = " ".join(text.split())
    text = re.sub(r"[^a-zA-Z0-9.,!?;:()'\" \n]", "", text)

    # Convert to lowercase for uniformity.
    text = text.lower()

    # Correct spelling using TextBlob.
    blob = TextBlob(text)
    text = str(blob.correct())

    # Capitalize the first letter of each sentence.
    text = ". ".join(sentence.capitalize() for sentence in text.split(". "))

    # Remove repetitive words or phrases.
    words = text.split()
    processed_words = []
    for i, word in enumerate(words):
        if i == 0 or word != words[i - 1]:
            processed_words.append(word)
    text = " ".join(processed_words)

    return text

def encode_image_to_base64(image_data: bytes) -> str:
    """
    Encode image bytes to base64 string.
    """
    return base64.b64encode(image_data).decode('utf-8')

# API call functions
def call_llm_api(processed_text: str) -> str:
    """
    Calls the LLM endpoint from Mistral with advanced prompting.
    """
    url = "https://api.mistral.ai/v1/chat/completions"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {MISTRAL_API_KEY}"
    }

    system_prompt = """
    You are a helpful assistant designed to correct and improve user-provided complaint descriptions.
    Your goal is to transform the input text into clear, grammatically correct, and professional English.
    """

    user_prompt = f"""
    Please correct and refine the following complaint description:

    "{processed_text}"

    Ensure the corrected description is:
    - Grammatically accurate.
    - Free of spelling errors.
    - Written in a professional tone.
    - Concise and clear.

    Provide only the corrected description as the output.
    """

    payload = {
        "model": "ministral-8b-latest",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    }

    response = requests.post(url, json=payload, headers=headers)

    if response.status_code != 200:
        raise Exception(f"LLM API call failed: {response.text}")

    try:
        response_json = response.json()
        if "choices" in response_json and len(response_json["choices"]) > 0 and "message" in response_json["choices"][0] and "content" in response_json["choices"][0]["message"]:
            result = response_json["choices"][0]["message"]["content"].strip()
        else:
            raise Exception("Invalid or incomplete response from LLM API.")

    except (KeyError, IndexError, TypeError, ValueError) as e:
        raise Exception(f"Error processing LLM API response: {str(e)}")

    return result

def call_pixtral_api(image_content: str, is_url: bool = False) -> str:
    """
    Calls the Pixtral 12B model from Mistral API for image description.
    """
    url = "https://api.mistral.ai/v1/chat/completions"
    api_key = os.getenv("MISTRAL_API_KEY")  # Use environment variable for security
    if not api_key:
        raise Exception("MISTRAL_API_KEY environment variable not set.")

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    system_prompt = """
    You are a helpful assistant specialized in providing detailed descriptions of images.
    Examine the image carefully and provide a comprehensive description that includes:
    - Main subjects and their relationships
    - Visual elements (colors, composition, lighting)
    - Setting and context
    - Any text visible in the image
    - Overall mood or atmosphere

    Your description should be clear, accurate, and well-structured.
    """

        # Prepare content based on whether it's a URL or base64 image
    if is_url:
        content = [{"type": "text", "text": "Please describe this image in detail:"}, 
                {"type": "image_url", "image_url": {"url": image_content}}]
    else:
        # For base64, we need to prefix with the data URI scheme
        data_uri = f"data:image/jpeg;base64,{image_content}"
        content = [{"type": "text", "text": "Please describe this image in detail:"}, 
                {"type": "image_url", "image_url": {"url": data_uri}}]
    
    payload = {
    "model": "pixtral-12b-2409",
    "messages": [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content}
    ]
    }

    response = requests.post(url, json=payload, headers=headers)

    if response.status_code != 200:
        raise Exception(f"Pixtral API call failed: {response.text}")

    try:
        response_json = response.json()
        if "choices" in response_json and len(response_json["choices"]) > 0 and "message" in response_json["choices"][0] and "content" in response_json["choices"][0]["message"]:
            result = response_json["choices"][0]["message"]["content"].strip()
        else:
            raise Exception("Invalid or incomplete response from Pixtral API.")

    except (KeyError, IndexError, TypeError, ValueError) as e:
        raise Exception(f"Error processing Pixtral API response: {str(e)}")

    return result

# API endpoints
@app.post("/process-complaint")
async def process_complaint(complaint: RawComplaint):
    """
    API endpoint that accepts a raw complaint description,
    processes the text, and sends it to an LLM for correction.
    """
    try:
        processed_text = process_text(complaint.description)
        accurate_description = call_llm_api(processed_text)
        return {"accurate_description": accurate_description}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/describe-image/upload")
async def describe_uploaded_image(file: UploadFile = File(...)):
    """
    API endpoint that accepts an uploaded image file and returns a detailed description.
    """
    try:
        # Validate file type
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Unsupported file type. Please upload an image.")

        # Read the file data
        image_data = await file.read()

        # Check if the file is empty
        if not image_data:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        # Encode the image to base64
        base64_image = encode_image_to_base64(image_data)

        # Call the Pixtral API to describe the image
        image_description = call_pixtral_api(base64_image)

        return {"description": image_description}

    except HTTPException as http_exc:
        # Return HTTP exceptions as-is
        raise http_exc
    except Exception as e:
        # Handle unexpected errors
        raise HTTPException(status_code=500, detail=f"An error occurred while processing the image: {str(e)}")

@app.post("/describe-image/url")
async def describe_image_from_url(image_data: ImageURL):
    """
    API endpoint that accepts an image URL and returns a detailed description.
    """
    try:
        image_description = call_pixtral_api(image_data.url, is_url=True)
        return {"description": image_description}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/transcribe-audio/upload")
async def transcribe_uploaded_audio(file: UploadFile = File(...)):
    """
    API endpoint that accepts an uploaded audio file and returns the transcription.
    """
    try:
        # Validate file type more robustly
        allowed_extensions = ['.mp3', '.wav', '.ogg', '.m4a']
        file_extension = os.path.splitext(file.filename)[1].lower()
        
        if file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file type: {file_extension}. Allowed types are: {', '.join(allowed_extensions)}"
            )

        # Save the uploaded file temporarily
        temp_file_path = f"temp_{file.filename}"
        try:
            # Read the file data
            audio_data = await file.read()
            
            # Check if the file is empty
            if not audio_data or len(audio_data) < 100:
                raise HTTPException(status_code=400, detail="Uploaded file is empty or corrupted.")
            
            # Write to temporary file
            with open(temp_file_path, "wb") as temp_file:
                temp_file.write(audio_data)
            
            # Transcribe the audio file
            transcription = transcribe_audio(temp_file_path, is_url=False)
            
            return {"transcription": transcription}
            
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    except HTTPException as http_exc:
        # Return HTTP exceptions as-is
        raise http_exc
    except Exception as e:
        # Handle unexpected errors
        raise HTTPException(status_code=500, detail=f"An error occurred while processing the audio: {str(e)}")
    
    
@app.post("/transcribe-audio/url")
async def transcribe_audio_from_url(audio_data: AudioURL):
    """
    API endpoint that accepts an audio URL and returns the transcription.
    """
    try:
        # Validate URL (basic check)
        if not audio_data.url.startswith(('http://', 'https://')):
            raise HTTPException(status_code=400, detail="Invalid URL. URL must start with http:// or https://")
            
        # Transcribe the audio from URL
        transcription = transcribe_audio(audio_data.url, is_url=True)
        
        return {"transcription": transcription}
        
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))