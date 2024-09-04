import os
import json
import asyncio
import requests
import logging
import re
from io import BytesIO
from dotenv import load_dotenv
from openai import OpenAI
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
API_KEY = os.getenv("ELEVENLABS_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = "https://api.elevenlabs.io/v1/text-to-speech/<voice-id>"

# Fetch vector store and assistant IDs from environment variables
FARMER_VECTOR_STORE_ID = os.getenv("FARMER_VECTOR_STORE_ID")
BEEKEEPER_VECTOR_STORE_ID = os.getenv("BEEKEEPER_VECTOR_STORE_ID")
FARMER_ASSISTANT_ID = os.getenv("FARMER_ASSISTANT_ID")
BEEKEEPER_ASSISTANT_ID = os.getenv("BEEKEEPER_ASSISTANT_ID")

app = FastAPI()
logging.basicConfig(level=logging.DEBUG)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the CHUNK_SIZE constant used for streaming data
CHUNK_SIZE = 1024

# Function to remove special characters and control characters like newlines
def filter_response(text):
    text = re.sub(r'[\x00-\x1F]+', ' ', text)  # Remove control characters
    return text

def chunk_response(text, max_length=100):
    # Split text into sentences
    sentences = re.split(r'(?<=[.!?]) +', text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 <= max_length:
            current_chunk += (sentence + " ")
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + " "
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

class AssistantManager:
    """Manages assistants and their vector stores."""

    def __init__(self):
        self.assistants = {
            "farmer": {"assistant_id": FARMER_ASSISTANT_ID, "thread_id": None},
            "beekeeper": {"assistant_id": BEEKEEPER_ASSISTANT_ID, "thread_id": None}
        }

    async def update_assistant_with_vector_store(self, assistant_type, vector_store_id):
        assistant_id = self.assistants[assistant_type]["assistant_id"]
        logging.debug(f"Updating assistant {assistant_id} with vector store {vector_store_id}")
        try:
            await asyncio.to_thread(
                client.beta.assistants.update,
                assistant_id=assistant_id,
                tool_resources={"file_search": {"vector_store_ids": [vector_store_id]}},
            )
        except Exception as e:
            logging.error(f"Failed to update assistant with vector store: {str(e)}")

    async def ensure_assistant_and_thread(self, assistant_type):
        """Ensure the assistant and thread exist for the given assistant type."""
        if assistant_type == "farmer":
            vector_store_id = FARMER_VECTOR_STORE_ID
        elif assistant_type == "beekeeper":
            vector_store_id = BEEKEEPER_VECTOR_STORE_ID
        else:
            logging.error(f"Unknown assistant type: {assistant_type}")
            return None
        
        assistant_id = self.assistants[assistant_type]["assistant_id"]
        if assistant_id:
            await self.update_assistant_with_vector_store(assistant_type, vector_store_id)
        else:
            logging.error(f"Assistant ID for {assistant_type} is not set in environment variables.")
            return None

        if self.assistants[assistant_type]["thread_id"] is None:
            thread = await asyncio.to_thread(client.beta.threads.create)
            self.assistants[assistant_type]["thread_id"] = thread.id

        return self.assistants[assistant_type]

    def format_messages(self, messages):
        formatted = []
        for message in messages:
            role = message.role
            text = filter_response(message.content[0].text.value)  # Filter response text
            formatted.append({"role": role, "text": text})
        return formatted

assistant_manager = AssistantManager()

class ConnectionManager:
    """Manages WebSocket connections."""
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_text(self, websocket: WebSocket, text: str):
        await websocket.send_text(text)

    async def broadcast_text(self, text: str):
        for connection in self.active_connections:
            await connection.send_text(text)

manager = ConnectionManager()

@app.websocket("/ws/chat")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            logging.debug(f"Data received: {data}")
            
            try:
                parsed_data = json.loads(data)
                assistant_type = parsed_data['assistant_type']
                user_prompt = parsed_data['prompt']
            except json.JSONDecodeError as e:
                logging.error(f"JSON decode error: {e}")
                await websocket.send_json({'success': False, 'error': f"Invalid JSON received: {e}"})
                continue
            
            logging.debug(f"Assistant type: {assistant_type}, User prompt: {user_prompt}")

            assistant_info = await assistant_manager.ensure_assistant_and_thread(assistant_type)
            logging.debug(f"Assistant info: {assistant_info}")
            assistant_id, thread_id = assistant_info['assistant_id'], assistant_info['thread_id']

            try:
                await asyncio.to_thread(
                    client.beta.threads.messages.create,
                    thread_id=thread_id,
                    role="user",
                    content=user_prompt
                )

                run = await asyncio.to_thread(
                    client.beta.threads.runs.create_and_poll,
                    thread_id=thread_id,
                    assistant_id=assistant_id
                )

                if run.status == 'completed':
                    messages = await asyncio.to_thread(
                        client.beta.threads.messages.list,
                        thread_id=thread_id
                    )
                    formatted_messages = assistant_manager.format_messages(messages.data)
                    logging.debug(f"Formatted messages: {formatted_messages}")
                    await manager.send_text(websocket, json.dumps({'success': True, 'response': formatted_messages}))
                else:
                    await manager.send_text(websocket, json.dumps({'success': False, 'error': f"Run did not complete successfully: {run.status}"}))

            except Exception as e:
                logging.error(f"Error during message handling: {e}")
                await manager.send_text(websocket, json.dumps({'success': False, 'error': str(e)}))

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logging.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)
        await websocket.close(code=1000, reason=str(e))


@app.websocket("/ws/audio")
async def websocket_audio(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            logging.debug(f"Received audio WebSocket message: {data}")
            assistant_type = data.get('type')
            response_text = data.get('response')

            voice_id = "1VeWDjFRlbnYGhhSvihY" if assistant_type == "farmer" else "3giviIRqITyV81sX559A"
            url = BASE_URL.replace('<voice-id>', voice_id)
            headers = {
                "Accept": "audio/mpeg",
                "Content-Type": "application/json",
                "xi-api-key": API_KEY,
            }

            # Break the response into smaller chunks
            chunks = chunk_response(response_text)

            for chunk in chunks:
                payload = {
                    "text": filter_response(chunk),  # Filter response text
                    "model_id": "eleven_multilingual_v2",
                    "voice_settings": {
                        "stability": 0.5,
                        "similarity_boost": 0.5,
                    }
                }

                response = requests.post(url, json=payload, headers=headers)
                response.raise_for_status()

                logging.debug("Audio data received successfully from ElevenLabs.")

                audio_bytes = response.content
                logging.debug(f"Audio bytes (first 100 bytes): {audio_bytes[:100]}")
                logging.debug(f"Total audio bytes length: {len(audio_bytes)}")

                await websocket.send_bytes(audio_bytes)

                # Calculate the duration of the audio using audio length and bitrate
                audio_duration = len(audio_bytes) / 16000  # Assuming 16000 bytes/sec
                await asyncio.sleep(audio_duration)  # Wait the length of the audio + 2 seconds

    except WebSocketDisconnect:
        logging.info("Client disconnected")
    except Exception as e:
        logging.error(f"Exception: {str(e)}")
        await websocket.send_json({'error': str(e)})

if __name__ == "__main__":
    import uvicorn
    # Run the FastAPI server
    uvicorn.run(app, host="0.0.0.0", port=6969)
