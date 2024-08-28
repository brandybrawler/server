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

# Improved function to remove special characters and control characters like newlines
def filter_response(text):
    # Define regex pattern to remove special characters and newlines
    pattern = re.compile(r'[\x00-\x1F]+')
    clean_text = re.sub(pattern, ' ', text)  # Replace with a space
    return clean_text

class AssistantManager:
    """Manages assistants and their vector stores."""

    def __init__(self):
        self.assistants = {}
        self.vector_store_ids = self.load_persistent_vector_store_ids()

    def load_persistent_vector_store_ids(self):
        if os.path.exists("vector_store_ids.json"):
            with open("vector_store_ids.json", "r") as file:
                return json.load(file)
        return {"farmer": None, "beekeeper": None}

    def save_persistent_vector_store_ids(self):
        with open("vector_store_ids.json", "w") as file:
            json.dump(self.vector_store_ids, file)

    async def create_assistant(self, assistant_type, name, instructions):
        """Create an assistant with specific instructions."""
        try:
            assistant = await asyncio.to_thread(
                client.beta.assistants.create,
                name=name,
                instructions=instructions,
                tools=[{"type": "file_search"}],
                model="gpt-4o",
            )
            return assistant.id
        except Exception as e:
            logging.error(f"Failed to create assistant: {str(e)}")
            return None

    async def create_vector_store(self, assistant_type):
        folder_path = "Farmer data" if assistant_type == "farmer" else "Beekeeper data"
        file_paths = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith(('.pdf', '.txt', '.md', '.docx'))]
        file_streams = [open(path, "rb") for path in file_paths]

        try:
            vector_store = await asyncio.to_thread(
                client.beta.vector_stores.create,
                name=f"{assistant_type.capitalize()} Data"
            )
            file_batch = await asyncio.to_thread(
                client.beta.vector_stores.file_batches.upload_and_poll,
                vector_store_id=vector_store.id,
                files=file_streams
            )
            self.vector_store_ids[assistant_type] = vector_store.id
            self.save_persistent_vector_store_ids()
            logging.debug(file_batch.status)
            logging.debug(file_batch.file_counts)
            return vector_store.id
        except Exception as e:
            logging.error(f"Failed to upload files: {str(e)}")
            return None
        finally:
            for stream in file_streams:
                stream.close()

    async def update_assistant_with_vector_store(self, assistant_id, vector_store_id):
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
        if assistant_type not in self.assistants:
            if assistant_type == "farmer":
                name = "Farm Expert"
                instructions = (
                    "You are an old farmer in Kenya with vast knowledge on farming and are willing to share it with others. "
                    "Be casual with your responses and let your responses be short."
                )
            elif assistant_type == "beekeeper":
                name = "Beekeeper Expert"
                instructions = (
                    "You are an expert beekeeper with vast knowledge on beekeeping and are willing to share it with others. "
                    "Be casual with your responses and let your responses be short."
                )
            else:
                logging.error(f"Unknown assistant type: {assistant_type}")
                return None
            
            assistant_id = await self.create_assistant(assistant_type, name, instructions)
            if self.vector_store_ids.get(assistant_type) is None:
                vector_store_id = await self.create_vector_store(assistant_type)
            else:
                vector_store_id = self.vector_store_ids[assistant_type]
            await self.update_assistant_with_vector_store(assistant_id, vector_store_id)

            self.assistants[assistant_type] = {'assistant_id': assistant_id, 'thread_id': None}

        if self.assistants[assistant_type]['thread_id'] is None:
            thread = await asyncio.to_thread(client.beta.threads.create)
            self.assistants[assistant_type]['thread_id'] = thread.id

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
    assistant_type = "farmer"  # or "beekeeper" based on client preference, you can modify this logic
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            user_prompt = json.loads(data)['prompt']
            
            assistant_info = await assistant_manager.ensure_assistant_and_thread(assistant_type)
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
                await manager.send_text(websocket, json.dumps({'success': False, 'error': str(e)}))

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        manager.disconnect(websocket)
        await websocket.close(code=1000, reason=str(e))


@app.websocket("/ws/audio")
async def websocket_audio(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            assistant_type = data.get('type')
            response_text = data.get('response')

            voice_id = "1VeWDjFRlbnYGhhSvihY" if assistant_type == "farmer" else "3giviIRqITyV81sX559A"

            url = BASE_URL.replace('<voice-id>', voice_id)

            headers = {
                "Accept": "audio/mpeg",
                "Content-Type": "application/json",
                "xi-api-key": API_KEY,
            }

            payload = {
                "text": filter_response(response_text),  # Filter response text
                "model_id": "eleven_multilingual_v2",
                "voice_settings": {
                    "stability": 0.5,
                    "similarity_boost": 0.5,
                }
            }

            response = requests.post(url, json=payload, headers=headers)
            response.raise_for_status()

            logging.debug("Audio data received successfully.")

            audio_bytes = response.content
            logging.debug(f"Audio bytes (first 100 bytes): {audio_bytes[:100]}")
            logging.debug(f"Total audio bytes length: {len(audio_bytes)}")

            await websocket.send_bytes(audio_bytes)

    except WebSocketDisconnect:
        logging.info("Client disconnected")
    except Exception as e:
        logging.error(f"Exception: {str(e)}")
        await websocket.send_json({'error': str(e)})

if __name__ == "__main__":
    import uvicorn
    # Run the FastAPI server
    uvicorn.run(app, host="0.0.0.0", port=6969)
