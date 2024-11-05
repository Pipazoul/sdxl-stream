import torch
from diffusers import DiffusionPipeline, TCDScheduler
from huggingface_hub import hf_hub_download
from fastapi import FastAPI, WebSocket, Request
from fastapi.templating import Jinja2Templates
import uvicorn
from PIL import Image
import io
import base64
import asyncio

# Initialize FastAPI app and Jinja2 templates
app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Load model
base_model_id = "runwayml/stable-diffusion-v1-5"
repo_name = "ByteDance/Hyper-SD"
ckpt_name = "Hyper-SD15-1step-lora.safetensors"

pipe = DiffusionPipeline.from_pretrained(base_model_id, torch_dtype=torch.float16, variant="fp16",  safety_checker=None).to("cuda")
pipe.load_lora_weights(hf_hub_download(repo_name, ckpt_name))
pipe.fuse_lora()
pipe.scheduler = TCDScheduler.from_config(pipe.scheduler.config)

# Default parameters
params = {
    "prompt": "A man smiling",
    "num_inference_steps": 1,
    "guidance_scale": 0,
    "width": 512,
    "height": 512,
}

# Route to serve the HTML page
@app.get("/")
async def get_home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/config")
async def get_config(request: Request):
    return templates.TemplateResponse("config.html", {"request": request})

# WebSocket route for streaming images
@app.websocket("/stream")
async def image_stream(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # Generate image using updated parameters
            image = pipe(
                params["prompt"],
                num_inference_steps=params["num_inference_steps"],
                guidance_scale=params["guidance_scale"],
                width=params["width"],
                height=params["height"],
            ).images[0]
            
            # Convert image to bytes
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG")
            image_bytes = buffer.getvalue()
            
            # Encode image to base64
            encoded_image = base64.b64encode(image_bytes).decode("utf-8")
            
            # Send the base64-encoded image over WebSocket
            await websocket.send_text(f"data:image/jpeg;base64,{encoded_image}")
            
            # Control frame rate by setting a delay (approx 24 FPS)
            await asyncio.sleep(1 / 60)
    except Exception as e:
        print(f"Error: {e}")
    finally:
        await websocket.close()

# WebSocket route for updating configuration
@app.websocket("/update_config")
async def update_config(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # Receive JSON data from client
            data = await websocket.receive_json()
            # Update parameters based on received data
            params["prompt"] = data.get("prompt", params["prompt"])
            params["width"] = data.get("width", params["width"])
            params["height"] = data.get("height", params["height"])
            params["guidance_scale"] = data.get("guidance_scale", params["guidance_scale"])
    except Exception as e:
        print(f"Error: {e}")
    finally:
        await websocket.close()
