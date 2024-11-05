import io
import time
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from diffusers import DiffusionPipeline, TCDScheduler
from huggingface_hub import hf_hub_download
from PIL import Image
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import random

# Initialize FastAPI app
app = FastAPI()

# Load the Stable Diffusion model with LoRA weights
base_model_id = "runwayml/stable-diffusion-v1-5"
repo_name = "ByteDance/Hyper-SD"
ckpt_name = "Hyper-SD15-1step-lora.safetensors"

pipe = DiffusionPipeline.from_pretrained(
    base_model_id,
    torch_dtype=torch.float16,
    variant="fp16",
    safety_checker=None,
    cache_dir="cache/sd_v1_5",
).to("cuda")

# Load and fuse the LoRA weights
pipe.load_lora_weights(hf_hub_download(
    repo_name,
    ckpt_name,
    cache_dir="cache/lora/hyper_sd",
))
pipe.fuse_lora()
pipe.scheduler = TCDScheduler.from_config(pipe.scheduler.config)

# Default parameters
current_params = {
    "prompt": "a cat",
    "width": 512,
    "height": 512,
    "guidance_scale": 0,
    "seed": -1
}

def generate_frames():
    """Generate frames from the Stable Diffusion model at approximately 24 FPS."""
    while True:
        start_time = time.time()

        # Set the random seed if `seed` is -1, else use the specified seed
        seed = current_params["seed"]
        if seed == -1:
            seed = random.randint(0, 2**32 - 1)

        # Generate an image from the current parameters
        generator = torch.manual_seed(seed)
        image = pipe(
            prompt=current_params["prompt"],
            num_inference_steps=1,
            guidance_scale=current_params["guidance_scale"],
            width=current_params["width"],
            height=current_params["height"],
            generator=generator
        ).images[0]

        # Convert to JPEG format
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='JPEG')
        frame = img_byte_arr.getvalue()

        # Yield the frame in MJPEG format
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n\r\n")

        # Calculate the time taken and sleep to maintain ~24 FPS
        elapsed_time = time.time() - start_time
        time_to_sleep = max(0, (1/24) - elapsed_time)
        time.sleep(time_to_sleep)

@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

# API to update parameters
class ConfigParams(BaseModel):
    prompt: str
    width: int
    height: int
    guidance_scale: float = 0  # Default to 0 if not provided
    seed: int = -1  # Use -1 to indicate a random seed

@app.post("/update_config")
async def update_config(params: ConfigParams):
    global current_params
    current_params["prompt"] = params.prompt
    current_params["width"] = max(64, (params.width // 64) * 64)  # Ensure width is a multiple of 64
    current_params["height"] = max(64, (params.height // 64) * 64)  # Ensure height is a multiple of 64
    current_params["guidance_scale"] = max(0, params.guidance_scale)  # Ensure guidance_scale is at least 0
    current_params["seed"] = params.seed  # Set seed (random if -1)
    return {"message": "Configuration updated successfully"}

# WebSocket for real-time config updates
@app.websocket("/ws/update_config")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            # Update current parameters with received data, validating each parameter
            prompt = data.get("prompt", current_params["prompt"])
            width = max(64, (data.get("width", current_params["width"]) // 64) * 64)
            height = max(64, (data.get("height", current_params["height"]) // 64) * 64)
            guidance_scale = max(0, data.get("guidance_scale", current_params["guidance_scale"]))
            
            # Try to retrieve the seed and handle cases where it might be None or invalid
            seed = data.get("seed", current_params["seed"])
            try:
                # Convert seed to an integer if it exists and isn't None
                seed = int(seed) if seed is not None else -1
            except ValueError:
                seed = -1  # Default to random if conversion fails
            
            # Debug print statements to confirm the received values
            print(f"Received seed: {seed}")

            # Update global parameters
            current_params.update({
                "prompt": prompt,
                "width": width,
                "height": height,
                "guidance_scale": guidance_scale,
                "seed": seed
            })

    except WebSocketDisconnect:
        print("WebSocket disconnected")