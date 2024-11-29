from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.responses import StreamingResponse
from diffusers import DiffusionPipeline, DDIMScheduler,ControlNetModel, StableDiffusionControlNetPipeline
from huggingface_hub import hf_hub_download
from pydantic import BaseModel
from PIL import Image
import torch
import random
import io
import time
import cv2
import numpy as np

# Initialize FastAPI app
app = FastAPI()

# Load the Stable Diffusion model with LoRA weights
controlnet_checkpoint = "lllyasviel/control_v11p_sd15_canny"
base_model_id = "runwayml/stable-diffusion-v1-5"
repo_name = "ByteDance/Hyper-SD"
ckpt_name = "Hyper-SD15-1step-lora.safetensors"

controlnet = ControlNetModel.from_pretrained(
    controlnet_checkpoint, 
    torch_dtype=torch.float16,
    cache_dir="cache/controlnet",
)

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    base_model_id,
    controlnet=controlnet,
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
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")



# Default parameters
current_params = {
    "prompt": "a cat",
    "width": 512,
    "height": 512,
    "guidance_scale": 0,
    "seed": -1,
    "controlnet_conditioning_scale": 0.8,
    "init_image": None
}

def canny_edge_detection(image):
    low_threshold = 100
    high_threshold = 200
    image = cv2.Canny(image, low_threshold, high_threshold)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    control_image = Image.fromarray(image)
    control_image.save("control.png")
    return control_image

def generate_frames():
    """Generate frames from the Stable Diffusion model at approximately 24 FPS."""
    while True:
        start_time = time.time()

        # Set the random seed if `seed` is -1, else use the specified seed
        seed = current_params["seed"]
        if seed == -1:
            seed = random.randint(0, 2**32 - 1)

        # Prepare the generator and the input image (if provided)
        generator = torch.manual_seed(seed)
        init_image = current_params["init_image"]
        
        if init_image:
            # Resize the init image to the specified width and height
            init_image = init_image.resize((current_params["width"], current_params["height"]))
            init_image.save("resized_init_image.jpg")
        else:
            # create a random image if init_image is not provided
            init_image = Image.new("RGB", (current_params["width"], current_params["height"]), (255, 255, 255))
            init_image.save("resized_init_image.jpg")
        
        #convert image to numpy array
        init_image = np.array(init_image)
        control_image = canny_edge_detection(init_image)
        
        # Generate an image with or without init_image based on availability
        image = pipe(
            prompt=current_params["prompt"],
            num_inference_steps=1,
            guidance_scale=current_params["guidance_scale"],
            width=current_params["width"],
            height=current_params["height"],
            generator=generator,
            controlnet_conditioning_scale=current_params["controlnet_conditioning_scale"],
            image=control_image,
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

import base64
# WebSocket for real-time config updates
@app.websocket("/ws/update_config")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            prompt = data.get("prompt", current_params["prompt"])
            width = max(64, (data.get("width", current_params["width"]) // 64) * 64)
            height = max(64, (data.get("height", current_params["height"]) // 64) * 64)
            guidance_scale = max(0, data.get("guidance_scale", current_params["guidance_scale"]))
            
            # Retrieve and validate the seed
            seed = data.get("seed", current_params["seed"])
            try:
                seed = int(seed) if seed is not None else -1
            except ValueError:
                seed = -1  # Default to random if conversion fails
            
            # Check if there's a base64 image and update `init_image`
            base64_image = data.get("base64_image")
            if base64_image:
                image_data = base64.b64decode(base64_image)
                image = Image.open(io.BytesIO(image_data))
                current_params["init_image"] = image
            else:
                # Clear init_image if no base64_image provided in this update
                current_params["init_image"] = None

            # Update other parameters
            current_params.update({
                "prompt": prompt,
                "width": width,
                "height": height,
                "guidance_scale": guidance_scale,
                "seed": seed,
                "controlnet_conditioning_scale": 0.8
            })

    except WebSocketDisconnect:
        print("WebSocket disconnected")