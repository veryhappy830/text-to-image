import os
import uuid
import base64
from io import BytesIO
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import torch
from diffusers import MVDreamPipeline
from PIL import Image
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="MVDream Text-to-Multi-Image Server",
    description="A server for generating multiple consistent images from text using MVDream",
    version="1.0.0"
)

class GenerationRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = None
    num_images: int = 4
    guidance_scale: float = 7.5
    num_inference_steps: int = 50
    seed: Optional[int] = None
    output_format: str = "base64"  # "base64" or "url"

class GenerationResponse(BaseModel):
    images: List[str]
    format: str
    prompt: str
    generation_id: str

# Global variable for the pipeline
pipeline = None
device = "cuda" if torch.cuda.is_available() else "cpu"

@app.on_event("startup")
async def load_model():
    """Load the MVDream model on startup"""
    global pipeline
    
    try:
        logger.info(f"Loading MVDream model on device: {device}")
        
        # Load the MVDream pipeline
        pipeline = MVDreamPipeline.from_pretrained(
            "ashawkey/mvdream-sd2.1-diffusers",
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        )
        
        pipeline = pipeline.to(device)
        
        # Enable memory efficient attention if available
        if hasattr(pipeline, "enable_xformers_memory_efficient_attention"):
            pipeline.enable_xformers_memory_efficient_attention()
        
        logger.info("MVDream model loaded successfully")
        
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise RuntimeError(f"Model loading failed: {str(e)}")

def image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 string"""
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

@app.post("/generate", response_model=GenerationResponse)
async def generate_images(request: GenerationRequest):
    """Generate multiple consistent images from text prompt"""
    
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if request.num_images not in [1, 2, 4]:
        raise HTTPException(
            status_code=400, 
            detail="num_images must be 1, 2, or 4 (MVDream supports these multi-view configurations)"
        )
    
    try:
        # Set seed if provided
        if request.seed is not None:
            torch.manual_seed(request.seed)
        
        logger.info(f"Generating {request.num_images} images for prompt: {request.prompt}")
        
        # Generate images
        with torch.autocast(device) if device == "cuda" else torch.no_grad():
            images = pipeline(
                prompt=request.prompt,
                negative_prompt=request.negative_prompt,
                num_images_per_prompt=request.num_images,
                guidance_scale=request.guidance_scale,
                num_inference_steps=request.num_inference_steps,
            ).images
        
        # Convert images to requested format
        if request.output_format == "base64":
            encoded_images = [image_to_base64(img) for img in images]
        else:
            # For URL format, you'd save to disk and return URLs
            # This is a placeholder - implement your own file serving logic
            encoded_images = [f"image_{i}.png" for i in range(len(images))]
        
        generation_id = str(uuid.uuid4())
        
        logger.info(f"Successfully generated {len(images)} images")
        
        return GenerationResponse(
            images=encoded_images,
            format=request.output_format,
            prompt=request.prompt,
            generation_id=generation_id
        )
        
    except Exception as e:
        logger.error(f"Generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": pipeline is not None,
        "device": device,
        "gpu_available": torch.cuda.is_available()
    }

@app.get("/info")
async def model_info():
    """Get model information"""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_name": "MVDream",
        "model_version": "sd2.1",
        "device": device,
        "dtype": str(pipeline.dtype),
        "supported_image_counts": [1, 2, 4]
    }

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )