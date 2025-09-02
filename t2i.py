import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import os

def generate_image_from_text(prompt, output_path="generated_image.png", num_inference_steps=50):
    """
    Generate an image from text using Stable Diffusion 2
    
    Args:
        prompt (str): Text description of the image you want to generate
        output_path (str): Path to save the generated image
        num_inference_steps (int): Number of denoising steps (higher = better quality but slower)
    """
    
    # Check if GPU is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load the Stable Diffusion 2 model
    model_id = "stabilityai/stable-diffusion-2-1"
    
    # Create the pipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        safety_checker=None  # Optional: disable safety checker for more creative freedom
    )
    
    # Move to device
    pipe = pipe.to(device)
    
    # Generate the image
    print(f"Generating image for prompt: '{prompt}'")
    
    with torch.autocast(device) if device == "cuda" else torch.no_grad():
        result = pipe(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=7.5,  # Controls how closely the image follows the prompt
            generator=torch.Generator(device=device).manual_seed(42)  # Optional: for reproducible results
        )
    
    # Save the image
    image = result.images[0]
    image.save(output_path)
    print(f"Image saved to: {output_path}")
    
    return image

# Example usage
if __name__ == "__main__":
    # Your text prompt
    prompt = input("Prompt: ")
    
    # Generate and save the image
    generated_image = generate_image_from_text(
        prompt=prompt,
        output_path="output.png",
        num_inference_steps=50
    )
    
    # Display the image (optional)
    # generated_image.show()