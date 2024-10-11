from diffusers import StableDiffusionPipeline
import torch
import gradio as gr

# Load the Stable Diffusion model
model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id)
pipe = pipe.to("cuda")  # Use GPU

# Define the function that generates images from text prompts
def generate_image(prompt):
    image = pipe(prompt).images[0]
    return image

# Create the Gradio interface
iface = gr.Interface(
    fn=generate_image,
    inputs="text",
    outputs="image",
    title="Text-to-Image Generator",
    description="Enter a prompt and generate an image using Stable Diffusion!"
)

# Launch the interface
iface.launch()
