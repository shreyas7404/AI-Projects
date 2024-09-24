# Task2: Image Generation with Pre-trained Models 

# Problem Statement:Utilize pre-trained generative models like DALL-E mini or Stable Diffusion to create images from text prompts

# Install necessary libraries
!pip install torch transformers diffusers

# Import libraries
import torch
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import StableDiffusionPipeline
from PIL import Image
from IPython.display import display

# Load the Stable Diffusion model
model_id = "CompVis/stable-diffusion-v1-4"
pipe = StableDiffusionPipeline.from_pretrained(model_id)

# Check if GPU is available and use it, otherwise use CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = pipe.to(device)

# Define the text prompt
prompt = "A beautiful landscape with ocean and beach"

# Generate the image
image = pipe(prompt).images[0]

# Display the image in the notebook
display(image)
