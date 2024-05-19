import streamlit as st
from PIL import Image
import requests
from io import BytesIO
import torch
from lavis.models import load_model_and_preprocess

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Streamlit app title
st.title("Image Caption Generator")

# Option to upload an image or provide an image URL
st.header("Upload an Image or Provide an Image URL")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
image_url = st.text_input("Or enter an image URL:")

# Button to generate the caption
generate_button = st.button("Generate Caption")


# Function to generate image caption
def generate_caption(image):
    # Load the BLIP caption base model with finetuned checkpoints on MSCOCO captioning dataset
    model, vis_processors, _ = load_model_and_preprocess(
        name="blip_caption", model_type="base_coco", is_eval=True, device=device
    )
    # Preprocess the image
    image = vis_processors["eval"](image).unsqueeze(0).to(device)
    # Generate caption
    caption = model.generate({"image": image})
    return caption[0]


# Function to load image from URL
def load_image_from_url(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content)).convert("RGB")
    return img


if generate_button:
    if uploaded_file is not None:
        try:
            # Load the uploaded image
            raw_image = Image.open(uploaded_file).convert("RGB")
            st.image(raw_image, caption='Uploaded Image', use_column_width=True)
            st.write("Image successfully uploaded.")

            # Generate and display the caption
            with st.spinner('Generating caption...'):
                caption = generate_caption(raw_image)
            st.write("**Generated Caption:**")
            st.write(f"**{caption}**")
        except Exception as e:
            st.write("Error processing the uploaded image. Please try again.")
            st.error(str(e))
    elif image_url:
        try:
            # Load the image from the URL
            raw_image = load_image_from_url(image_url)
            st.image(raw_image, caption='Image from URL', use_column_width=True)
            st.write("Image successfully loaded from URL.")

            # Generate and display the caption
            with st.spinner('Generating caption...'):
                caption = generate_caption(raw_image)
            st.write("**Generated Caption:**")
            st.write(f"**{caption}**")
        except Exception as e:
            st.write("Error loading image from URL. Please make sure the URL is correct and the image is accessible.")
            st.error(str(e))
    else:
        st.write("Please upload an image or provide an image URL.")
