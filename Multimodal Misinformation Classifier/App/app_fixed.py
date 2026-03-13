import streamlit as st
from PIL import Image
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Model.clip_model import CLIPEncoder
from Utils.similarity import compute_similarity

st.title("Multimodal Misinformation Classifier")

model = CLIPEncoder()

uploaded_image = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

caption = st.text_input("Enter Caption")

threshold = 0.26

if uploaded_image and caption:

    image = Image.open(uploaded_image).convert("RGB")

    st.image(image, width="stretch")

    image_embedding = model.encode_image(image)
    text_embedding = model.encode_text(caption)

    similarity = compute_similarity(image_embedding, text_embedding)

    st.write("Similarity Score:", similarity)

    if similarity > threshold:
        st.success("Likely Real Information")
    else:
        st.error("Potential Misinformation")
