import streamlit as st
from PIL import Image
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Model.clip_model import CLIPEncoder
from Utils.similarity import compute_similarity

st.title("Image–Caption Similarity Analyzer")

model = CLIPEncoder()

uploaded_image = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

st.write("Enter multiple captions (one per line)")

captions_input = st.text_area("Captions")

if uploaded_image and captions_input:

    image = Image.open(uploaded_image).convert("RGB")

    st.image(image, width="stretch")

    captions = captions_input.split("\n")

    image_embedding = model.encode_image(image)

    best_caption = ""
    best_score = -1

    st.subheader("Similarity Scores")

    for caption in captions:

        if caption.strip() == "":
            continue

        text_embedding = model.encode_text(caption)

        similarity = compute_similarity(image_embedding, text_embedding)

        st.write(f"{caption} → {similarity}")

        if similarity > best_score:
            best_score = similarity
            best_caption = caption

    st.success(f"Best Caption for this Image: {best_caption}")
