```python
import streamlit as st
from PIL import Image
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Model.clip_model import CLIPEncoder
from Utils.similarity import compute_similarity

@st.cache_resource
def load_model():
    return CLIPEncoder()

model = load_model()

st.set_page_config(page_title="Multimodal Misinformation Detector", layout="centered")

st.title("🧠 Multimodal Misinformation Detector")

uploaded_file = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"])

caption = st.text_input("Enter Caption for the Image")

threshold = st.slider("Select Similarity Threshold", 0.0, 1.0, 0.26)

if uploaded_file and caption:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Check Post"):

        with st.spinner("Analyzing..."):

            image_embedding = model.encode_image(image)

            text_embedding = model.encode_text(caption)

            similarity = compute_similarity(image_embedding, text_embedding)

        st.subheader(f"Similarity Score: {similarity:.3f}")

        st.progress(float(similarity))

        if similarity > threshold:
            st.error("⚠️ Potential Misinformation Detected")
        else:
            st.success("✅ Image and Caption Match")
```
