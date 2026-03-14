# 🧠 Image–Caption Similarity Analyzer

A **Multimodal AI system** that identifies the **best suited caption for an image** by analyzing the **semantic similarity between visual and textual representations** using the CLIP vision–language model.

The system compares **image embeddings** and **text embeddings** to determine how well a caption describes the given image.

---

## 🚀 Live Demo

🔗 **Try the app here:**  
https://multimodalmisinformationclassifier.streamlit.app/

Upload an image and provide multiple captions to see **which caption best matches the image**.

The application is deployed using **Streamlit**.

---

## 💡 Project Idea

Images on the internet often have multiple possible descriptions.  
This project uses a **multimodal AI model** to determine **which caption best represents an image**.

Example:

Image → Dog running in a park  

Captions:
1. *A dog playing in a grassy park*  
2. *A cat sleeping on a sofa*  
3. *A car driving on a highway*

The system computes similarity scores and selects the caption that **best aligns with the image content**.

---

## 🏗️ System Architecture

Image → CLIP Encoder → Image Embedding  
Captions → CLIP Encoder → Text Embeddings  

↓

Cosine Similarity

↓

Caption Ranking

↓

Best Caption Selected

The system uses **CLIP (Contrastive Language–Image Pretraining)** to encode both image and text into the **same embedding space**, enabling direct comparison between them.

---

## 🧠 Model Used

Pretrained Vision–Language Model:

```
openai/clip-vit-base-patch32
```

CLIP learns a shared representation for **images and text**, allowing similarity comparison between visual content and textual descriptions.

---

## 📂 Project Structure

```
Image_Caption_Similarity_Analyzer
│
├── Image Caption Similarity Analyzer
│   │
│   ├── App
│   │   └── app_fixed.py          # Streamlit UI for image and caption comparison
│   │
│   ├── Model
│   │   └── clip_model.py         # CLIP encoder for generating embeddings
│   │
│   └── Utils
│       └── similarity.py         # cosine similarity computation
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

Clone the repository:

```
git clone https://github.com/Yash-Vardhan-Rajpoot/Multimodal_Misinformation_classifier.git
cd Multimodal_Misinformation_classifier
```

Install dependencies:

```
pip install -r requirements.txt
```

---

## ▶️ Run the Application

```
streamlit run "Multimodal Misinformation Classifier/App/app_fixed.py"
```

The application will start locally and open in your browser.

---

## 🖥️ How to Use

1. Upload an image
2. Enter multiple captions describing the image
3. The model computes similarity between the image and each caption
4. Captions are ranked based on similarity score
5. The caption with the **highest similarity score is selected as the best match**

---

## 📊 Similarity Computation

The similarity between image and caption embeddings is calculated using **cosine similarity**.

Higher similarity score → **better caption match**

---

## 🛠️ Tech Stack

* Python
* PyTorch
* Transformers
* CLIP Model
* Streamlit
* NumPy
* Pillow

---

## 🌐 Deployment

The project is deployed using **Streamlit** and accessible here:

https://multimodalmisinformationclassifier.streamlit.app/

---

## 📌 Future Improvements

* Automatic **caption generation**
* **Top-k caption ranking**
* Support for **image–text search**
* Integration with **large multimodal models**
* Improve UI with **caption confidence visualization**

---

## 👨‍💻 Author

**Yash Vardhan Rajpoot**  
NIT Patna  

GitHub:  
https://github.com/Yash-Vardhan-Rajpoot

---

## ⭐ Support

If you found this project useful, consider **starring the repository** on GitHub.
