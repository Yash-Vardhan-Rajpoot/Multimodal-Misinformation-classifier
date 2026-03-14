import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Model.clip_model import CLIPEncoder
from Utils.similarity import compute_similarity

model = CLIPEncoder()

image_embedding = model.encode_image("Data/Images/sharukh_khan.jpg")
text_embedding = model.encode_text("this image is of sharukh khan")

score = compute_similarity(image_embedding, text_embedding)

print("Similarity Score:", score)

if score > 0.3:
    print("Result: MATCH")
else:
    print("Result: MISINFORMATION")
