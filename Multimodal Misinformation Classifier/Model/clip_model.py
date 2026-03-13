import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

class CLIPEncoder:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def encode_image(self, image):
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
        # Handle both raw tensor and BaseModelOutputWithPooling object
        if hasattr(image_features, 'image_embeds'):
            image_features = image_features.image_embeds
        elif hasattr(image_features, 'pooler_output'):
            image_features = image_features.pooler_output
        return image_features.cpu()

    def encode_text(self, text):
        inputs = self.processor(text=[text], return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
        # Handle both raw tensor and BaseModelOutputWithPooling object
        if hasattr(text_features, 'text_embeds'):
            text_features = text_features.text_embeds
        elif hasattr(text_features, 'pooler_output'):
            text_features = text_features.pooler_output
        return text_features.cpu()
