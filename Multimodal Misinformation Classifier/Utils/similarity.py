import torch
import torch.nn.functional as F

def compute_similarity(image_embedding, text_embedding):

    image_embedding = F.normalize(image_embedding, p=2, dim=1)
    text_embedding = F.normalize(text_embedding, p=2, dim=1)

    similarity = torch.mm(image_embedding, text_embedding.T)

    return similarity.item()
