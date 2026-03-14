import torch
import torch.nn.functional as F


def compute_similarity(image_embedding, text_embedding):

    # Normalize embeddings
    image_embedding = F.normalize(image_embedding, dim=-1)
    text_embedding = F.normalize(text_embedding, dim=-1)

    # Cosine similarity
    similarity = torch.cosine_similarity(image_embedding, text_embedding)

    return similarity.item()