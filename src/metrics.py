import torch
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel

class ClipScorer:
  """
    This Utility class is used for scoring image–text similarity using a pretrained CLIP model.

    It loads the HuggingFace CLIP model (openai/clip-vit-base-patch32) and its
    corresponding processors, then provides a simple method to compute cosine similarity
    between an input image and a text generated.

    Typical use cases:
        - Evaluate how well a generated image matches a predicted/ground-truth caption
        - Monitor alignment during training with validation metric
        - Compare multiple generated samples and select the best one
    Attributes:
        device (str):
            The device where CLIP runs on either "cuda" or "cpu".
        model (CLIPModel):
            Pretrained CLIP model loaded and set for evaluation mode.
        proc (CLIPProcessor):
            CLIP processor tokenizes the text and preprocesses the images into tensors.
    """
    def __init__(self, device="cuda"):
        self.device = device
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        self.proc  = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.model.eval()

    @torch.no_grad()
    def image_text_cosine(self, pil_image, text: str) -> float:
      """
        It compute CLIP cosine similarity between a PIL image and a text string.
        This method extracts CLIP embeddings for the image and text, normalizes them,
        and returns their cosine similarity (dot product of normalized vectors).
        Args:
            pil_image:
                Input image in PIL.Image format (RGB recommended).
            text (str):
                Text caption to compare against the image.
        Returns:
            float:
                Cosine similarity score between the CLIP image embedding and CLIP text embedding.
                Higher values indicate stronger semantic alignment.
                Typical range is approximately [-1, 1], but most practical values fall in [0, 1].
        Notes:
            - Uses `torch.no_grad()` to avoid building gradients (safe for evaluation).
            - If your captions are very long, you may want to add truncation:
              `self.proc(__, truncation=True, max_length=77)` for CLIP text limit.
        """
        inp = self.proc(images=pil_image, text=[text], return_tensors="pt", padding=True).to(self.device)
        out = self.model(**inp)
        img = out.image_embeds
        txt = out.text_embeds
        img = F.normalize(img, dim=-1)
        txt = F.normalize(txt, dim=-1)
        return float((img * txt).sum(dim=-1).item())
