from transformers import AutoImageProcessor, ViTMAEForPreTraining
from thesis.config import HF_MODEL_ID


def load_mae(model_id=HF_MODEL_ID):
    """Return the ViT-MAE model (ImageNet-1k weights) and its image processor."""
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = ViTMAEForPreTraining.from_pretrained(model_id)
    return model, processor
