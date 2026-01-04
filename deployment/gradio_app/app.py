"""Gradio web demo for CrossMal-ViT."""

from pathlib import Path
import sys

import gradio as gr
import torch
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from crossmal_vit.models import build_crossmal_vit
from crossmal_vit.data.transforms import MultiViewTransform


CLASS_NAMES = [
    "Allaple.A",
    "Allaple.L",
    "Yuner.A",
    "Lolyda.AA1",
    "Lolyda.AA2",
    "Lolyda.AA3",
    "C2Lop.P",
    "C2Lop.gen!g",
    "Instantaccess",
    "Swizzor.gen!E",
    "Swizzor.gen!I",
    "VB.AT",
    "Fakerean",
    "Alueron.gen!J",
    "Malex.gen!J",
    "Autorun.K",
    "Rbot!gen",
    "Benign",
]


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None
transform = None


def load_model() -> None:
    global model, transform

    config = {"img_size": 224, "num_classes": 18, "pretrained": False}
    model = build_crossmal_vit(config)

    checkpoint_path = Path(__file__).parent / "crossmal_vit_best.pth"
    if checkpoint_path.exists():
        state_dict = torch.load(checkpoint_path, map_location=device)
        if "model" in state_dict:
            model.load_state_dict(state_dict["model"], strict=False)
        elif "state_dict" in state_dict:
            model.load_state_dict(state_dict["state_dict"], strict=False)
        else:
            model.load_state_dict(state_dict, strict=False)

    model = model.to(device)
    model.eval()

    transform = MultiViewTransform(img_size=224)


def predict(image):
    global model, transform

    if model is None:
        load_model()

    if isinstance(image, np.ndarray):
        image = Image.fromarray(image).convert("L")
    else:
        image = image.convert("L")

    views = transform(image)
    views = {k: v.unsqueeze(0).to(device) for k, v in views.items()}

    with torch.no_grad():
        outputs = model(views)
        probs = torch.softmax(outputs["logits"], dim=-1)[0]

    results = {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}
    results = dict(sorted(results.items(), key=lambda x: x[1], reverse=True))

    return results


demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload Malware Byteplot"),
    outputs=gr.Label(num_top_classes=5, label="Predictions"),
    title="CrossMal-ViT: Malware Classification",
    description=(
        "Upload a grayscale malware byteplot image for classification. "
        "CrossMal-ViT uses multi-view representation learning with Vision Transformers."
    ),
)


if __name__ == "__main__":
    demo.launch(share=True)
