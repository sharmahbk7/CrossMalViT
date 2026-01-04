"""TorchServe handler for CrossMal-ViT."""

import io
from pathlib import Path
from typing import Any, Dict, List

import torch
from PIL import Image
from ts.torch_handler.base_handler import BaseHandler
from omegaconf import OmegaConf

from crossmal_vit.data import MultiViewTransform
from crossmal_vit.models import build_crossmal_vit


class CrossMalViTHandler(BaseHandler):
    """Custom TorchServe handler for malware classification."""

    def initialize(self, ctx: Any) -> None:
        self.manifest = ctx.manifest
        properties = ctx.system_properties
        model_dir = Path(properties.get("model_dir", "."))

        config_path = model_dir / "config.yaml"
        if config_path.exists():
            config = OmegaConf.load(str(config_path))
            model_cfg = OmegaConf.to_container(config.model)
            self.img_size = int(config.data.img_size)
            self.entropy_window = int(config.data.entropy_window)
        else:
            model_cfg = {"img_size": 224, "num_classes": 18, "pretrained": False}
            self.img_size = 224
            self.entropy_window = 9

        self.model = build_crossmal_vit(model_cfg)

        model_file = self.manifest["model"]["serializedFile"]
        checkpoint = torch.load(model_dir / model_file, map_location="cpu")
        if "model" in checkpoint:
            self.model.load_state_dict(checkpoint["model"], strict=False)
        elif "state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["state_dict"], strict=False)
        else:
            self.model.load_state_dict(checkpoint, strict=False)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        self.transform = MultiViewTransform(img_size=self.img_size, entropy_window=self.entropy_window)

    def preprocess(self, data: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        images = []
        for record in data:
            if "data" in record:
                image_bytes = record["data"]
            else:
                image_bytes = record.get("body")
            image = Image.open(io.BytesIO(image_bytes)).convert("L")
            images.append(image)

        batch = []
        for image in images:
            views = self.transform(image)
            batch.append(views)

        stacked = {"raw": [], "entropy": [], "frequency": []}
        for views in batch:
            for key in stacked:
                stacked[key].append(views[key])

        for key in stacked:
            stacked[key] = torch.stack(stacked[key], dim=0).to(self.device)

        return stacked

    def inference(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        with torch.no_grad():
            outputs = self.model(inputs)
            probs = torch.softmax(outputs["logits"], dim=-1)
        return probs

    def postprocess(self, inference_output: torch.Tensor) -> List[List[float]]:
        return inference_output.cpu().tolist()
