import shutil
from data.build import build_transform
from model.build import Swin4MRI
import torch
import os
from typing import List
from PIL import Image
from scripts import main_dir

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
checkpoint_dir = os.path.join(str(main_dir), 'logs/MRI_model/version_1/checkpoints')


class InferenceModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.transform = build_transform(is_train=False)
        self.checkpoint_dir = os.path.join(main_dir, "scripts", "checkpoint")
        self.load_checkpoint()
        self.model = Swin4MRI.load_from_checkpoint(self.checkpoint_dir)
        self.idx_to_labels: dict = {0: "glioma", 1: "healthy", 2: "meningioma", 3: "pituitary"}

    def forward(self, data):
        return self.model(data)

    def load_checkpoint(self):

        os.makedirs(self.checkpoint_dir, exist_ok=True)

        file_name = [f for f in os.listdir(checkpoint_dir) if f.endswith('.ckpt')]

        shutil.copy(os.path.join(checkpoint_dir, file_name[0]), self.checkpoint_dir)
        self.checkpoint_dir = os.path.join(checkpoint_dir, file_name[0])

    def process_image(self, image):
        return self.transform(image)

    def predict(self, batch, is_batched: bool = True):
        with torch.inference_mode():
            if is_batched:
                batch = self.batch_images(batch)

            else:
                batch = self.transform(batch)
                batch = batch.unsqueeze(0)

            outputs = self.model(batch)

        return outputs

    def encode_response(self, logits):
        probabilities = torch.softmax(logits, dim=1)
        max_probabilities, indices = torch.max(probabilities, dim=1)
        labels = [self.idx_to_labels[label] for label in indices.cpu().tolist()]
        result = {
            "predictions": [
                        {
                            "probability": f"{prob:.4f}",
                            "label": label
                        }
                        for prob, label in zip(max_probabilities.cpu().tolist(), labels)

                            ]

                }
        return result

    def batch_images(self, images: List[Image.Image]) -> torch.Tensor:

        processed_images = [self.transform(image) for image in images]
        return torch.stack(processed_images)
