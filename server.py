import logging
import numpy as np
import torch

from scripts.predict import InferenceModel
import litserve as ls
from io import BytesIO
from PIL import Image
import json
from litserve import loggers

logger = loggers.Logger


class MriAPI(ls.LitAPI):
    def setup(self, device):
        self.mri_model = InferenceModel()

    def decode_request(self, request):
        decoded = []
        data = request["image_bytes"]
        for hex_img in data:
            image_bytes = bytes.fromhex(hex_img)
            image = Image.open(BytesIO(image_bytes))
            decoded.append(image)
        return decoded


    def predict(self, inputs):
        my_device = self.device
        tensors = [self.mri_model.process_image(img).unsqueeze(0) for img in inputs]
        batched_data = torch.cat(tensors).to(my_device)

        assert isinstance(batched_data, torch.Tensor), "batched_data is not a torch.Tensor"
        assert batched_data.size() == torch.Size([4, 3, 224, 224])

        logits = self.mri_model(batched_data)
        return logits

    def encode_response(self, inputs: torch.Tensor):
        response=self.mri_model.encode_response(inputs)
        return json.dumps(response)



def main():
    api = MriAPI()
    server = ls.LitServer(api, accelerator='cuda')
    server.run(port=8000)


if __name__ == '__main__':
    main()
