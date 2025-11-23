import load_vla from OpenVLAOFTModel
from PIL import Image
import numpy as np


def test_forward_pass():
    model_id = "moojink/openvla-7b-oft-finetuned-libero-object"

    print(f"Loading model from {model_id}")
    model = OpenVLAOFTModel(model_id)

    print(f"Loading a test image (red color) ...")
    test_image = Image.new("RGB", (224, 224), color='red')

    # print(f"Forwarding test image through model ...")
    # output = model.forward(image)

    # print(f"Output: {output}")