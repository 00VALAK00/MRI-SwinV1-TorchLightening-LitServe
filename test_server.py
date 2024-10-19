import json
import requests
from PIL import Image
import os
import random
import io

data_dir = os.path.join(os.getcwd(), "data_folder", "brain-tumor-mri-scans")


def hex_encode(image):
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='JPEG')
    img_bytes = img_byte_arr.getvalue()

    # Hex encode the byte array
    img_hex = img_bytes.hex()

    return img_hex


def prepare_images(num_images):
    images = []
    # Chose one of the 4 classes
    for i in range(num_images):
        random_class = random.choice(os.listdir(data_dir))
        random_image_name = random.choice(os.listdir(os.path.join(data_dir, random_class)))
        random_img = Image.open(os.path.join(data_dir, random_class, random_image_name))

        images.append(hex_encode(random_img))

    return images


def to_json(list_of_images):
    return json.dumps({"image_bytes": list_of_images})


def main():
    data = prepare_images(num_images=4)
    json_data = to_json(data)

    url = "http://localhost:8000/predict"
    headers = {'content-type': 'application/json'}
    response = requests.post(url, data=json_data, headers=headers)

    print(response.json())

if __name__ == '__main__':
    main()

