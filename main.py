import os
import torch
from PIL import Image
import numpy as np
from RealESRGAN import RealESRGAN
from convert_to_jpg import convert_images_to_jpg
from fastapi import FastAPI, File, UploadFile, Response
import uvicorn
import tempfile
import io
import cv2

main = FastAPI()


def image_clearity(image_path):
    image = cv2.imread(image_path)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    sharpened = cv2.filter2D(image, -1, kernel)
    return sharpened


@main.post("/rescaling_1x")
async def rescaling_1x(image: UploadFile = File(...)):
    try:
        # Convert uploaded image to jpg in a temporary file
        temp_jpg_path = convert_images_to_jpg(image.file)
        if not temp_jpg_path:
            return {"error": "Failed to process the uploaded image"}

        sr_image = image_clearity(temp_jpg_path)
        # Convert NumPy array to bytes using OpenCV
        _, img_encoded = cv2.imencode('.jpg', sr_image,
                                      [cv2.IMWRITE_JPEG_QUALITY, 95])
        img_byte_arr = io.BytesIO(img_encoded.tobytes())
        img_byte_arr.seek(0)
        os.remove(temp_jpg_path)
        return Response(content=img_byte_arr.getvalue(),
                        media_type="image/jpeg")
    except Exception as e:
        return {"error": str(e)}


@main.post("/rescaling_2x")
async def rescaling_2x(image: UploadFile = File(...)):
    try:
        # Convert uploaded image to jpg in a temporary file
        temp_jpg_path = convert_images_to_jpg(image.file)
        if not temp_jpg_path:
            return {"error": "Failed to process the uploaded image"}

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = RealESRGAN(device, scale=2)
        model.load_weights('weights/RealESRGAN_x2.pth')
        sr_image = model.predict(Image.open(temp_jpg_path))
        # Convert PIL Image to bytes
        img_byte_arr = io.BytesIO()
        sr_image.save(img_byte_arr, format='JPEG', quality=95)
        img_byte_arr.seek(0)
        os.remove(temp_jpg_path)
        return Response(content=img_byte_arr.getvalue(),
                        media_type="image/jpeg")
    except Exception as e:
        return {"error": str(e)}


@main.post("/rescaling_4")
async def rescaling_4x(image: UploadFile = File(...)):
    try:
        # Convert uploaded image to jpg in a temporary file
        temp_jpg_path = convert_images_to_jpg(image.file)
        if not temp_jpg_path:
            return {"error": "Failed to process the uploaded image"}

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = RealESRGAN(device, scale=4)
        model.load_weights('weights/RealESRGAN_x4.pth')
        sr_image = model.predict(Image.open(temp_jpg_path))
        # Convert PIL Image to bytes
        img_byte_arr = io.BytesIO()
        sr_image.save(img_byte_arr, format='JPEG', quality=95)
        img_byte_arr.seek(0)
        os.remove(temp_jpg_path)
        return Response(content=img_byte_arr.getvalue(),
                        media_type="image/jpeg")
    except Exception as e:
        return {"error": str(e)}


@main.post("/rescaling_8x")
async def rescaling_8x(image: UploadFile = File(...)):
    try:
        # Convert uploaded image to jpg in a temporary file
        temp_jpg_path = convert_images_to_jpg(image.file)
        if not temp_jpg_path:
            return {"error": "Failed to process the uploaded image"}

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = RealESRGAN(device, scale=2)
        model.load_weights('weights/RealESRGAN_x8.pth')
        sr_image = model.predict(Image.open(temp_jpg_path))
        # Convert PIL Image to bytes
        img_byte_arr = io.BytesIO()
        sr_image.save(img_byte_arr, format='JPEG', quality=95)
        img_byte_arr.seek(0)
        os.remove(temp_jpg_path)
        return Response(content=img_byte_arr.getvalue(),
                        media_type="image/jpeg")
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    uvicorn.run(main, host="0.0.0.0", port=8000)
