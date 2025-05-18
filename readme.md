# Image Upscaling API

A high-performance FastAPI application for image enhancement and upscaling using RealESRGAN models.

![Sample Comparison](./results/comparison.jpg)

## Overview

This project provides a REST API for image upscaling and enhancement using state-of-the-art deep learning models. It supports multiple scaling factors (1x, 2x, 4x, and 8x) and handles various input image formats.

## Features

- **Multiple Upscaling Options**: Scale images by 1x (clarity enhancement), 2x, 4x, and 8x
- **Format Conversion**: Automatically converts various image formats to optimized JPG
- **High-Quality Output**: Produces high-quality enhanced images with preserved details
- **GPU Acceleration**: Utilizes CUDA when available for faster processing
- **RESTful API**: Simple HTTP interface for easy integration

## Technology Stack

- **FastAPI**: High-performance web framework
- **PyTorch**: Deep learning framework
- **RealESRGAN**: State-of-the-art image upscaling models
- **OpenCV**: Image processing operations
- **PIL/Pillow**: Image manipulation

## Installation

### Prerequisites

- Python 3.7+
- CUDA-compatible GPU (optional, for faster processing)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/image-upscaling-api.git
   cd image-upscaling-api
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Download model weights:
   ```bash
   mkdir -p weights
   # Download RealESRGAN weights to the weights directory
   # RealESRGAN_x2.pth, RealESRGAN_x4.pth, RealESRGAN_x8.pth
   ```

## Usage

### Starting the Server

```bash
cd backend
python main.py
```

The API will be available at `http://localhost:8000`.

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/rescaling_1x` | POST | Enhance image clarity without upscaling |
| `/rescaling_2x` | POST | Upscale image by 2x |
| `/rescaling_4` | POST | Upscale image by 4x |
| `/rescaling_8x` | POST | Upscale image by 8x |

### Example Request

Using cURL:

```bash
curl -X POST -F "image=@./inputs/sample_low_res.jpg" http://localhost:8000/rescaling_4 --output enhanced_image.jpg
```

Using Python requests:

```python
import requests

url = "http://localhost:8000/rescaling_4"
files = {"image": open("./inputs/sample_low_res.jpg", "rb")}
response = requests.post(url, files=files)

with open("enhanced_image.jpg", "wb") as f:
    f.write(response.content)
```

## Sample Results

| Input | 2x Upscaling | 4x Upscaling |
|-------|--------------|--------------|
| ![Input Image](./inputs/sample1.jpg) | ![2x Upscaled](./results/sample1_2x.jpg) | ![4x Upscaled](./results/sample1_4x.jpg) |
| ![Input Image](./inputs/sample2.jpg) | ![2x Upscaled](./results/sample2_2x.jpg) | ![4x Upscaled](./results/sample2_4x.jpg) |

## Technical Details

### Image Processing Pipeline

1. **Input Validation**: Verify the uploaded file is a supported image format
2. **Format Conversion**: Convert the input to a standardized JPG format
3. **Model Selection**: Choose the appropriate RealESRGAN model based on the requested scale
4. **Inference**: Process the image through the neural network
5. **Response**: Return the enhanced image as a JPEG

### Models

The application uses RealESRGAN models for super-resolution:

- **1x**: Custom OpenCV-based sharpening filter
- **2x**: RealESRGAN_x2.pth
- **4x**: RealESRGAN_x4.pth
- **8x**: RealESRGAN_x8.pth

## Project Structure

```
.
├── backend/
│   ├── main.py                # FastAPI application and endpoints
│   ├── convert_to_jpg.py      # Image format conversion utilities
│   └── RealESRGAN/            # RealESRGAN model implementation
├── weights/                   # Model weights directory
│   ├── RealESRGAN_x2.pth
│   ├── RealESRGAN_x4.pth
│   └── RealESRGAN_x8.pth
├── inputs/                    # Sample input images
├── results/                   # Sample result images
└── README.md                  # This documentation
```

## Known Issues

- The 8x upscaling model is loaded with scale=2 in the code, which may need correction
- Processing large images may require significant memory

## License

[MIT License](LICENSE)

## Acknowledgements

- [RealESRGAN](https://github.com/xinntao/Real-ESRGAN) for the super-resolution models
- FastAPI for the web framework
