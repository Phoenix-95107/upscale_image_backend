from PIL import Image
import os
import uuid
import tempfile

def convert_images_to_jpg(file_input):
    """
    Convert an image to JPG format with a UUID filename using a temporary file.
    
    Args:
        file_input: A file-like object from an upload
        
    Returns:
        Path to the temporary JPG file or None if conversion failed
    """
    supported_exts = ('.png', '.webp', '.bmp', '.tiff', '.jpeg', '.jpg')
    
    try:
        # Check if the uploaded file has a valid extension
        if hasattr(file_input, 'filename'):
            original_filename = file_input.filename
            if not any(original_filename.lower().endswith(ext) for ext in supported_exts):
                print(f"Unsupported file format: {original_filename}")
                return None
        
        # Open the image
        img = Image.open(file_input)
        
        # Convert to RGB (necessary for JPG, which doesn't support alpha)
        img = img.convert("RGB")
        
        # Generate a unique filename with UUID
        unique_id = str(uuid.uuid4())
        temp_dir = tempfile.gettempdir()
        temp_filename = os.path.join(temp_dir, f"{unique_id}.jpg")
        
        # Save as high-quality JPEG
        img.save(temp_filename, "JPEG", quality=95, optimize=True, progressive=True)
        print(f"Converted to temporary file: {temp_filename}")
        return temp_filename
    except Exception as e:
        print(f"Failed to convert image: {e}")
        return None
