from huggingface_hub import snapshot_download
from PIL import Image
import numpy as np
import folder_paths
import os
def load_model_from_hug(model):
    model_name = model.rsplit('/', 1)[-1]
    model_path = os.path.join(folder_paths.models_dir, "LLM", model_name)        
    if not os.path.exists(model_path):
        print(f"Downloading model to: {model_path}")            
        snapshot_download(repo_id=model,
                        local_dir=model_path,
                        local_dir_use_symlinks=False)
    return model_path

# Tensor to PIL
def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def resize_image(image, max_size=768):
    """
    调整图像大小，使得长边不超过max_size像素。
    """
    width, height = image.size
    if width > height:
        new_width = max_size
        new_height = int(height * (new_width / width))
    elif height > width:
        new_height = max_size
        new_width = int(width * (new_height / height))
    else:
        new_width = max_size
        new_height = max_size
    return image.resize((new_width, new_height), Image.LANCZOS)