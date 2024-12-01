from transformers import AutoProcessor, AutoModelForVision2Seq
from .llm_libs import load_model_from_hug, tensor2pil, resize_image
import torch

def load_model(model_name, device):
    dtype = torch.float16
    model_path = load_model_from_hug(model_name)
    model = AutoModelForVision2Seq.from_pretrained(model_path, 
                                             device_map=device, 
                                             torch_dtype=dtype,
                                             trust_remote_code=True).to(0)
    processor = AutoProcessor.from_pretrained(model_path, 
                                          trust_remote_code=True)
    return model, processor

def generate_content(model, processor, image, device,max_tokens=128):
    messages = [
        {
            "role": "user",
            "content": [
                    {
                    "type": "image",
                    "image": image,  
                    },
                    {"type": "text", 
                     "text": "Can you describe the image?"
                     },
                ],
            }
        ]

    # Preparation for inference
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    if prompt is None:
        raise ValueError("Prompt is None, check the messages format or processor configuration.")
    inputs = processor(
          text=prompt, 
          images=image, 
          return_tensors="pt").to(device)
    generated_ids = model.generate(**inputs, max_new_tokens=max_tokens)
    output_text = processor.batch_decode(
        generated_ids, 
        skip_special_tokens=True, 
        clean_up_tokenization_spaces=False
    )[0]
    return output_text

def replace_image_text(image_text):
    image_text = image_text.replace("User:<image>","")
    image_text = image_text.replace("Assistant:","")
    image_text = image_text.replace("system","")
    image_text = image_text.replace("You are a helpful assistant.","") 
    image_text = image_text.replace("user","")   
    image_text = image_text.replace("Can you describe the image?","") 
    image_text = image_text.replace("assistant","")
    image_text = image_text.replace('\n',"")
    return image_text

class LLMImage:
    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(self):
        self.model = None
        self.processor = None
        self.modelname = None
        return {
            "required": {                
                "Input_Image": ("IMAGE",),
                "max_tokens": ("INT", {"default": 128, "step": 1, "min": 1, "max": 1024, "display": "slider"}),                   
                "modelname": (['Qwen/Qwen2-VL-2B-Instruct','HuggingFaceTB/SmolVLM-Instruct'], {"default": 'Qwen/Qwen2-VL-2B-Instruct'}),
                "device": (['cuda', 'cpu'], {"default": 'cuda'}),
            }
        }
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("image2text",)
    FUNCTION = "run"
    CATEGORY = "ZSQ/LLM"
    
    def run(self, modelname, device, max_tokens,Input_Image=None): 
        if self.model is None or self.modelname != modelname:
            self.unload_model()  # 卸载之前的模型
            print("Loading model:" + modelname)       
            self.model, self.processor = load_model(modelname, device) 
            self.modelname = modelname
        image = Input_Image[0]            
        image = tensor2pil(image)
        image = image.convert('RGB')
        image = resize_image(image)
        prompt = generate_content(self.model, self.processor, image, device,max_tokens)
        prompt = replace_image_text(prompt)
        return(prompt,)    
    def unload_model(self):
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        torch.cuda.empty_cache()  # 释放未使用的 CUDA 缓存

