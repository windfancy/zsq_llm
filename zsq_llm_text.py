from transformers import AutoModelForCausalLM, AutoTokenizer
from .llm_libs import load_model_from_hug
import torch
def load_model(model_name, device):
    model_path = load_model_from_hug(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        device_map=device, 
        torch_dtype="auto", 
        trust_remote_code=True
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    torch.cuda.empty_cache()
    return model, tokenizer

def generate_content(model, tokenizer, prompt, system_instruction, device,max_tokens=128):
    messages = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": prompt},
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=max_tokens
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]    
    return response


class LLMText:
    @classmethod
    def INPUT_TYPES(self):
        self.model = None
        self.tokenizer = None
        self.modelname = None
        return {
            "required": {                
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True, "default":"请输入中文提示词"}),
                "max_tokens": ("INT", {"default": 128, "step": 1, "min": 1, "max": 1024, "display": "slider"}),                   
                "system_role": (['Chinese-to-English translator','prompt word engineer'], {"default":"Chinese-to-English translator"}),
                "modelname": (['Qwen/Qwen2.5-1.5B-Instruct','cognitivecomputations/dolphin-2.9.4-gemma2-2b'], {"default":'Qwen/Qwen2.5-1.5B-Instruct'}),
                "device": (['cuda', 'cpu'], {"default": 'cuda'}),
            }
        }
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("outtext",)
    FUNCTION = "run"
    CATEGORY = "ZSQ/LLM"
    
    def run(self, modelname, device,prompt, system_role,max_tokens):
        if system_role == 'Chinese-to-English translator':
            system_instruction = 'You are a Chinese-to-English translator'
            prompt = 'Only translate the following from Chinese to English and output：' + prompt
        elif system_role == 'prompt word engineer':
            system_instruction = 'you are a stable-diffusion prompt word engineer'
            prompt = 'Conceive a picture, describe the details of the picture in just 5 phrases, do not have your subjective thoughts, output in English:'+ prompt
        else:
            system_instruction = system_instruction
        if self.model is None or self.modelname != modelname:
            self.unload_model()  # 卸载之前的模型        
            self.model, self.tokenizer = load_model(modelname, device) 
            self.modelname = modelname
        
        response = generate_content(self.model, self.tokenizer, prompt, system_instruction, device,max_tokens)
        response = response.replace("\n", "")
        response = response.strip()
        return(response,)
    def unload_model(self):
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        torch.cuda.empty_cache()  # 释放未使用的 CUDA 缓存
