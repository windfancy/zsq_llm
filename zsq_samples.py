import sys
import torch
import folder_paths
import latent_preview
import numpy as np
import comfy.utils, comfy.sample, comfy.samplers, comfy.controlnet, comfy.model_base, comfy.model_management, comfy.sampler_helpers, comfy.supported_models

BASE_RESOLUTIONS = [
    ("width", "height"),
    (512, 512),
    (512, 768),
    (576, 1024),
    (768, 512),
    (768, 768),
    (768, 1024),
    (768, 1280),
    (768, 1344),
    (768, 1536),
    (816, 1920),
    (832, 1152),
    (832, 1216),
    (896, 1152),
    (896, 1088),
    (1024, 1024),
    (1024, 576),
    (1024, 768),
    (1080, 1920),
    (1440, 2560),
    (1088, 896),
    (1216, 832),
    (1152, 832),
    (1152, 896),
    (1280, 768),
    (1344, 768),
    (1536, 640),
    (1536, 768),
    (1920, 816),
    (1920, 1080),
    (2560, 1440)
]


resolution_strings = [f"{width} x {height} (custom)" if width == 'width' and height == 'height' else f"{width} x {height}" for width, height in BASE_RESOLUTIONS]
class zsqLoader_sampler:
    def __init__(self):
        self.device = comfy.model_management.intermediate_device()
        self.model = None
        self.model_name = None
    @classmethod
    def INPUT_TYPES(cls):

        return {"required": {
            "ckpt_name": (folder_paths.get_filename_list("checkpoints"),),
            "config_name": (["Default", ] + folder_paths.get_filename_list("configs"), {"default": "Default"}),
            "resolution": (resolution_strings,),          
            "positive_text": ("STRING", {"default": "", "placeholder": "Positive", "multiline": True}),
            "negative_text": ("STRING", {"default": "", "placeholder": "Negative", "multiline": True}),
            "batch_size": ("INT", {"default": 1, "step": 1, "min": 1, "max": 1024, "display": "slider"}),

            "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "tooltip": "The random seed used for creating the noise."}),
            "cfg": ("FLOAT", {"default": 8, "min": 0.0, "max": 100.0}),
            "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
            "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
            "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
},

        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)

    FUNCTION = "run"
    CATEGORY = "zsq/Loaders"

    def run(self, ckpt_name, config_name, resolution,positive_text,negative_text,batch_size, 
            seed,steps, cfg, sampler_name, scheduler, denoise):

        image =None
        # Clean models from loaded_objects
        if self.model_name != ckpt_name:
            self.model = None
            self.model_name = ckpt_name
        # Load models
        if self.model is None:
            print("Loading models...")
            self.model, clip, vae = self.load_checkpoint(config_name,ckpt_name)
        # Empty Latent width, height
        width, height = resolution.split(" x ")

        # Prompt to Conditioning
        positive = self.prompt_to_conditioning(clip, positive_text)
        negative = self.prompt_to_conditioning(clip, negative_text)
        image_latent = self.common_ksampler(self.model, seed, steps, cfg, sampler_name, scheduler, 
                                            positive, negative, denoise,width, height, batch_size)
        #image = self.decode(vae, image_latent)

        return image_latent
    def prompt_to_conditioning(self, clip, text):
        tokens = clip.tokenize(text)
        output = clip.encode_from_tokens(tokens, return_pooled=True, return_dict=True)
        cond = output.pop("cond")
        return ([[cond, output]])
    
    def decode(self, vae, samples):
        images = vae.decode(samples["samples"])
        if len(images.shape) == 5: #Combine batches
            images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
        return images
    def load_checkpoint(self, config_name, ckpt_name):
        config_path = folder_paths.get_full_path("configs", config_name)
        ckpt_path = folder_paths.get_full_path_or_raise("checkpoints", ckpt_name)
        return comfy.sd.load_checkpoint(config_path, ckpt_path, output_vae=True, output_clip=True, embedding_directory=folder_paths.get_folder_paths("embeddings"))

    def empty_latent(self, width=512, height=512, batch_size=1):
        latent = torch.zeros([batch_size, 4, int(height) // 8, int(width) // 8], device=self.device)
        return {"samples":latent}
    
    def common_ksampler(self,model, seed, steps, cfg, sampler_name, scheduler, positive, negative, 
                        denoise=1.0,width=512, height=512, batch_size=1,
                        disable_noise=False, start_step=None, last_step=None, force_full_denoise=False):
        input_latent = torch.zeros([batch_size, 4, int(height) // 8, int(width) // 8], device=self.device)
        latent_image = comfy.sample.fix_empty_latent_channels(model, input_latent)

        if disable_noise:
            noise = torch.zeros(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout, device="cpu")
        else:
            #batch_inds = input_latent["batch_index"] if "batch_index" in input_latent else None
            noise = comfy.sample.prepare_noise(latent_image, seed)

        callback = latent_preview.prepare_callback(model, steps)
        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

        callback = latent_preview.prepare_callback(model, steps)
        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
        samples = comfy.sample.sample(model, noise, steps, cfg, sampler_name, scheduler, positive, negative, latent_image,
                                    denoise=denoise, disable_noise=disable_noise, start_step=start_step, last_step=last_step,
                                    force_full_denoise=force_full_denoise, callback=callback, disable_pbar=disable_pbar, seed=seed)

        return ({"samples":samples})
    
NODE_CLASS_MAPPINGS = {
    "zsqLoader_sampler": zsqLoader_sampler
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "zsqLoader_sampler": "zsq Loader sampler"
}