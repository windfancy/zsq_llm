from .zsq_llm import LLMText,LLMImage
from .zsq_tool import Loader_sampler

NODE_CLASS_MAPPINGS = {
    "LLMText": LLMText,
    "LLMImage": LLMImage,
    "Loader_sampler": Loader_sampler
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LLMText": "LLM Text",
    "LLMImage": "LLM Image",
    "Loader_sampler": "Simple Loader&sampler"
}
