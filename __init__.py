from .zsq_llm_text import LLMText
from .zsq_llm_image import LLMImage

NODE_CLASS_MAPPINGS = {
    "LLMQwenText": LLMText,
    "LLMImage": LLMImage
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LLMText": "LLM Text",
    "LLMImage": "LLM Image"
}








