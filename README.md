提示词翻译，使用qwen2.5 1.5b模型进行提示词翻译，也可以进行扩句，中文效果很不错的。模型第一次会从huggingface，下载到comfyui模型目录下的LLM目录。同时提供了dolphin-2.9.4-gemma2-2b模型，说实话，中文支持非常烂。
![image](https://github.com/user-attachments/assets/dcb48d31-7fd6-4b71-9d09-c1c54a637d54)

反推提示词：使用Qwen2-VL-2B，SmolVLM两个模型实现提示词反推，在小参数模型中效果非常好。8G 显存，反推6秒左右，SDXL跑图在20秒左右。模型第一次会从huggingface，下载到comfyui模型目录下的LLM目录。
![image](https://github.com/user-attachments/assets/3e23e8af-0825-4903-82e9-41151265281e)

使用SmolVLM模型需要transformers>=4.46.3，当transformers==4.45时会报错。
