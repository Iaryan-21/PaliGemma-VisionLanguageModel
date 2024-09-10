from email.mime import image
from sys import implementation
from typing import Dict, List, Optional, Union, Tuple, Iterable
import numpy as np 
from PIL import Image
import torch

IMAGENET_STANDARD_MEAN = [0.5,0.5,0.5]
IMAGENET_STANDARD_STD = [0.5,0.5,0.5]

class PaliGemmaProcessor:
    IMAGE_TOKEN = "<image>" # Constant placeholder tokens that will be replaced by Image EMbedding Tokens.
    def __init__(self, tokenizer, num_image_tokens: int, image_size: int):
        super().__init__()
        
        self.image_seq_length = num_image_tokens
        self.image_size = image_size
        
        tokens_to_add = {"additional_special_tokens": [self.IMAGE_TOKEN]}
        tokenizer.add_special_tokens(tokens_to_add)
        EXTRA_TOKENS = [
            f"<loc{i:04d}>" for i in range(1024) # Adding extra tokens for bounding box detection
        ]
        EXTRA_TOKENS += [
            f"<seg{i:03d}>" for i in range(128) # Adding extra tokens for segmentation
        ]
        tokenizer.add_tokens(EXTRA_TOKENS)
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.IMAGE_TOKEN)
        tokenizer.add_bos_token = False
        tokenizer.add_eos_token = False
        
        self.tokenizer = tokenizer
    
    def __call__(self, text: List[str], images: List[Image.Image], padding: str = "longest", truncation: bool = True) -> dict:
        assert len(images) == 1 and len(text) == 1, f"Recieved {len(images)} images for {len(text)} prompts."
    
        pixel_values = process_image(
            images,
            size=(self.image_size, self.image_size),
            resample = Image.Resampling.BICUBIC,
            resacle_factor = 1/255.0,
            image_mean = IMAGENET_STANDARD_MEAN,
            image_std = IMAGENET_STANDARD_STD,    
        )
        pixel_values = np.stack(pixel_values, axis=0) # [Batch_size, Channel, Height, Width]
        pixel_values = torch.tensor(pixel_values) # COnvert Numpy Array to a PyTorch Tensor.
        
        input_strings = [
            add_image_tokens_to_prompt(
                prefix_prompt = prompt, 
                bos_token = self.tokenizer.bos_token,
                image_seq_len = self.image_seq_length,
                image_token = self.IMAGE_TOKEN,
            )
            for prompt in text 
        ]
