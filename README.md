# PaliGemma Vision-Language Model Repository

## Overview

This repository contains the implementation of a Vision-Language Model (VLM) based on PaliGemma 3B architecture. The model integrates Rotary Positional Encoding (RoPE), KV Cache, and Grouped Query Key Attention, combined with a Vision Transformer (ViT) for image encoding. It is designed for various vision-language tasks such as image captioning, visual question answering (VQA), and multimodal reasoning.

## Key Features:

Vision Transformer-based Image Encoder
Gemma-2B Language Model
Rotary Positional Encoding (RoPE)
Grouped Query Key Attention
KV Cache for efficient attention computations

## Reference Paper:
The implementation is based on the architecture and methodology presented in PaliGemma: A Versatile 3B Vision-Language Model by Lucas Beyer et al. You can find the model’s detailed architecture and benchmarks in the reference paper.

## Model Architecture

The architecture consists of two primary components:

Vision Encoder: A Vision Transformer (ViT) model is used to encode images into feature tokens.
Language Decoder: The Gemma-2B language model is used to autoregressively generate text from the visual and textual input.
The input consists of images and/or text descriptions. The model produces text outputs such as captions, answers to questions, or descriptions of visual content.

## Installation
1.  Clone this repository
    ```
    git clone https://github.com/Iaryan-21/PaliGemma-VisionLanguageModel.git
    cd PaliGemma-VisionLanguageModel

    ```

## Running Inference 

You can use the provided inference.sh script to run inference with the PaliGemma model. Below is an example of running the model with a prompt and image:

```
#!/bin/bash
MODEL_PATH="paligemma-3b-pt-224"
PROMPT="What color is the road"
IMAGE_FILE_PATH="/home/amishr17/Desktop/PaliGema_VLM/test_image/test_2.jpeg"
MAX_TOKENS_TO_GENERATE=500
TEMPERATURE=0.2
TOP_P=0.92
DO_SAMPLE="False"
ONLY_CPU="True"

python inference.py \
 --model_path "$MODEL_PATH" \
 --prompt "$PROMPT" \
 --image_file_path "$IMAGE_FILE_PATH" \
 --max_tokens_to_generate $MAX_TOKENS_TO_GENERATE \
 --temperature $TEMPERATURE \
 --top_p $TOP_P \
 --do_sample $DO_SAMPLE \
 --only_cpu $ONLY_CPU

```

## Running the Inference Script:
To run the model and generate a response based on your prompt and image, execute:

```
bash inference.sh
```

## Pretrained Weights 

The model achieves strong performance on several benchmarks, including:

COCO Captioning
Visual Question Answering (VQA)
InfographicVQA
Remote-Sensing VQA

## Citation 

```
@article{beyer2024paligemma,
  title={PaliGemma: A Versatile 3B Vision-Language Model},
  author={Lucas Beyer and others},
  journal={arXiv preprint arXiv:2407.07726},
  year={2024}
}

```
