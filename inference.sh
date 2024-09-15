#!/bin/bash
MODEL_PATH="paligemma-3b-pt-224"
PROMPT="Tell me about this image  "
IMAGE_FILE_PATH="/home/amishr17/Desktop/PaliGema_VLM/test_image/test_2.jpeg"
MAX_TOKENS_TO_GENERATE=200
TEMPERATURE=0.5
TOP_P=0.9
DO_SAMPLE="True"
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
