
# /Users/zhanghuiyao/Desktop/_R1/r1_mindspore/hf_configs/Qwen/Qwen2.5-1.5B-Instruct

python train_r1_zero.py \
  --model-path /Users/zhanghuiyao/Desktop/_R1/r1_mindspore/hf_configs/nyu-visionx-cambrian-8b \
  --dataset-path Jiayi-Pan/Countdown-Tasks-3to4 \
  --bf16 \
  --is-distribute False
