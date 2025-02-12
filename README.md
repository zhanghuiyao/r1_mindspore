
r1_mindspore

# /Users/zhanghuiyao/Desktop/_R1/r1_mindspore/hf_configs/Qwen/Qwen2.5-1.5B-Instruct

pip install git+https://github.com/mindspore-lab/mindone.git

python train_r1_zero.py \
  --model-path /Users/zhanghuiyao/Desktop/_R1/r1_mindspore/hf_configs/nyu-visionx-cambrian-8b \
  --dataset-path Jiayi-Pan/Countdown-Tasks-3to4 \
  --max-completion-length 3 \
  --bf16 \
  --is-distribute False
