import torch
import numpy as np

from transformers import AutoTokenizer
from datasets import load_dataset


x_np = np.array([[1, 2], [3, 4], [5, 6]])
# x_np = np.array([1, 2, 3, 4])
x_pt = torch.from_numpy(x_np)

x_np_new = x_np.astype(np.int).argmax(axis=1)

# print(f"{x_np[None].repeat(2, 0)=}")
# print(f"{x_pt.expand(2, -1)=}")

# prompts = [1, 2, 3]
# x = [prompt for prompt in prompts for _ in range(3)]
#
# print(x)
#
#
# repo_id = "/Users/zhanghuiyao/Desktop/_R1/r1_mindspore/Qwen/Qwen2.5-3B-Instruct"
#
# tokenizer = AutoTokenizer.from_pretrained(repo_id, padding_side="left")
#
#
# def generate_r1_prompt(numbers, target):
#     r1_prefix = [{
#         "role": "system",
#         "content": "You are a helpful assistant. You first thinks about the reasoning process in the mind and then provides the user with the answer."
#     },
#         {
#             "role": "user",
#             "content": f"Using the numbers {numbers}, create an equation that equals {target}. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. Show your work in <think> </think> tags. And return the final equation and answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 = 1 </answer>."
#         },
#         {
#             "role": "assistant",
#             "content": "Let me solve this step by step.\n<think>"
#         }]
#     return {"prompt": tokenizer.apply_chat_template(r1_prefix, tokenize=False, continue_final_message=True),
#             "target": target}
#
#
# r1_prompt = generate_r1_prompt(
#     numbers=[63, 95, 96],
#     target=64
# )
#
# print(f"{r1_prompt=}")
#
#
# tokenized_inputs = tokenizer(
#     [r1_prompt["prompt"],],
#     return_tensors="np",
#     padding=True,
#     padding_side="left",
#     add_special_tokens=False
# )
#
# tokenized_inputs_2 = tokenizer(
#     [r1_prompt["prompt"],],
#     return_tensors="np",
#     padding="max_length",
#     truncation=True,
#     max_length=256,
#     padding_side="right",
#     add_special_tokens=False
# )
#
# print(f"{tokenized_inputs=}")


# from datasets import load_dataset
#
# ds = load_dataset("Jiayi-Pan/Countdown-Tasks-3to4")


import pdb;pdb.set_trace()
