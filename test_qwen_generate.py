import numpy as np
import argparse
import ast

import mindspore as ms
from mindspore import nn, ops, Tensor, Parameter

from transformers import AutoTokenizer
from datasets import load_dataset
from mindone.transformers.mindspore_adapter import auto_mixed_precision, TrainOneStepWrapper

from transformers_models.qwen2 import Qwen2ForCausalLM
from src.grpo import GRPO
from src.rewards import format_reward, countdown_game_accuracy_reward


def run():
    model_path = "/Users/zhanghuiyao/Desktop/_R1/r1_mindspore/hf_configs/Qwen/Qwen2.5-1.5B-Instruct"
    dtype = ms.float16

    batch = {
        "prompts": [
            "<|im_start|>system\nYou are a helpful assistant. You first thinks about the reasoning process in the mind and then provides the user with the answer.<|im_end|>\n<|im_start|>user\nUsing the numbers [83, 91, 4], create an equation that equals 32. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. Show your work in <think> </think> tags. And return the final equation and answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 = 1 </answer>.<|im_end|>\n<|im_start|>assistant\nLet me solve this step by step.\n<think>"],
        "nums": [[83, 91, 4]],
        "targets": [32],
        "prompt_ids": np.array([[151644, 8948, 198, 2610, 525, 264, 10950, 17847,
                                 13, 1446, 1156, 15482, 911, 279, 32711, 1882,
                                 304, 279, 3971, 323, 1221, 5707, 279, 1196,
                                 448, 279, 4226, 13, 151645, 198, 151644, 872,
                                 198, 16429, 279, 5109, 508, 23, 18, 11,
                                 220, 24, 16, 11, 220, 19, 1125, 1855,
                                 458, 23606, 429, 16819, 220, 18, 17, 13,
                                 1446, 646, 990, 6770, 34784, 7525, 17973, 11,
                                 85922, 11777, 608, 8, 323, 1817, 1372, 646,
                                 1172, 387, 1483, 3055, 13, 6928, 697, 975,
                                 304, 366, 26865, 29, 690, 26865, 29, 9492,
                                 13, 1597, 470, 279, 1590, 23606, 323, 4226,
                                 304, 366, 9217, 29, 690, 9217, 29, 9492,
                                 11, 369, 3110, 366, 9217, 29, 320, 16,
                                 488, 220, 17, 8, 608, 220, 18, 284,
                                 220, 16, 690, 9217, 14276, 151645, 198, 151644,
                                 77091, 198, 10061, 752, 11625, 419, 3019, 553,
                                 3019, 624, 13708, 766, 29, 151645, 151645, 151645,
                                 151645, 151645, 151645, 151645, 151645, 151645, 151645, 151645,
                                 151645, 151645, 151645, 151645, 151645, 151645, 151645, 151645,
                                 151645, 151645, 151645, 151645, 151645, 151645, 151645, 151645,
                                 151645, 151645, 151645, 151645, 151645, 151645, 151645, 151645,
                                 151645, 151645, 151645, 151645, 151645, 151645, 151645, 151645,
                                 151645, 151645, 151645, 151645, 151645, 151645, 151645, 151645,
                                 151645, 151645, 151645, 151645, 151645, 151645, 151645, 151645,
                                 151645, 151645, 151645, 151645, 151645, 151645, 151645, 151645,
                                 151645, 151645, 151645, 151645, 151645, 151645, 151645, 151645,
                                 151645, 151645, 151645, 151645, 151645, 151645, 151645, 151645,
                                 151645, 151645, 151645, 151645, 151645, 151645, 151645, 151645,
                                 151645, 151645, 151645, 151645, 151645, 151645, 151645, 151645,
                                 151645, 151645, 151645, 151645, 151645, 151645, 151645, 151645,
                                 151645, 151645, 151645, 151645, 151645, 151645, 151645, 151645]], np.int32),
        "attention_mask": np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                     1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], np.int32)
    }

    ms.set_context(mode=ms.GRAPH_MODE, jit_config={"jit_level": "O0"}, pynative_synchronize=True)  # FIXME

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="Qwen/Qwen2.5-1.5B-Instruct", help="pretrained model")
    parser.add_argument("--dataset-path", type=str, default="Jiayi-Pan/Countdown-Tasks-3to4", help="dataset path.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=5e-7)
    parser.add_argument("--lr-scheduler-type", type=str, default="cosine")  # FIXME
    parser.add_argument("--max-prompt-length", type=int, default=256)
    parser.add_argument("--max-completion-length", type=int, default=3)  # 1024
    parser.add_argument("--num-generations", type=int, default=2)
    parser.add_argument("--beta", type=float, default=0.001)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument(
        "--zero-stage", type=int, default=0, choices=[0, 1, 2], help="stage of ZeRO optimizer parallelism"
    )
    parser.add_argument(
        "--fp16", action="store_true", default=False, help="whether or not to enable mix precision with float16"
    )
    parser.add_argument(
        "--bf16", action="store_true", default=False, help="whether or not to enable mix precision with bfloat16"
    )
    parser.add_argument(
        "--is-distribute", type=ast.literal_eval, default=False, help="whether or not to run distribute"
    )
    parser.add_argument("--rank", type=int, default=0, help="id of card")
    parser.add_argument("--rank_size", type=int, default=1, help="num of cards")
    args = parser.parse_args()
    print(f"{args=}")

    # create train network and mix precision
    policy_model = Qwen2ForCausalLM.from_pretrained(
        model_path,
        use_flash_attention_2=False,  # True,  # FIXME, for mac
        mindspore_dtype=ms.float16,
    )
    reference_model = Qwen2ForCausalLM.from_pretrained(
        model_path,
        use_flash_attention_2=False,  # True, # FIXME, for mac
        mindspore_dtype=dtype,
    )

    policy_model.gradient_checkpointing_enable()

    policy_model = auto_mixed_precision(policy_model, "O2", dtype)
    reference_model = auto_mixed_precision(reference_model, "O2", dtype)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.pad_token = tokenizer.eos_token

    grpo_model = GRPO(policy_model, reference_model, [format_reward, countdown_game_accuracy_reward], tokenizer, args)

    # 3. get reward
    prompt_completion_ids, num_logits_to_keep, completion_mask, rewards = \
        grpo_model.get_completion_and_reward(batch)

    print(f"outputs shape: {[x.shape for x in [prompt_completion_ids, completion_mask, rewards]]}")
    print(f"rewards: {rewards}")


if __name__ == '__main__':
    run()
