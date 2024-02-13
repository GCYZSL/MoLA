import os
import re
import json
import argparse
import random
from tqdm import tqdm

import torch
from src.mola_peft_model_hacked import PeftModel
from transformers import GenerationConfig, LlamaTokenizer, AutoConfig
import sys
from typing import Union
from src.mola_modeling_llama_hacked import LlamaForCausalLM_d

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass

seed = 10
random.seed(seed)  # random seed
torch.manual_seed(0)


class Prompter(object):
    __slots__ = ("template", "_verbose")

    def __init__(self, template_name: str = "", verbose: bool = False):
        self._verbose = verbose
        self.template = {
            "description": "Template used by Alpaca-LoRA.",
            "prompt_input": "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n",
            "prompt_no_input": "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n",
            "response_split": "### Response:"
        }

        if self._verbose:
            print(
                f"Using prompt template {template_name}: {self.template['description']}"
            )

    def generate_prompt(
            self,
            instruction: str,
            input: Union[None, str] = None,
            label: Union[None, str] = None,
    ) -> str:
        # returns the full prompt from instruction and optional input
        # if a label (=response, =output) is provided, it's also appended.
        if input:
            res = self.template["prompt_input"].format(
                instruction=instruction, input=input
            )
        else:
            res = self.template["prompt_no_input"].format(
                instruction=instruction
            )
        if label:
            res = f"{res}{label}"
        if self._verbose:
            print(res)
        return res

    def get_response(self, output: str) -> str:
        return output.split(self.template["response_split"])[1].strip()


def main():
    parser = argparse.ArgumentParser(description='Evaluation')
    # Defining arguments
    parser.add_argument('--test_dataset', type=str, default="./scienceqa/scienceq_test.json",
                        help='test_dataset')
    parser.add_argument('--base_model', type=str, default="NousResearch/Llama-2-7b-hf", help='base_model')
    parser.add_argument('--mola_weights', type=str, default="./scienceqa_mola",
                        help='mola_model')
    parser.add_argument('--number_experts', type=str,
                        default="2,2,2,2,2,2,2,2,4,4,4,4,4,4,4,4,6,6,6,6,6,6,6,6,8,8,8,8,8,8,8,8",
                        help='experts number')
    parser.add_argument('--top_k', type=str,
                        default="2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2",
                        help='lora_model')
    parser.add_argument('--save_path', type=str,
                        default="./results/mola_test_sciqa_seed_10.json",
                        help='lora_model')
    parser.add_argument('--lora_target_modules', type=str,
                        default="q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj",   help='lora_target_modules')
    parser.add_argument('--batch_size', type=int, default=8, help='base_model')

    # Parsing arguments
    args = parser.parse_args()
    data_a = json.load(open(args.test_dataset))
    base_model = args.base_model
    mola_weights = args.mola_weights
    max_batch_size = args.batch_size

    lora_target_modules = args.lora_target_modules.split(",")
    lora_target_modules = [str(lr) for lr in lora_target_modules]
    print(lora_target_modules)
    number_experts = args.number_experts.split(",")
    number_experts = [int(lr) for lr in number_experts]
    print(number_experts)
    top_k = args.top_k.split(",")
    top_k = [int(lr) for lr in top_k]
    print(top_k)

    print(args.test_dataset)
    print(args.base_model)
    print(args.mola_weights)

    load_8bit = False

    tokenizer = LlamaTokenizer.from_pretrained(base_model, padding_side='left')
    config = AutoConfig.from_pretrained(base_model)
    config.lora_target_modules = lora_target_modules
    if device == "cuda":
        model = LlamaForCausalLM_d.from_pretrained(
            base_model,
            config=config,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(
            model,
            mola_weights,
            torch_dtype=torch.float16,
            number_experts=number_experts,
            top_k=top_k,
        )
    else:
        model = LlamaForCausalLM_d.from_pretrained(
            base_model, config=config, device_map={"": device}, low_cpu_mem_usage=True
        )
        model = PeftModel.from_pretrained(
            model,
            mola_weights,
            device_map={"": device},
        )
    obalance = False
    model.get_new_parameters(number_experts, top_k, obalance)

    print(model.config.pad_token_id, tokenizer.pad_token_id)
    print(model.config.bos_token_id, tokenizer.bos_token_id)
    print(model.config.eos_token_id, tokenizer.eos_token_id)
    # unwind broken decapoda-research config
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    if not load_8bit:
        model.half()  # seems to fix bugs for some users.

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    prompter = Prompter(template_name="alpaca")

    max_new_tokens = 128
    save_every = 200

    correct = 0
    results = []
    outputs = []
    gt = []

    for start_idx in tqdm(range(0, len(data_a), max_batch_size)):
        end_idx = min(start_idx + max_batch_size, len(data_a))
        batch = data_a[start_idx:end_idx]

        answers = [str(example["answer"]) for example in batch]

        # generate prompt
        prompts = [prompter.generate_prompt(example['instruction'], example['input']) for example in batch]

        inputs = tokenizer(prompts, padding=True, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)


        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )
        s = generation_output.sequences
        output = tokenizer.batch_decode(s)
        output = [prompter.get_response(otp) for otp in output]

        # extract the answer
        print(output)
        # pattern = re.compile(r'The anwser to the question is (\d+):*')
        pattern = re.compile(r'The answer is ([A-Z]).')
        res = [pattern.findall(otp) for otp in output]
        print(res)
        pred = []
        for r_i in range(len(res)):
            if len(res[r_i]) == 1:
                answer = res[r_i][0]  # 'A', 'B', ...
            else:
                answer = "FAILED"
                print(res[r_i])
            pred.append(answer)
            results.append(res[r_i])
            outputs.append(output[r_i])
            gt.append(answers[r_i])

            if str(answer) == str(answers[r_i]):
                correct += 1
                print('correct:', str(answer), str(answers[r_i]))
            else:
                print('gt-ans:', str(answer), str(answers[r_i]))


        acc = correct / len(results) * 100

        if end_idx % save_every == 0 or end_idx == len(data_a):
            result_file = args.save_path
            print(f"{len(results)}/{len(data_a)}, correct: {correct}, acc: {round(acc, 2)}%, saving to {result_file}")
            data = {}
            data['acc'] = acc
            data['correct'] = correct
            data['len'] = len(results)
            data['results'] = results
            data['outputs'] = outputs
            with open(result_file, 'w') as f:
                json.dump(data, f, indent=2, separators=(',', ': '))


if __name__ == "__main__":
    main()