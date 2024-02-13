from datasets import load_dataset
import datasets
import json
import os
import pandas as pd
import argparse
from tqdm import tqdm
import random
seed = 10
random.seed(seed)

def get_context_text(problem, use_caption):
    txt_context = problem['hint']
    img_context = problem['caption'] if use_caption else ""
    context = " ".join([txt_context, img_context]).strip()
    if context == "":
        context = "N/A"
    return context


def get_choice_text(probelm, options):
    choices = probelm['choices']
    choice_list = []
    for i, c in enumerate(choices):
        choice_list.append("({}) {}".format(options[i], c))
    choice_txt = " ".join(choice_list)
    return choice_txt


def get_answer(problem, options):
    return options[problem['answer']]


def get_lecture_text(problem):
    # \\n: GPT-3 can generate the lecture with more tokens.
    lecture = problem['lecture'].replace("\n", "\\n")
    return lecture


def get_solution_text(problem):
    # \\n: GPT-3 can generate the solution with more tokens
    solution = problem['solution'].replace("\n", "\\n")
    return solution

def create_one_example(format, question, context, choice, answer, lecture, solution, test_example=True):

    input_format, output_format = format.split("-")

    ## Inputs
    if input_format == "CQM":
        input = f"Context: {context}\nQuestion: {question}\nOptions: {choice}\n"
    elif input_format == "QCM":
        input = f"Question: {question}\nContext: {context}\nOptions: {choice}\n"
    # upper bound experiment
    elif input_format == "QCML":
        input = f"Question: {question}\nContext: {context}\nOptions: {choice}\nBECAUSE: {lecture}\n"
    elif input_format == "QCME":
        input = f"Question: {question}\nContext: {context}\nOptions: {choice}\nBECAUSE: {solution}\n"
    elif input_format == "QCMLE":
        input = f"Question: {question}\nContext: {context}\nOptions: {choice}\nBECAUSE: {lecture} {solution}\n"

    elif input_format == "QCLM":
        input = f"Question: {question}\nContext: {context}\nBECAUSE: {lecture}\nOptions: {choice}\n"
    elif input_format == "QCEM":
        input = f"Question: {question}\nContext: {context}\nBECAUSE: {solution}\nOptions: {choice}\n"
    elif input_format == "QCLEM":
        input = f"Question: {question}\nContext: {context}\nBECAUSE: {lecture} {solution}\nOptions: {choice}\n"

    # Outputs
    if test_example:
        output = "Answer:"
    elif output_format == 'A':
        output = f"Answer: The answer is {answer}."

    elif output_format == 'AL':
        output = f"Answer: The answer is {answer}. BECAUSE: {solution}"
    elif output_format == 'AE':
        output = f"Answer: The answer is {answer}. BECAUSE: {lecture}"
    elif output_format == 'ALE':
        output = f"Answer: The answer is {answer}. BECAUSE: {lecture} {solution}"
    elif output_format == 'AEL':
        output = f"Answer: The answer is {answer}. BECAUSE: {solution} {lecture}"

    elif output_format == 'LA':
        output = f"Answer: {lecture} The answer is {answer}."
    elif output_format == 'EA':
        output = f"Answer: {solution} The answer is {answer}."
    elif output_format == 'LEA':
        output = f"Answer: {lecture} {solution} The answer is {answer}."
    elif output_format == 'ELA':
        output = f"Answer: {solution} {lecture} The answer is {answer}."

    text = input + output
    text = text.replace("  ", " ").strip()
    if text.endswith("BECAUSE:"):
        text = text.replace("BECAUSE:", "").strip()
    return input, output, text

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description='Process Instruction data')
    argparser.add_argument('--dataset', type=str, default="derek-thomas/ScienceQA", help='Instruction Dataset')
    argparser.add_argument('--save_path', type=str, default='./scienceqa', help='Save Path')
    args = argparser.parse_args()
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # We use huggingface datasets library to load datasets.
    # Here, we load the datasets from Huggingface Hub.
    # If you want to load it from local files or other methods, please refer https://huggingface.co/docs/datasets/en/loading
    datasets_scienceqa = load_dataset(args.dataset)
    splits = ['test', 'validation', 'train']
    options = ["A", "B", "C", "D", "E"]
    prompt_format = 'CQM-A'

    datasets_text_scienceqa = {'train': [], 'validation': [], 'test': []}
    for split in splits:
        for n in tqdm(range(len(datasets_scienceqa[split]))):
            if datasets_scienceqa[split][n]['image'] == None:
                data_sample = datasets_scienceqa[split][n]
                question = datasets_scienceqa[split][n]['question']
                context = get_context_text(datasets_scienceqa[split][n], False)
                choice = get_choice_text(datasets_scienceqa[split][n], options)
                answer = get_answer(datasets_scienceqa[split][n], options)
                lecture = get_lecture_text(datasets_scienceqa[split][n])
                solution = get_solution_text(datasets_scienceqa[split][n])

                data_sample['answer'] = answer
                data_sample['input'] = ""
                data_sample['instruction'], data_sample['output'], _ = create_one_example(prompt_format,
                                                                                          question,
                                                                                          context,
                                                                                          choice,
                                                                                          answer,
                                                                                          lecture,
                                                                                          solution,
                                                                                          test_example=False)
                datasets_text_scienceqa[split].append(data_sample)
        j_file = 'scienceq_' + split + '.json'
        with open(os.path.join(args.save_path, j_file), 'w') as json_file:
            json.dump(datasets_text_scienceqa[split], json_file)
        datasets_text_scienceqa[split] = datasets.Dataset.from_pandas(
            pd.DataFrame(data=datasets_text_scienceqa[split]))
    datasets_text_scienceqa = datasets.DatasetDict(datasets_text_scienceqa)
    datasets_text_scienceqa.save_to_disk(os.path.join(args.save_path, "science_qa.hf"))
    print(datasets_text_scienceqa)


