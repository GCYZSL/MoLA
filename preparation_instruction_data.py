from datasets import load_dataset
from datasets import load_from_disk
import datasets
import json
import os
import string
import pandas as pd
from tqdm import tqdm
import random
import argparse

seed = 10
random.seed(seed)

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description='Process Instruction data')
    argparser.add_argument('--dataset', type=str, default="Open-Orca/OpenOrca", help='Instruction Dataset')
    argparser.add_argument('--num_samples', type=int, default=50000, help='Number of Samples')
    argparser.add_argument('--save_path', type=str, default='./meta_moe_50k.hf', help='Save Path')
    args = argparser.parse_args()

    print(args.dataset)
    print(args.num_samples)
    print(args.save_path)

    number_vali_test = 2000
    openorca_dataset = load_dataset(args.dataset)
    final_dataset = []
    final_dataset_vali = []
    final_dataset_test = []
    # Filter prepare
    type_filtered = {}
    openorca_dataset_filtered = []
    openorca_dataset_filtered_vali = []
    openorca_dataset_filtered_test = []
    print("GET DISTRIBUTION...")
    for i in tqdm(range(len(openorca_dataset['train']))):
        type = openorca_dataset['train'][i]['id'].split('.')[0]
        if type not in type_filtered:
            type_filtered[type] = []
            type_filtered[type].append(i)
        else:
            type_filtered[type].append(i)

    for i in type_filtered:
        n = int(len(type_filtered[i]) * (args.num_samples) / len(openorca_dataset['train']))
        n_vali_test = int(len(type_filtered[i]) * (number_vali_test) / len(openorca_dataset['train']))

        random_data = random.choices(type_filtered[i], k=n+n_vali_test)
        openorca_dataset_filtered += random_data[:n]
        openorca_dataset_filtered_vali += random_data[n:n+int(n_vali_test/2)]
        openorca_dataset_filtered_test += random_data[n+int(n_vali_test / 2):]

    print("Before filter: ", len(openorca_dataset['train']), "After filter: ", len(openorca_dataset_filtered))
    print("Validation: ", len(openorca_dataset_filtered_vali), "Test: ", len(openorca_dataset_filtered_test))

    datasets_split = {'train': [], 'validation': [], 'test': []}
    for i in tqdm(openorca_dataset_filtered):
        datasample = openorca_dataset['train'][i]
        question = datasample['question'].strip() + '\n'
        response = datasample['response']

        output = "".join([question, response]).strip()

        instruction = None
        input = None
        data_one = {'input': input, 'instruction': instruction, 'output': output}
        final_dataset.append(data_one)
    datasets_split['train'] = datasets.Dataset.from_pandas(
        pd.DataFrame(data=final_dataset))

    # Sample 1000 for vali 1000 for test
    for i in tqdm(openorca_dataset_filtered_vali):
        datasample = openorca_dataset['train'][i]
        question = datasample['question'].strip() + '\n'
        response = datasample['response']

        output = "".join([question, response]).strip()

        instruction = None
        input = None
        data_one = {'input': input, 'instruction': instruction, 'output': output}
        final_dataset_vali.append(data_one)
    datasets_split['validation'] = datasets.Dataset.from_pandas(
        pd.DataFrame(data=final_dataset_vali))

    for i in tqdm(openorca_dataset_filtered_test):
        datasample = openorca_dataset['train'][i]
        question = datasample['question'].strip() + '\n'
        response = datasample['response']

        output = "".join([question, response]).strip()

        instruction = None
        input = None
        data_one = {'input': input, 'instruction': instruction, 'output': output}
        final_dataset_test.append(data_one)
    datasets_split['test'] = datasets.Dataset.from_pandas(
        pd.DataFrame(data=final_dataset_test))


    datasets_split = datasets.DatasetDict(datasets_split)
    datasets_split.save_to_disk(args.save_path)
    print(datasets_split)