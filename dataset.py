import json
import datasets
import pandas as pd
from datasets import load_from_disk, Dataset
from torch.utils.data import DataLoader
from typing import List, Optional



def get_input_prompt(instruction):
    prompt_config = {
        'system_format': "<|system|>\n{system}",
        'system_no_input_prompt': "Below is a query related to banking compliance. Please provide appropriate response in a formal language",
        'turn_no_input_format': "\n<|user|>\n{instruction}\n<|assistant|>\n"
    }
    res = prompt_config['system_format'].format(system=prompt_config['system_no_input_prompt']) \
        + prompt_config['turn_no_input_format'].format(instruction=instruction)
    return res


def process_datasets(dataset_paths: list[str]):
    # Load the raw csv 
    dfs = [pd.read_csv(i) for i in dataset_paths]
    print([len(i) for i in dfs])

    # Combine all the raw csvs
    df = pd.concat(dfs)
    print(len(df))
    
    # Process the combined csv--> ['instruction', 'response']
    df.rename(columns = {"Question" : "instruction", "Answer" : "response"}, inplace= True)
    df.drop(columns = ["Source", "Type", "Comments"], inplace = True)
    df.dropna(inplace = True)

    data = [
        dict(
            question = j['instruction'],
            instruction = get_input_prompt(j['instruction']),
            reponse = j['response']
            )
            for i, j in df.iterrows()
            ]
    df = pd.DataFrame(data)
    
    # Return the csv
    return df

def get_loader(
        df: pd.DataFrame, # path of the combined_csv
        batch_size: int # Batch size for the data loader
        ):
    
    # Define the arrow dataset
    dataset = Dataset.from_pandas(df)

    # Define the dataloader
    loader = DataLoader(dataset, batch_size = batch_size)

    return loader

if __name__ == "__main__":
    dataset_paths = [
        "/home/atharva_inamdar/system_evaluation/datasets/nayan.csv",
        "/home/atharva_inamdar/system_evaluation/datasets/tony.csv"
    ]
    df = process_datasets(dataset_paths)
    loader = get_loader(df, 2)
    # for batch in loader:
    #     print(batch['question'][0])
    #     print(batch['instruction'][0])
    #     break
    # response = """afm;sadfm; sa;fm ;fms
    # |<assistant>| This is the assistancr response"""
    # print(process_response(response))