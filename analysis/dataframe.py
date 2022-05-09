
import ast
import tqdm
from glob import glob
import pickle as pkl
import das.config
import numpy as np
import pandas as pd

from mllg.parse import RunSummary
from yamlargs import YAMLConfig


def config_to_data(config, l):
    name = config["model"].get("lazyObject", None)
    name = name[1:-2]

    dataset_type = config["DFA"].get("lazyObject", None)
    dataset_type = dataset_type[1:-2]

    alpha_num = config["DFA"]["kwargs"].get("alpha_num", None) 
    n_states = config["DFA"]["kwargs"].get("n_states", None) 
    width = config["DFA"]["kwargs"].get("width", None) 
    depth = config["DFA"]["kwargs"].get("depth", None) 
    
    model_layers = config["model"]["kwargs"].get("nlayers", None)

    data = {
        'model_name': name,
        'model_layers': model_layers,
        'dataset_type': dataset_type,
        'n_states': n_states,
        'width': width,
        'depth': depth,
        'alpha_num': alpha_num,
    }

    rs = RunSummary.from_file(l)

    times, vals = rs.validation_vals("ACC")
    if len(vals) == 5:
        for i in range(5):
            data[f'epoch_{i}'] = vals[i]
        return data

    return None


def create_df(logs):

    total = []

    for l in tqdm.tqdm(logs):

        with open(l, "r") as data:
            config_str = data.readline()

        try:
            config = YAMLConfig.from_json(ast.literal_eval(config_str)).to_json()
        except:
            print(config_str)
            continue

        data = config_to_data(config, l)
        if data:
            total.append(data)

    df = pd.DataFrame(total)
    df.to_pickle('df.pkl')

if __name__ == '__main__':
    logs = glob("./data/*/train.log")
    create_df(logs)
