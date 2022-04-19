import pickle as pkl
from dataclasses import dataclass

import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F

from data import make_binary_datasets
from mllg import LogWriter, TestInfo, TrainInfo, ValidationInfo

from yamlargs.parser import load_config_and_create_parser, parse_args_into_config

import config


def eval_model(model, test_dl, sample_size):
    correct = 0.0
    total = 0.0
    for (x, y) in test_dl:
        with torch.no_grad():
            out = model(x)
        pred = torch.argmax(out, dim=1).cpu()

        correct += torch.sum(pred == y).item()
        total += pred.shape[0]

        if total > sample_size:
            break

    return correct / total


@dataclass
class StandardTrain:

    batch_size: int = 1024
    num_epochs: int = 1
    eval_freq: int = 1000
    lr: float = 1e-4

    def __call__(self, dfa, model, tokenizer, train_data, test_data, logger):

        train_dl = DataLoader(
            train_data,
            batch_size=self.batch_size,
            collate_fn=train_data.create_collate(tokenizer),
            shuffle=True,
        )

        test_dl = DataLoader(
            test_data,
            batch_size=self.batch_size,
            collate_fn=test_data.create_collate(tokenizer),
            shuffle=True,
        )
        opt = optim.Adam(model.parameters(), lr=self.lr)

        eval_acc = eval_model(model, test_dl, 1000)
        logger.log_info(ValidationInfo(0, 0, [TestInfo("ACC", eval_acc)]))

        for epoch in range(self.num_epochs):
            for (batch_idx, (x, y)) in enumerate(train_dl):

                opt.zero_grad()
                out = model(x)
                loss = F.cross_entropy(out, y.cuda())
                loss.backward()
                logger.log_info(TrainInfo(epoch, batch_idx, loss.item()))
                opt.step()

                if batch_idx % self.eval_freq == 0 and batch_idx > 0:
                    eval_acc = eval_model(model, test_dl, 1000)
                    logger.log_info(
                        ValidationInfo(epoch, batch_idx, [TestInfo("ACC", eval_acc)])
                    )

            eval_acc = eval_model(model, test_dl, 1000)
            logger.log_info(
                ValidationInfo(epoch, batch_idx, [TestInfo("ACC", eval_acc)])
            )


if __name__ == "__main__":
    config, parser = load_config_and_create_parser()
    parser.add_argument("log_path")
    args = parser.parse_args()

    config = parse_args_into_config(config, args)

    logger = LogWriter(args.log_path)
    config_data = config.to_json()
    config_data["type"] = "config"
    logger.log_str(str(config_data))

    dfa = config["DFA"](config["datasets"]["max_size"])

    with open(f"{args.log_path}/config.yml", "w") as cfg_save:
        cfg_save.write(config.to_yaml())

    dfa.show_diagram(path=f"{args.log_path}/dfa.png")
    pkl.dump(dfa, open(f"{args.log_path}/dfa.pkl", "wb"))

    train_data, test_data = config["datasets"](dfa)

    model = config["model"]()
    model = model.cuda()

    tokenizer = config["tokenizer"]()
    StandardTrain()(dfa, model, tokenizer, train_data, test_data, logger)
