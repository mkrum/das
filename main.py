import random
import time
from collections import deque

import torch
import numpy as np
from automata.fa.dfa import DFA
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from transformers import PreTrainedTokenizer

from model import SimpleEncoder
from data import make_binary_datasets


class SimpleDFATokenizer(PreTrainedTokenizer):
    """
    Custom Tokenizer that create tokens at the nucelotide level.

    Nucleotide-level tokenizer, implements the huggingface tokenization class.
    Represents treating each individual nucleotide as its own token.
    """

    def __init__(self, symbols, max_len=1024):
        super().__init__(max_len=max_len, pad_token="<pad>")
        # add start and pad symbol
        self.symbols = symbols + ["<s>", "<pad>"]

    def _tokenize(self, sequence):
        return ["<s>"] + list(sequence)

    def _convert_token_to_id(self, token):
        return self.symbols.index(token)

    def _convert_id_to_token(self, idx):
        return self.symbols[idx]

    def get_vocab(self):
        return self.symbols

    def vocab_size(self):
        return len(self.symbols)


def generate_random_binary_dfa(n_states=640):
    states = [f"q{i}" for i in range(n_states)]

    transitions = {}
    final_states = []
    for s in states:
        zero_state = np.random.choice(states)
        one_state = np.random.choice(states)
        transitions[s] = {"0": zero_state, "1": one_state}
        if random.random() < 0.5:
            final_states.append(s)

    dfa = DFA(
        states=set(states),
        input_symbols={"0", "1"},
        transitions=transitions,
        initial_state="q0",
        final_states=set(final_states),
    )
    dfa = dfa.minify()
    return dfa

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
    
    return correct/total


if __name__ == "__main__":
    dfa = generate_random_binary_dfa(n_states=20)

    train_data, test_data = make_binary_datasets(dfa, 19, 0.8)

    tokenizer = SimpleDFATokenizer(["0", "1"])#, max_len=19 + 2)

    train_dl = DataLoader(
        train_data,
        batch_size=1024,
        collate_fn=train_data.create_collate(tokenizer),
        shuffle=True,
    )

    test_dl = DataLoader(
        test_data,
        batch_size=1024,
        collate_fn=test_data.create_collate(tokenizer),
        shuffle=True,
    )

    model = SimpleEncoder(2, nlayers=10, d_model=128, d_hid=128).cuda()
    opt = optim.Adam(model.parameters(), lr=1e-3)

    eval_acc = eval_model(model, test_dl, 10000)
    print(f"EVAL 0: {eval_acc}")
    
    losses = deque(maxlen=100)
    for epoch in range(20):
        for (batch_idx, (x, y)) in enumerate(train_dl):
            opt.zero_grad()
            out = model(x)
            loss = F.nll_loss(out, y.cuda())
            loss.backward()
            losses.append(loss.item())
            opt.step()
            print(f'({epoch:03} {batch_idx:04}/{len(train_dl):04}) {round(np.mean(losses), 2):.2f}', end='\r')

        eval_acc = eval_model(model, test_dl, 10000)
        print(f"EVAL {epoch+1}: {eval_acc}")
