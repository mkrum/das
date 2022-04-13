import random

import torch
import numpy as np
from automata.fa.dfa import DFA
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from transformers import PreTrainedTokenizer

from model import SimpleEncoder

def sample_string(vocab, min_len, max_len):
    # TODO: Fix this, it is slow
    size = np.random.choice(range(min_len, max_len + 1))
    return ''.join(np.random.choice(list(vocab), size=size))

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

class DFADataset(Dataset):

    def __init__(self, dfa, min_len, max_len):
        super().__init__()
        self.dfa = dfa 
        self.min_len = min_len
        self.max_len = max_len

    def __len__(self): 
        # Return some very, very large number to simulate infinite data
        return 2 ** 30

    def __getitem__(self, idx):
        """
        Randomly samples a string, labels whether or not it accepts
        """
        random_string = sample_string(set(["0", "1"]), self.min_len, self.max_len)
        label = int(self.dfa.accepts_input(random_string))
        return random_string, label

    def create_collate(self, tokenizer):
        def collate_fn(batch):
            inputs, labels = zip(*batch)
            return tokenizer(
                inputs, padding=True, return_tensors="pt"
            ), torch.LongTensor(labels)

        return collate_fn

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

if __name__ == '__main__':
    min_len = 1
    max_len = 21

    dfa = generate_random_binary_dfa(n_states=11)

    data = DFADataset(dfa, min_len, max_len)

    tokenizer = SimpleDFATokenizer(["0", "1"], max_len + 2)

    dl = DataLoader(data, batch_size=1024, collate_fn=data.create_collate(tokenizer))

    model = SimpleEncoder(2, nlayers=6, nhead=8, d_model=32, d_hid=32).cuda()
    opt = optim.Adam(model.parameters(), lr=1e-3)
    for (x, y) in dl:
        opt.zero_grad()
        out = model(x)
        loss = F.cross_entropy(out, y.cuda())
        loss.backward()
        print(loss.item())
        opt.step()
