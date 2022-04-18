import torch
import random
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from automata.fa.dfa import DFA
import numpy as np


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


def generate_random_binary_dfa(max_input_len, n_states=640):
    states = [f"q{i}" for i in range(n_states)]

    transitions = {}
    final_states = []
    for s in states:
        zero_state = ""
        one_state = ""
        while zero_state == one_state:
            zero_state = np.random.choice(states)
            one_state = np.random.choice(states)

        transitions[s] = {"0": zero_state, "1": one_state}
        if random.random() <= 0.5:
            final_states.append(s)

    dfa = DFA(
        states=set(states),
        input_symbols={"0", "1"},
        transitions=transitions,
        initial_state="q0",
        final_states=set(final_states),
    )
    dfa = dfa.minify()
    if len(dfa.states) != n_states:
        return generate_random_binary_dfa(
            max_input_len=max_input_len, n_states=n_states
        )

    N = 1000
    accept = 0.0
    for i in range(N):
        random_string = sample_string(["0", "1"], max_input_len, max_input_len)
        accept += int(dfa.accepts_input(random_string))

    acc = accept / N
    if (acc > 0.55) or (acc < 0.45):
        return generate_random_binary_dfa(
            max_input_len=max_input_len, n_states=n_states
        )

    return dfa


def generate_data(max_size):
    binary_data = ["0", "1"]
    for j in range(1, max_size):
        for i in range(len(binary_data)):
            d = binary_data[i]
            if len(d) == j:
                binary_data.append(d + "0")
                binary_data.append(d + "1")
    random.shuffle(binary_data)
    return binary_data


def make_binary_datasets(dfa, max_size, train_per):
    data = generate_data(max_size)

    total = len(data)
    train_size = int(train_per * total)

    train_data = data[:train_size]
    test_data = data[train_size:]
    return DFADataset(dfa, train_data), DFADataset(dfa, test_data)


class DFADataset(Dataset):
    def __init__(self, dfa, data):
        super().__init__()
        self.dfa = dfa
        self.data = data

    def __len__(self):
        # Return some very, very large number to simulate infinite data
        return len(self.data)

    def __getitem__(self, idx):
        """
        Randomly samples a string, labels whether or not it accepts
        """
        random_string = self.data[idx]
        label = int(self.dfa.accepts_input(random_string))
        return random_string, label

    def create_collate(self, tokenizer):
        def collate_fn(batch):
            inputs, labels = zip(*batch)
            return tokenizer(
                inputs, padding=True, return_tensors="pt"
            ), torch.LongTensor(labels)

        return collate_fn


def sample_string(vocab, min_len, max_len):
    # TODO: Fix this, it is slow
    size = np.random.choice(range(min_len, max_len + 1))
    return "".join(np.random.choice(list(vocab), size=size))


class RandomDFADataset(Dataset):
    def __init__(self, dfa, min_len, max_len):
        super().__init__()
        self.dfa = dfa
        self.min_len = min_len
        self.max_len = max_len

    def __len__(self):
        # Return some very, very large number to simulate infinite data
        return 2**30

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
