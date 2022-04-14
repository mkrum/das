import torch
import random
from torch.utils.data import Dataset


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


def sample_string(vocab, min_len, max_len):
    # TODO: Fix this, it is slow
    size = np.random.choice(range(min_len, max_len + 1))
    return "".join(np.random.choice(list(vocab), size=size))


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
