"""Shuffle/drop-last loader with explicit, resumeable order and private RNG."""
import numpy as np
import torch

class StatefulBatches:
    def __init__(self, x, labels, batch_size, seed):
        self.x, self.labels, self.batch_size = x, labels, batch_size
        if len(x) < batch_size:
            raise ValueError('Dataset is smaller than batch_size')
        self.generator = torch.Generator().manual_seed(seed)
        self.order = torch.randperm(len(x), generator=self.generator)
        self.cursor = 0
        self.last_indices = None

    def __next__(self):
        if self.cursor + self.batch_size > len(self.x):
            self.order = torch.randperm(len(self.x), generator=self.generator)
            self.cursor = 0
        idx = self.order[self.cursor:self.cursor+self.batch_size].numpy()
        self.cursor += self.batch_size
        self.last_indices = idx.copy()
        return torch.from_numpy(self.x[idx].copy()), {'y':torch.as_tensor(self.labels[idx], dtype=torch.long)}

    def state_dict(self):
        return dict(generator=self.generator.get_state(), order=self.order, cursor=self.cursor,
                    size=len(self.x), batch_size=self.batch_size)

    def load_state_dict(self, state):
        if state['size'] != len(self.x) or state['batch_size'] != self.batch_size:
            raise ValueError('Resume data shape/batch size changed')
        self.generator.set_state(state['generator']); self.order = state['order']; self.cursor = state['cursor']
