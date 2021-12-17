from __future__ import annotations
from dataclasses import dataclass, fields

import numpy as np
from numpy.typing import ArrayLike

from typing import List


@dataclass
class Parameters:
    Wxh: ArrayLike
    Whh: ArrayLike
    Why: ArrayLike
    bh: ArrayLike
    by: ArrayLike


class RNNCell:
    def __init__(self, p: Parameters, d_p: Parameters, vocab_size: int):
        self.vocab_size = vocab_size
        self.p = p
        self.d_p = d_p

        # states
        self.xs: ArrayLike = None
        self.hs: ArrayLike = None
        self.ys: ArrayLike = None
        self.ps: ArrayLike = None

    def forward(self, inputs: int, prev_hs: ArrayLike):
        self.xs = np.zeros((self.vocab_size, 1))  # encode in 1-of-k representation
        self.xs[inputs] = 1
        x1 = np.dot(self.p.Wxh, self.xs)
        x2 = np.dot(self.p.Whh, prev_hs)
        x3 = x2 + self.p.bh
        x4 = x1 + x3
        self.hs = np.tanh(x4)
        # hidden state
        self.ys = np.dot(self.p.Why, self.hs) + self.p.by  # unnormalized log probabilities for
        # next chars
        self.ps = np.exp(self.ys) / np.sum(np.exp(self.ys))  # probabilities for next chars
        return self.ps, self.hs

    def backward(self, target, prev_hs, next_d_hs):
        d_y = np.copy(self.ps)
        d_y[target] -= 1  # see http://cs231n.github.io/neural-networks-case-study/#grad

        self.d_p.Why += np.dot(d_y, self.hs.T)
        self.d_p.by += d_y
        dh = np.dot(self.p.Why.T, d_y) + next_d_hs  # backprop into h
        d_hraw = (1 - self.hs * self.hs) * dh  # backprop through tanh nonlinearity
        self.d_p.bh += d_hraw
        self.d_p.Wxh += np.dot(d_hraw, self.xs.T)
        self.d_p.Whh += np.dot(d_hraw, prev_hs.T)
        return np.dot(self.p.Whh.T, d_hraw)

    def reset(self):
        self.xs: ArrayLike = None
        self.hs: ArrayLike = None
        self.ys: ArrayLike = None
        self.ps: ArrayLike = None


class RNN:
    def __init__(self, hidden_size: int, sequence_length: int, vocab_size: int):
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size

        # model parameters
        self.p = Parameters(np.random.randn(hidden_size, vocab_size) * 0.01,
                            np.random.randn(hidden_size, hidden_size) * 0.01,
                            np.random.randn(vocab_size, hidden_size) * 0.01,
                            np.zeros((hidden_size, 1)),
                            np.zeros((vocab_size, 1)))

        self.d_p = Parameters(np.zeros_like(self.p.Wxh),
                              np.zeros_like(self.p.Whh),
                              np.zeros_like(self.p.Why),
                              np.zeros_like(self.p.bh),
                              np.zeros_like(self.p.by))

        self.cells = [RNNCell(self.p, self.d_p, vocab_size) for _ in range(sequence_length)]

        self._init_state = np.zeros_like(self.cells[0].hs)

    def forward(self, inputs: List[int], targets: List[int], hprev: ArrayLike) -> float:
        self._init_state = np.copy(hprev)
        prev_state = hprev
        loss: float = 0.0
        for i in range(len(inputs)):
            cell = self.cells[i]
            cell.forward(inputs[i], prev_state)
            prev_state = cell.hs

            loss += -np.log(cell.ps[targets[i], 0])  # softmax (cross-entropy loss)

        return loss

    def backward(self, targets) -> None:
        next_d_hs = np.zeros_like(self.cells[0].hs)
        for i in reversed(range(self.sequence_length)):
            cell = self.cells[i]
            prev_hs = self._init_state if i == 0 else self.cells[i - 1].hs
            next_d_hs = cell.backward(targets[i], prev_hs, next_d_hs)

    def sample(self, idx: int, hidden_init: ArrayLike, sample_size: int) -> List[int]:
        """ Sample a sequence of integers from the model """
        cell = RNNCell(self.p, self.d_p, self.vocab_size)
        indices: List[int] = [idx]
        hidden = np.copy(hidden_init)

        for _ in range(sample_size):
            probs, hidden = cell.forward(idx, hidden)
            idx = np.random.choice(range(self.vocab_size), p=probs.ravel())
            indices.append(idx)
        return indices

    def reset_states(self):
        [cell.reset() for cell in self.cells]


class AdaGrad:
    def __init__(self, p: Parameters, d_p: Parameters, learning_rate: float):
        self.learning_rate = learning_rate
        self.p = p
        self.d_p = d_p
        self.m_p = Parameters(Wxh=np.zeros_like(p.Wxh),
                              Whh=np.zeros_like(p.Whh),
                              Why=np.zeros_like(p.Why),
                              bh=np.zeros_like(p.bh),
                              by=np.zeros_like(p.by))

    def update(self):
        self.clip()
        for field in fields(self.p):
            param, d_param, m_param = [getattr(o, field.name) for o in [self.p, self.d_p, self.m_p]]
            m_param += d_param * d_param
            param += -self.learning_rate * d_param / np.sqrt(m_param + 1e-8)

    def zero_grads(self):
        self.d_p.Wxh = np.zeros_like(self.p.Wxh)
        self.d_p.Whh = np.zeros_like(self.p.Whh)
        self.d_p.Why = np.zeros_like(self.p.Why)
        self.d_p.bh = np.zeros_like(self.p.bh)
        self.d_p.by = np.zeros_like(self.p.by)

    def clip(self):
        for field in fields(self.d_p):
            d_param = getattr(self.d_p, field.name)
            np.clip(d_param, -5, 5, out=d_param)


def train_rnn():
    data = open("resources/pan-tadeusz.txt", "r", encoding="UTF-8").read()  # should be simple plain
    # text file
    chars = list(set(data))
    data_size, vocab_size = len(data), len(chars)
    print("data has %d characters, %d unique." % (data_size, vocab_size))
    char_to_ix = {ch: i for i, ch in enumerate(chars)}
    ix_to_char = {i: ch for i, ch in enumerate(chars)}

    # hyperparameters
    hidden_size = 100  # size of hidden layer of neurons
    sequence_length = 25  # number of steps to unroll the RNN for
    learning_rate = 1e-1

    n, p = 0, 0

    smooth_loss = -np.log(1.0 / vocab_size) * sequence_length  # loss at iteration 0

    init_h = np.zeros((hidden_size, 1))
    rnn = RNN(hidden_size, sequence_length, vocab_size)
    optimizer = AdaGrad(rnn.p, rnn.d_p, learning_rate)

    while True:
        # prepare inputs (we're sweeping from left to right in steps seq_length long)
        if p + sequence_length + 1 >= len(data) or n == 0:
            init_h = np.zeros((hidden_size, 1))  # reset RNN memory
            p = 0  # go from start of data
        inputs = [char_to_ix[ch] for ch in data[p: p + sequence_length]]
        targets = [char_to_ix[ch] for ch in data[p + 1: p + sequence_length + 1]]

        # sample from the model now and then
        if n % 100 == 0:
            sample_ix = rnn.sample(inputs[0], init_h, 200)
            txt = "".join(ix_to_char[ix] for ix in sample_ix)
            print("----\n %s \n----" % (txt,))

        loss = rnn.forward(inputs, targets, init_h)
        rnn.backward(targets)
        optimizer.update()

        init_h = np.copy(rnn.cells[-1].hs)
        # rnn.reset_states()
        optimizer.zero_grads()
        smooth_loss = smooth_loss * 0.999 + loss * 0.001
        if n % 100 == 0:
            print("iter %d, loss: %f" % (n, smooth_loss))

        p += sequence_length  # move data pointer
        n += 1  # iteration counter


if __name__ == '__main__':
    train_rnn()
