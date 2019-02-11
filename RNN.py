import torch
import numpy as np
import torch.nn as nn

class RNN(nn.Module):
	def __init__(self):
		super(RNN, self).__init__()

		self.rnn = nn.LSTM(
			input_size=3,
			hidden_size=3+1,
			num_layers=2,
			batch_first=True,
		)

	def forward(self, x):
		out, (h_n, h_c) = self.rnn(x, None)
		return out[:, -1, :]
