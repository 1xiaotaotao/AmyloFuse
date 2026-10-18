import torch
import torch.nn as nn


class AttentionModule(nn.Module):
	"""Lightweight sample-wise gating over a dynamic number of feature sources."""

	def __init__(self, dim: int, num_sources: int):
		super().__init__()

		self.dim = dim
		self.num_sources = num_sources
		self.norm = nn.LayerNorm(dim, elementwise_affine=False)
		self.scorer = nn.Linear(dim, 1, bias=False)
		self.source_bias = nn.Parameter(torch.zeros(num_sources))
		self.gate_logit = nn.Parameter(torch.zeros(()))
		# self.reset_parameters()

	def reset_parameters(self):
		nn.init.zeros_(self.scorer.weight)
		nn.init.zeros_(self.source_bias)
		nn.init.zeros_(self.gate_logit)


	def forward(self, x): # x: (batch, n, 64)
		scores = self.scorer(self.norm(x)).squeeze(-1) # scores: (batch, n)
		weights = torch.softmax(scores, dim=-1) # weights: (batch, n)
		return x * weights.unsqueeze(-1), weights
