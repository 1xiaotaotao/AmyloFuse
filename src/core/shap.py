from __future__ import annotations

import csv
import logging
import pickle
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

import shap

LOGGER = logging.getLogger(__name__)


def _display_name(source_name: str) -> str:
	"""Return stable manuscript labels while preserving the configured order."""
	labels = {
		'prot-t5': 'ProtT5',
		'prot-bert': 'ProtBERT',
		'esm-1': 'ESM',
		'esm-2': 'ESM',
	}
	return labels.get(source_name, source_name)


class SourceGroupShapAnalyzer:
	"""Compute exact group-level SHAP values for three PLM feature sources."""

	def __init__(
		self,
		model: torch.nn.Module,
		train_dataset,
		test_dataset,
		*,
		background_batch_size: int = 32,
		compute_device: torch.device | str | None = None,
	):
		self.model = model
		self.train_dataset = train_dataset
		self.test_dataset = test_dataset
		self.source_names = list(model.feature_names)
		self.feature_names = [_display_name(name) for name in self.source_names]
		self.num_sources = len(self.source_names)
		self.background_batch_size = background_batch_size
		if compute_device is None:
			compute_device = next(model.parameters()).device
		self.device = torch.device(compute_device)

		for source_name in self.source_names:
			if source_name not in train_dataset.features:
				raise KeyError(f'{source_name!r} is missing from the training set')
			if source_name not in test_dataset.features:
				raise KeyError(f'{source_name!r} is missing from the test set')
			train_shape = tuple(train_dataset.features[source_name].shape[1:])
			test_shape = tuple(test_dataset.features[source_name].shape[1:])
			if train_shape != test_shape:
				raise ValueError(
					f'{source_name!r} shape differs between train and test: '
					f'{train_shape} versus {test_shape}'
				)

		self.model.to(self.device)
		self.model.eval()
		self._empty_coalition_value: float | None = None

	def _background_features(
		self,
		test_index: int,
		coalition: tuple[int, ...],
		start: int,
		stop: int,
	) -> dict[str, torch.Tensor]:
		"""Construct one matched training-background batch for a coalition."""
		batch_size = stop - start
		features = {}
		for source_index, source_name in enumerate(self.source_names):
			if coalition[source_index]:
				test_value = self.test_dataset.features[source_name][test_index]
				test_value = test_value.unsqueeze(0)
				expand_shape = (batch_size,) + tuple(test_value.shape[1:])
				value = test_value.expand(expand_shape)
			else:
				# All absent groups use the same training row indices. This keeps
				# the relationships among simultaneously absent sources intact.
				value = self.train_dataset.features[source_name][start:stop]
			features[source_name] = value.to(self.device)
		return features

	def _coalition_value(
		self,
		test_index: int,
		coalition: tuple[int, ...],
	) -> float:
		"""Expected model logit for one source coalition and one test sample."""
		if len(coalition) != self.num_sources:
			raise ValueError('Coalition length does not match the source count')

		if not any(coalition) and self._empty_coalition_value is not None:
			return self._empty_coalition_value

		# The full coalition is independent of the background. Evaluating it
		# once avoids repeating the same test sample for every training row.
		if all(coalition):
			features = {
				name: self.test_dataset.features[name][test_index]
				.unsqueeze(0)
				.to(self.device)
				for name in self.source_names
			}
			with torch.inference_mode():
				logits, _, _ = self.model(features)
			return float(logits.reshape(-1)[0].detach().cpu())

		total_logit = 0.0
		total_samples = 0
		with torch.inference_mode():
			for start in range(
				0,
				len(self.train_dataset),
				self.background_batch_size,
			):
				stop = min(
					start + self.background_batch_size,
					len(self.train_dataset),
				)
				features = self._background_features(
					test_index,
					coalition,
					start,
					stop,
				)
				logits, _, _ = self.model(features)
				logits = logits.reshape(-1).detach().double().cpu()
				total_logit += float(logits.sum())
				total_samples += logits.numel()

		value = total_logit / total_samples
		if not any(coalition):
			self._empty_coalition_value = value
		return value

	def explain_sample(self, test_index: int):
		"""Return a one-row SHAP Explanation for one test sample."""
		if not 0 <= test_index < len(self.test_dataset):
			raise IndexError(f'test_index {test_index} is out of range')

		coalition_cache: dict[tuple[int, ...], float] = {}

		def coalition_model(mask_matrix: np.ndarray) -> np.ndarray:
			mask_matrix = np.asarray(mask_matrix)
			if mask_matrix.ndim == 1:
				mask_matrix = mask_matrix.reshape(1, -1)
			if mask_matrix.shape[1] != self.num_sources:
				raise ValueError(
					'SHAP mask must have one column per feature source'
				)

			outputs = []
			for mask in mask_matrix:
				coalition = tuple(int(value >= 0.5) for value in mask)
				if coalition not in coalition_cache:
					coalition_cache[coalition] = self._coalition_value(
						test_index,
						coalition,
					)
				outputs.append(coalition_cache[coalition])
			return np.asarray(outputs, dtype=np.float64)

		# ExactExplainer sees three binary group-presence variables. The zero
		# mask is the baseline; training-background replacement is performed by
		# coalition_model rather than by SHAP's numeric masker.
		mask_background = np.zeros((1, self.num_sources), dtype=np.float64)
		full_coalition = np.ones((1, self.num_sources), dtype=np.float64)
		explainer = shap.ExactExplainer(
			coalition_model,
			mask_background,
			feature_names=self.feature_names,
		)
		explanation = explainer(
			full_coalition,
			max_evals=2 ** self.num_sources,
		)
		full_value = self._coalition_value(
			test_index,
			tuple([1] * self.num_sources),
		)
		reconstructed_value = (
			float(np.asarray(explanation.base_values).reshape(-1)[0])
			+ float(np.asarray(explanation.values).sum())
		)
		if not np.isclose(
			reconstructed_value,
			full_value,
			rtol=1e-5,
			atol=1e-6,
		):
			raise RuntimeError(
				'SHAP additivity check failed for test sample '
				f'{test_index}: reconstructed={reconstructed_value}, '
				f'model_logit={full_value}'
			)
		return explanation

	def explain_all(self):
		"""Explain every test sample and return one SHAP Explanation object."""
		values = []
		base_values = []
		for test_index in range(len(self.test_dataset)):
			LOGGER.info(
				'Explaining test sample %d/%d',
				test_index + 1,
				len(self.test_dataset),
			)
			sample_explanation = self.explain_sample(test_index)
			sample_values = np.asarray(sample_explanation.values).reshape(
				-1,
				self.num_sources,
			)
			values.append(sample_values[0])
			base_values.append(
				float(np.asarray(sample_explanation.base_values).reshape(-1)[0])
			)

		return shap.Explanation(
			values=np.asarray(values, dtype=np.float64),
			base_values=np.asarray(base_values, dtype=np.float64),
			data=np.ones(
				(len(self.test_dataset), self.num_sources),
				dtype=np.float64,
			),
			feature_names=self.feature_names,
		)

def analyze_and_save_source_groups(
	model: torch.nn.Module,
	train_dataset,
	test_dataset,
	*,
	background_batch_size: int = 32,
	compute_device: torch.device | str | None = None,
):
	"""Calculate and save source-level SHAP for every test sample."""
	analyzer = SourceGroupShapAnalyzer(
		model,
		train_dataset,
		test_dataset,
		background_batch_size=background_batch_size,
		compute_device=compute_device,
	)
	explanation = analyzer.explain_all()
	LOGGER.info('SHAP values shape: %s', explanation.values.shape)
	return explanation
