import logging
import os
from pathlib import Path
import torch
from Bio import SeqIO
from torch.utils.data import Dataset
from ..config import AmyloidConfig

class AmyloidDataset(Dataset):
	def __init__(self, config: AmyloidConfig, features, labels):
		self.feature_names = self.get_feature_names(config)
		self.features = features
		self.labels = labels
		self.features_dim = {
			name: features[name].shape[-1] for name in self.feature_names
		}

	@staticmethod
	def get_feature_names(config: AmyloidConfig):
		return config.model.feature

	@staticmethod
	def get_feature_prefix(pos_file, set_num):
		path = Path(pos_file)
		set_type = path.name.replace('_pos.fasta', '')
		if set_num is not None:
			prefix_path = str(Path('features') / path.parent.name / str(set_num) / set_type)
		else:
			prefix_path = str(Path('features') / path.parent.name / set_type)
		return prefix_path, set_type

	@staticmethod
	def get_labels(pos_file, neg_file):
		pos_count = sum(1 for _ in SeqIO.parse(pos_file, 'fasta'))
		neg_count = sum(1 for _ in SeqIO.parse(neg_file, 'fasta'))
		return torch.tensor([1] * pos_count + [0] * neg_count).float()

	@staticmethod
	def make_extractor(name, pos_file, neg_file, config: AmyloidConfig):
		paths = config.pretrained_models
		if name == 'prot-t5':
			from src.plm.extract_features_protT5 import ExtractFeaturesProtT5
			return ExtractFeaturesProtT5(pos_file, neg_file, paths.t5_path)
		if name == 'prot-bert':
			from src.plm.extract_features_bert import ExtractFeaturesBert
			return ExtractFeaturesBert(pos_file, neg_file, paths.bert_path)
		if name == 'esm-2':
			from src.plm.extract_features_esm_2 import ExtractFeaturesEsm2
			return ExtractFeaturesEsm2(pos_file, neg_file)
		from src.plm.extract_features_esm_1 import ExtractFeaturesEsm1
		return ExtractFeaturesEsm1(pos_file, neg_file)

	@classmethod
	def load_or_create(cls, pos_file, neg_file, set_num, config: AmyloidConfig, metadata_path):
		feature_names = cls.get_feature_names(config)
		prefix, set_type = cls.get_feature_prefix(pos_file, set_num)
		metadata = torch.load(metadata_path) if os.path.exists(metadata_path) else None
		same_seed = metadata is not None and metadata.get('seed') == config.seed
		labels = metadata['labels'] if same_seed else cls.get_labels(pos_file, neg_file)
		features = {}
		missing = []
		for name in feature_names:
			feature_path = f'{prefix}/{name}.pt'
			if same_seed and name in metadata.get('feature_names', []) and os.path.exists(feature_path):
				features[name] = torch.load(feature_path, mmap=True)
				print(f"Feature '{name}' shape: {features[name].shape}")
			else:
				missing.append(name)
		for name in missing:
			extractor = cls.make_extractor(name, pos_file, neg_file, config)
			features[name] = extractor.extract()
			feature_path = f'{prefix}/{name}.pt'
			os.makedirs(os.path.dirname(feature_path), exist_ok=True)
			torch.save(features[name], feature_path)
			features[name] = torch.load(feature_path, mmap=True)
			del extractor
			if torch.cuda.is_available():
				torch.cuda.empty_cache()
		if missing or not same_seed:
			os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
			saved_names = feature_names
			if same_seed:
				saved_names = list(
					dict.fromkeys(metadata.get('feature_names', []) + feature_names)
				)
			torch.save(
				{
					'seed': config.seed,
					'feature_names': saved_names,
					'labels': labels,
				},
				metadata_path,
			)
		logging.getLogger(cls.__name__).info(f'all {set_type} features saved')
		return cls(config, features, labels)

	def __len__(self):
		return len(self.labels)

	def __getitem__(self, index):
		return (
			{name: self.features[name][index] for name in self.feature_names},
			self.labels[index],
		)
