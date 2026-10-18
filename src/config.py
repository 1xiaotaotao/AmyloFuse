import ast
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

def _value(text):
	text = text.strip()
	if text.lower() in ('true', 'false'):
		return text.lower() == 'true'
	if text.startswith('[') and text.endswith(']'):
		try:
			value = ast.literal_eval(text)
		except (SyntaxError, ValueError):
			pass
		else:
			if isinstance(value, list):
				return value
	try:
		return int(text)
	except ValueError:
		try:
			return float(text)
		except ValueError:
			return text.strip('\'\"')


def _simple_yaml(text):
	'''Small fallback parser for this project's two-level config file.'''
	config = {}
	section = config
	for raw_line in text.splitlines():
		line = raw_line.split('#', 1)[0].rstrip()
		if not line.strip():
			continue
		indent = len(line) - len(line.lstrip())
		key, value = line.strip().split(':', 1)
		if indent == 0:
			if value.strip():
				config[key] = _value(value)
			else:
				config[key] = {}
				section = config[key]
		else:
			section[key] = _value(value)
	return config

def _validate_fields(config, expected_fields, section):
	if not isinstance(config, Mapping):
		raise ValueError(f'{section} must be a mapping')
	missing_fields = sorted(expected_fields - config.keys())
	if missing_fields:
		missing_field_list = ', '.join(missing_fields)
		raise ValueError(f'{section} is missing required fields: {missing_field_list}')
	unknown_fields = sorted(config.keys() - expected_fields)
	if unknown_fields:
		unknown_field_list = ', '.join(unknown_fields)
		raise ValueError(f'{section} contains unknown fields: {unknown_field_list}')

@dataclass
class PretrainedModelsConfig:
	t5_path: Path
	bert_path: Path


@dataclass
class ModelConfig:
	feature: list
	use_attention: bool
	sequence_model: str
	residual_add: bool



@dataclass
class TrainingConfig:
	batch_size: int
	init_learning_rate: float
	epochs: int
	early_stopping_patience: int


@dataclass
class SaveConfig:
	model_path: Path
	shap_path: Path
	losses_path: Path
	probs_path: Path
	attn_weights_path: Path
	visual_features_path: Path
	masking_path: Path

	def initialize(self, dataset, seed):
		path_suffix = Path(dataset) / str(seed)
		self.model_path = self.model_path / path_suffix
		self.shap_path = self.shap_path / path_suffix
		self.losses_path = self.losses_path / path_suffix
		self.probs_path = self.probs_path / path_suffix
		self.attn_weights_path = self.attn_weights_path / path_suffix
		self.visual_features_path = self.visual_features_path / path_suffix
		self.masking_path = self.masking_path / path_suffix


@dataclass
class AmyloidConfig:
	dataset: str
	seed: int
	pretrained_models: PretrainedModelsConfig
	model: ModelConfig
	training: TrainingConfig
	save: SaveConfig

	def __post_init__(self):
		if not isinstance(self.pretrained_models, PretrainedModelsConfig):
			raise ValueError('pretrained_models must be a PretrainedModelsConfig')
		if not isinstance(self.model, ModelConfig):
			raise ValueError('model must be a ModelConfig')
		if not isinstance(self.training, TrainingConfig):
			raise ValueError('training must be a TrainingConfig')
		if not isinstance(self.save, SaveConfig):
			raise ValueError('save must be a SaveConfig')
		self.save.initialize(self.dataset, self.seed)

	@classmethod
	def from_mapping(cls, config):
		_validate_fields(
			config,
			{'dataset', 'seed', 'pretrained_models', 'model', 'training', 'save'},
			'config',
		)
		pretrained_models = config['pretrained_models']
		_validate_fields(
			pretrained_models,
			{'t5_path', 'bert_path'},
			'pretrained_models',
		)
		model = config['model']
		_validate_fields(
			model,
			{
				'feature',
				'use_attention',
				'sequence_model',
				'residual_add',
			},
			'model',
		)
		training = config['training']
		_validate_fields(
			training,
			{
				'batch_size',
				'init_learning_rate',
				'epochs',
				'early_stopping_patience',
			},
			'training',
		)
		save = config['save']
		_validate_fields(
			save,
			{
				'model_path',
				'shap_path',               
				'losses_path',
				'probs_path',
				'attn_weights_path',
				'visual_features_path',
				'masking_path',
			},
			'save',
		)
		return cls(
			dataset=config['dataset'],
			seed=config['seed'],
			pretrained_models=PretrainedModelsConfig(**pretrained_models),
			model=ModelConfig(**model),
			training=TrainingConfig(**training),
			save=SaveConfig(**save),
		)

def load_config(path: str | Path = 'config.yaml') -> AmyloidConfig:
	text = Path(path).read_text(encoding='utf-8')
	try:
		import yaml
		config = yaml.safe_load(text)
	except ImportError:
		config = _simple_yaml(text)
	return AmyloidConfig.from_mapping(config)
