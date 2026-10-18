import argparse
import json
import logging
import os
import pickle
import random
import sys
from datetime import datetime
from pathlib import Path, PosixPath

import numpy as np
import torch

from src.config import AmyloidConfig, load_config
from src.core.train import Train
from src.utils.calculate_metrics import summarize_results, summarize_results_dict
from src.utils.create_cross_beta_db_test import create_set_for_cross_beta
from src.utils.split_fasta import split_fasta_8_2

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['PYTHONHASHSEED'] = str(4294967291)


class PathEncoder(json.JSONEncoder):
	def default(self, obj):
		if isinstance(obj, (Path, PosixPath)):
			return str(obj)
		return super().default(obj)


def save_artifact(path, name, value):
	save_path = Path(path)
	save_path.mkdir(parents=True, exist_ok=True)
	if value is None:
		return
	try:
		if isinstance(value, dict):
			raise TypeError
		array = np.asarray(value)
		if array.dtype == object:
			raise TypeError
	except (TypeError, ValueError):
		with (save_path / f'{name}.pkl').open('wb') as file:
			pickle.dump(value, file)
			logging.info(f'Saved {name}')
	else:
		np.save(save_path / f'{name}.npy', array)
		logging.info(f'Saved {name}')

def save_trainer_artifacts(config, trainer):
	save_artifact(config.save.losses_path, 'losses', trainer.get_losses())
	save_artifact(config.save.probs_path, 'probs', trainer.get_probs())
	save_artifact(config.save.shap_path, 'shap', trainer.get_source_shap())
	save_artifact(
		config.save.masking_path,
	    'masking',
		trainer.get_masking_results()
	)
	save_artifact(
		config.save.visual_features_path,
		'visual_features',
		trainer.get_visual_features(),
	)
	save_artifact(
		config.save.attn_weights_path,
		'attn_weights',
		trainer.get_attn_weights(),
	)


def append_artifact(artifacts, value):
	if value is not None:
		artifacts.append(value)


def configure_logging():
	logging.basicConfig(
		level=logging.INFO,
		format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
		handlers=[
			logging.StreamHandler(sys.stdout),
			logging.FileHandler('AmyloFuse.log', encoding='utf-8'),
		],
	)
	logging.getLogger('urllib3').setLevel(logging.WARNING)
	logging.getLogger('matplotlib').setLevel(logging.WARNING)


def set_global_seed(seed):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed(seed)
		torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
	torch.use_deterministic_algorithms(True)



def run_amy(config: AmyloidConfig):
	dataset = config.dataset
	split_fasta_8_2(dataset)
	trainer = Train(config, dataset)
	trainer.train_on_set()  # train on the whole train set
	result = trainer.predict_model()
	trainer.shap_analysis()
	trainer.masking_analysis()
	save_trainer_artifacts(config, trainer)
	logging.getLogger(__name__).info('eval metrics of Amy: %s', result)
	return result




def run_cross_beta(config: AmyloidConfig):
	results = []
	losses_list = []
	probs_list = []
	visual_features_list = []
	attn_weights_list = []
	shap_explanation_list = []
	masking_results_list = []
	for set_num in range(1, 11):
		create_set_for_cross_beta(
			f'Cross_Beta_DB_test_set/test_set_{set_num}.csv'
		)
		trainer = Train(config, config.dataset, set_num)
		trainer.train_on_set()  # train on the whole train set
		result = trainer.predict_model()
		trainer.shap_analysis() 
		results.append(result)
		append_artifact(losses_list, trainer.get_losses())
		append_artifact(probs_list, trainer.get_probs())
		append_artifact(visual_features_list, trainer.get_visual_features())
		append_artifact(attn_weights_list, trainer.get_attn_weights())
		append_artifact(shap_explanation_list, trainer.get_source_shap())
	for set_num in range(1, 11):
		trainer = Train(config, config.dataset, set_num)
		trainer.masking_analysis()
		append_artifact(masking_results_list, trainer.get_masking_results())
	summary = summarize_results(results)
	masking_summary = summarize_results_dict(masking_results_list)
	save_artifact(config.save.losses_path, 'losses', losses_list or None)
	save_artifact(config.save.probs_path, 'probs', probs_list or None)
	save_artifact(config.save.visual_features_path, 'visual_features', visual_features_list or None)
	save_artifact(config.save.attn_weights_path, 'attn_weights', attn_weights_list or None)
	save_artifact(config.save.shap_path, 'shap', shap_explanation_list or None)
	save_artifact(config.save.masking_path, 'masking', masking_summary or None)
	logging.getLogger(__name__).info('eval metrics of Cross-Beta DB: %s', summary)
	return summary


def main(config_path='config.yaml'):
	configure_logging()
	config = load_config(config_path)
	set_global_seed(config.seed)
	logger = logging.getLogger(__name__)
	logger.info('=' * 60)
	logger.info('running time: %s', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
	from dataclasses import asdict
	logger.info(
		'effective config:\n%s',
		json.dumps(asdict(config), indent=2, ensure_ascii=False, cls=PathEncoder),
	)
	logger.info('=' * 60)
	if config.dataset == 'Amy':
		return run_amy(config)
	return run_cross_beta(config)


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Train and evaluate the amyloid classifier')
	parser.add_argument('--config', default='config.yaml', help='path to YAML configuration')
	args = parser.parse_args()
	main(args.config)
