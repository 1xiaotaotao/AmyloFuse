import torch
import torch.nn as nn
from copy import deepcopy
from pathlib import Path
from torch.utils.data import DataLoader
import torch.optim as optim
from .model import Model
from .shap import analyze_and_save_source_groups
from .dataset import AmyloidDataset
from ..utils.calculate_metrics import calculate_metrics
import numpy as np
import logging
from ..config import AmyloidConfig


generator = torch.Generator()

use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')


class EarlyStopping:
	def __init__(self, patience=10, delta=0.0):
		self.patience = patience
		self.delta = delta
		self.counter = 0
		self.best_score = None
		self.early_stop = False
		self.best_model_state = None
		self.best_epoch = None
	
	def __call__(self, score, model, epoch=None):
		if self.best_score is None:
			self.best_score = score
			self.save_checkpoint(model, epoch)
		elif self.best_score > score + self.delta:
			self.best_score = score
			self.save_checkpoint(model, epoch)
			self.counter = 0
		else:
			self.counter += 1
			if self.counter >= self.patience:
				self.early_stop = True
	
	def save_checkpoint(self, model, epoch=None):
		self.best_model_state = deepcopy(model.state_dict())
		self.best_epoch = epoch
	
	def load_best_model(self, model):
		if self.best_model_state is not None:
			model.load_state_dict(self.best_model_state)


class Train:
	def __init__(self, config: AmyloidConfig, dataset, set_num=None):
		self.dataset = dataset
		self.config = config
		train_pos_files = f'data/{self.dataset}/train_pos.fasta'
		train_neg_files = f'data/{self.dataset}/train_neg.fasta'
		test_pos_files = f'data/{self.dataset}/test_pos.fasta'
		test_neg_files = f'data/{self.dataset}/test_neg.fasta'
		self.train_set = AmyloidDataset.load_or_create(
			train_pos_files,
			train_neg_files,
			set_num,
			config,
			f'dataset/{dataset}/train_set.pt',
		)
		self.test_set = AmyloidDataset.load_or_create(
			test_pos_files,
			test_neg_files,
			set_num,
			config,
			f'dataset/{dataset}/test_set.pt',
		)
		self.training_config = config.training
		self.model = Model(config, self.train_set.features_dim).to(device)
		self.epochs = self.training_config.epochs
		self.batch_size = self.training_config.batch_size
		self.early_stopping = EarlyStopping(patience=self.training_config.early_stopping_patience)
		self.losses = None
		self.probs = None
		self.attn_weights = None
		self.visual_features = None
		self.shap_explanation = None
		self.masking_results = None       
		if set_num is not None:
			self.model_path = Path(self.config.save.model_path / str(set_num) / 'model_params.pth')
		else:
			self.model_path = Path(self.config.save.model_path / 'model_params.pth')
		self.model_path.parent.mkdir(parents=True, exist_ok=True)
		self.logger = logging.getLogger(self.__class__.__name__)
		generator.manual_seed(self.config.seed)
	
	
	def save_model(self, model):
		model_state = {
			name: value.detach().cpu().clone()
			for name, value in model.state_dict().items()
		}
		torch.save(model_state, self.model_path)
	
    
	def load_model(self, model):
		model_state = torch.load(self.model_path)
		model.load_state_dict(model_state)
        
        
	def train(self, model, train_loader, optimizer, scheduler):
		train_losses = []
		criterion = nn.BCEWithLogitsLoss()
		for epoch in range(self.epochs):
			model.train()
			total_loss = 0.0
			total_samples = 0
			for features, labels in train_loader:
				features = {name: value.to(device) for name, value in features.items()}
				labels = labels.to(device)
				optimizer.zero_grad()
				probs, _, _ = model(features)
				loss = criterion(probs, labels)
				batch_size = labels.size(0)
				total_loss += loss.item() * batch_size
				total_samples += batch_size
				loss.backward()
				torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
				optimizer.step()
			scheduler.step()
			train_loss = total_loss / total_samples
			train_losses.append(train_loss)
			self.logger.info('epoch %d/%d - train_loss: %.6f', epoch + 1, self.epochs, train_loss)
			self.early_stopping(train_loss, model, epoch=epoch + 1)
			if self.early_stopping.early_stop:
				self.logger.info(
					'early stopping; restoring epoch %s with train_loss %.6f',
					self.early_stopping.best_epoch,
					self.early_stopping.best_score,
				)
				break
		self.losses = train_losses

	
	def predict_probabilities(self, model, data_loader, masked_source=None):
		model.eval()
		y_true, y_pred_proba, attn_weights, visual_features = [], [], [], []
		with torch.no_grad():
			for features, labels in data_loader:
				features = {name: value.to(device) for name, value in features.items()}
				labels = labels.to(device)
				probs, weight, final_h = model(features, masked_source=masked_source)
				if masked_source is None:
					if weight is not None:
						attn_weights.extend(weight.cpu().numpy())
					visual_features.extend(final_h.cpu().numpy())
				probabilities = torch.sigmoid(probs)
				y_true.extend(labels.cpu().numpy())
				y_pred_proba.extend(probabilities.cpu().numpy())
		if masked_source is None:
			self.attn_weights = attn_weights
			self.visual_features = visual_features
		return np.asarray(y_true), np.asarray(y_pred_proba)
	
	def evaluate(self, model, data_loader, masked_source=None):
		y_true, y_pred_proba = self.predict_probabilities(model, data_loader, masked_source=masked_source)
		y_pred = (y_pred_proba > 0.5).astype(float)
		metrics = calculate_metrics(y_true, y_pred, y_pred_proba)
		if masked_source is None:
			self.probs = y_pred_proba.tolist()
		self.logger.info(f'metrics:{metrics}')
		return metrics
	
	def train_on_set(self):
		self.logger.info('Training on the full training set')
		model = deepcopy(self.model).to(device)
		optimizer = optim.AdamW(model.parameters(), lr=self.training_config.init_learning_rate, weight_decay=0.01)
		scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
		train_loader = DataLoader(
			self.train_set,
			batch_size=self.batch_size,
			shuffle=True,
			generator=generator,
		)
		self.train(model, train_loader, optimizer, scheduler)
		self.early_stopping.load_best_model(model)
		self.save_model(model)
	
	def predict_model(self):
		self.logger.info('predict test data on fitted model')
		test_loader = DataLoader(
			self.test_set,
			batch_size=self.batch_size,
			shuffle=False,
		)
		model = deepcopy(self.model).to(device)
		self.load_model(model)
		return self.evaluate(
			model,
			test_loader,
			masked_source=None,
		)
    
	def shap_analysis(self):
		self.logger.info('Shap analysis')
		model = deepcopy(self.model).to(device)
		self.load_model(model)
		self.shap_explanation = analyze_and_save_source_groups(
			model,
			self.train_set,
			self.test_set,
			background_batch_size=self.batch_size,
			compute_device=device,
		)
        
	def masking_analysis(self):
		self.logger.info('Evaluating inference-time source masking')
		test_loader = DataLoader(
			self.test_set,
			batch_size=self.batch_size,
			shuffle=False,
		)
		model = deepcopy(self.model).to(device)
		self.load_model(model)
		results = {}
		for source_name in model.feature_names:
			res = self.evaluate(
				model,
				test_loader,
				masked_source=source_name,
			)
			results[source_name] = res
		self.masking_results = results
	
	# return numpy arr
	def get_losses(self):
		if not self.losses:
			return None
		return np.array(self.losses)
	
	def get_probs(self):
		if not self.probs:
			return None
		return np.array(self.probs)
	
	def get_attn_weights(self):
		if not self.attn_weights:
			return None
		return np.array(self.attn_weights)
		
	def get_visual_features(self):
		if not self.visual_features:
			return None
		return np.array(self.visual_features)
    
	def get_source_shap(self):
		if self.shap_explanation is None:
			return None
		return self.shap_explanation
    

	def get_masking_results(self):
		if not self.masking_results:
			return None
		return self.masking_results
