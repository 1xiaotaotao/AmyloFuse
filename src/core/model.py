import torch
import torch.nn as nn
import torch.nn.init as init
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from src.config import AmyloidConfig
from .attention import AttentionModule



class Model(nn.Module):
	def __init__(self,
	             config: AmyloidConfig,
	             dim,
	             encoder_dim=64,
	             n_classes=1,
	             drop=0.1):
		
		super(Model, self).__init__()
		self.feature_names = config.model.feature
		self.use_attention = config.model.use_attention
		self.residual_add = config.model.residual_add
		self.num_sources = len(self.feature_names)
		self.encoders = nn.ModuleDict(
			{name: nn.Linear(dim[name], encoder_dim) for name in self.feature_names}
		)
		self.drop = nn.Dropout(p=drop)
		self.source_attention = (
			AttentionModule(encoder_dim, self.num_sources)
			if self.use_attention else None
		)
		self.sequence_model_name = config.model.sequence_model
		self.model_dim = encoder_dim
		if self.sequence_model_name == 'bi-lstm':
			self.sequence_model = nn.LSTM(
				input_size=self.model_dim,
				hidden_size=self.model_dim // 2,
				num_layers=2,
				batch_first=True,
				bidirectional=True,
			)
		elif self.sequence_model_name == 'lstm':
			self.sequence_model = nn.LSTM(
				input_size=self.model_dim,
				hidden_size=self.model_dim,
				num_layers=2,
				batch_first=True,
			)
		elif self.sequence_model_name == 'transformer':
			self.sequence_model = TransformerEncoder(
				TransformerEncoderLayer(d_model=self.model_dim,
				                        nhead=4,
				                        batch_first=True,
				                        activation='gelu'),
				num_layers=1
			)
		else:
			raise NotImplementedError
		self.norm = nn.LayerNorm(self.model_dim)
		self.mlp_head = nn.Sequential(
			nn.Linear(self.model_dim, self.model_dim // 2),
			nn.GELU(),
			nn.Linear(self.model_dim // 2, self.model_dim),
			nn.GELU(),
			nn.Dropout(p=drop),
		)
		self.classifier = nn.Linear(self.model_dim, n_classes)
		
		self._init_weights()
		if self.source_attention is not None:
			self.source_attention.reset_parameters()
	
	def _init_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Linear):
				if m.weight is not None:
					init.normal_(m.weight, std=0.02)
				if m.bias is not None:
					init.constant_(m.bias, 0)
			elif isinstance(m, nn.LayerNorm):
				if m.weight is not None:
					init.constant_(m.weight, 1)
				if m.bias is not None:
					init.constant_(m.bias, 0)
			elif isinstance(m, nn.Conv1d):
				init.kaiming_normal_(m.weight, nonlinearity='relu')
				if m.bias is not None:
					init.constant_(m.bias, 0)
			elif isinstance(m, nn.LSTM):
				for name, parameter in m.named_parameters():
					if 'weight_ih' in name:
						init.xavier_uniform_(parameter)
					elif 'weight_hh' in name:
						init.orthogonal_(parameter)
					elif 'bias' in name:
						init.constant_(parameter, 0)
						if 'bias_ih' in name:
							hidden_size = parameter.numel() // 4
							parameter.data[hidden_size:2 * hidden_size].fill_(1.0)
	
	def sequence_model_process(self, encoded):
		residual = encoded
		recurrent = self.sequence_model(encoded)[0] \
			if self.sequence_model_name != 'transformer' else self.sequence_model(encoded)
		if self.residual_add:        
			encoded = self.norm(residual + self.drop(recurrent))   
		else:
			encoded = self.norm(self.drop(recurrent))
		return encoded
	
	def forward(self, x, masked_source=None):
		plm_encoded = [self.drop(self.encoders[name](x[name])) for name in self.feature_names]  # [(batch, 1, 64)]
            
		plm_encoded = torch.cat(plm_encoded, dim=1)  # (batch, n, 64)
        
		# plm_encoded = torch.cat(plm_encoded, dim=-1)  # (batch, 1, 192)
        
		weight = None      
        
		if self.source_attention is not None:
            
			plm_encoded, weight = self.source_attention(plm_encoded)
            
		if masked_source is not None:
                
			retained_indices = [
				index
				for index, source_name in enumerate(self.feature_names)
				if source_name != masked_source
			]
                
			plm_encoded = plm_encoded[:, retained_indices, :]
            
		plm_encoded = self.sequence_model_process(plm_encoded)  # (batch, n, 64)
        
		out = torch.mean(plm_encoded, dim=1)  # (batch, 64)
        
		final_out = self.mlp_head(out)  # (batch, 64)
        
		prob = self.classifier(final_out).squeeze(-1)  # (batch,)
        
		return prob, weight, final_out
