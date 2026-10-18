from src.utils.read_fasta_file import read_sequences

import esm
import torch
from tqdm import tqdm
import logging

use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')


class ExtractFeaturesEsm1:
	def __init__(self, pos_file, neg_file):
		self.input_pos_file = pos_file
		self.input_neg_file = neg_file
		self.model = None
		self.alphabet = None
		self.batch_size = 8
		self.logger = logging.getLogger(self.__class__.__name__)
	
	def extract_process(self, sequences):
		model = self.model.to(device)
		converter = self.alphabet.get_batch_converter()
		features = []
		for i in tqdm(range(0, len(sequences), self.batch_size)):
			batch_sequences = sequences[i:i + self.batch_size]
			batch_data = [('', seq) for seq in batch_sequences]
			_, _, tokens = converter(batch_data)
			tokens = tokens.to(device)
			with torch.no_grad():
				results = model(tokens, repr_layers=[34])
			token_representations = results['representations'][34]
			sequence_representations = []
			for j, seq in enumerate(batch_sequences):
				sequence_representations.append(token_representations[j, 1: len(seq) + 1].mean(0))
			features.extend(sequence_representations)
		return torch.stack(features, dim=0)
	
	def extract(self):
		self.logger.info('extracting esm1 features')
		self.model, self.alphabet = esm.pretrained.esm1_t34_670M_UR50S()
		self.model.eval()
		pos_sequences = read_sequences(self.input_pos_file)
		neg_sequences = read_sequences(self.input_neg_file)
		self.logger.info('extracting features of pos by esm1')
		pos_features = self.extract_process(pos_sequences)
		self.logger.info('extracting features of neg by esm1')
		neg_features = self.extract_process(neg_sequences)
		return torch.cat([pos_features, neg_features], dim=0).unsqueeze(dim=1)  # (n, 1, 1280)





