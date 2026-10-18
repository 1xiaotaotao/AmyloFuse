import torch
from src.utils.read_fasta_file import read_sequences
from tqdm import tqdm
import re
from transformers import T5Tokenizer, T5EncoderModel
import logging

use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')


class ExtractFeaturesProtT5:
	def __init__(self, input_pos_file, input_neg_file, model_path):
		self.input_pos_file = input_pos_file
		self.input_neg_file = input_neg_file
		self.model_path = str(model_path)
		self.model = None
		self.tokenizer = None
		self.batch_size = 8
		self.logger = logging.getLogger(self.__class__.__name__)
	
	def extract_process(self, sequences):
		model = self.model.to(device)
		features = []
		for i in tqdm(range(0, len(sequences), self.batch_size)):
			batch_sequences = sequences[i:i + self.batch_size]
			filled_sequences = [' '.join(list(re.sub(r'[UZOB]', 'X', str(sequence)))) for sequence in batch_sequences]
			output = self.tokenizer.batch_encode_plus(filled_sequences, add_special_tokens=True, padding='longest',
			                                          return_tensors='pt')
			ids, attention_mask = output['input_ids'].to(device), output['attention_mask'].to(device)
			with torch.no_grad():
				result = model(ids, attention_mask)
			embeddings = result.last_hidden_state
			sequence_representations = []
			for j, seq in enumerate(batch_sequences):
				sequence_representations.append(embeddings[j, :len(seq)].mean(0))
			features.extend(sequence_representations)
		return torch.stack(features, dim=0)
	
	def extract(self):
		self.logger.info('extracting Prot-T5 features')
		self.tokenizer = T5Tokenizer.from_pretrained(self.model_path, local_files_only=True, legacy=False)
		self.model = T5EncoderModel.from_pretrained(self.model_path, local_files_only=True)
		self.model.eval()
		pos_sequences = read_sequences(self.input_pos_file)
		neg_sequences = read_sequences(self.input_neg_file)
		self.logger.info('extracting features of pos by Prot-T5')
		pos_features = self.extract_process(pos_sequences)
		self.logger.info('extracting features of neg by Prot-T5')
		neg_features = self.extract_process(neg_sequences)
		return torch.cat([pos_features, neg_features], dim=0).unsqueeze(dim=1)  # (n, 1, 1024)
