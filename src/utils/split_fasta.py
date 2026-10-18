from Bio import SeqIO
import random
import logging
logger = logging.getLogger()


def split_fasta_8_2(dataset):
	pos = list(SeqIO.parse(f'data/{dataset}/pos.fasta', 'fasta'))
	neg = list(SeqIO.parse(f'data/{dataset}/neg.fasta', 'fasta'))
	
	
	
	
	for label, samples in [('pos', pos),('neg', neg)]:
		random.shuffle(samples)
		total = len(samples)
		split_point = int(total * 0.8)
		
		train = samples[:split_point]
		test = samples[split_point:]
		
		out_dir = f'data/{dataset}'
		SeqIO.write(train, f'{out_dir}/train_{label}.fasta', 'fasta')
		SeqIO.write(test, f'{out_dir}/test_{label}.fasta', 'fasta')
	logger.info('all files have split to train and test')
