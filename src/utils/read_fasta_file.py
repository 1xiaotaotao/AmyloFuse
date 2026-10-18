import logging
from Bio import SeqIO
logger = logging.getLogger(__name__)


def read_sequences(fasta_file):
	logging.info(f'reading fasta file {fasta_file}')
	sequences = []
	with open(fasta_file, 'r') as f:
		for record in SeqIO.parse(f, 'fasta'):
			sequences.append(str(record.seq[:510]))
	return sequences

