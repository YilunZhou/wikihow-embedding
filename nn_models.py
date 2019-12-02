
from __future__ import division

import torch
from torch import nn
import torch.nn.utils.rnn as rnn_utils
import nltk, random

class RandEncoder(nn.Module):
	'''a place-holding random encoder'''
	def __init__(self, encoding_dim, gpu):
		super(RandEncoder, self).__init__()
		self.gpu = gpu
		self.encoding_dim = encoding_dim
		self.fake_param = nn.Linear(1,1)

	def forward(self, input):
		encoding = torch.rand((len(input), self.encoding_dim))
		if self.gpu:
			encoding = encoding.cuda()
		return encoding

	def check_padding_unchanged(self):
		pass

class AugmentedEmbedding(nn.Module):
	def __init__(self, word_indexer, freeze_embedding, gpu):
		super(AugmentedEmbedding, self).__init__()
		self.gpu = gpu
		self.word_indexer = word_indexer
		self.freeze_embedding = freeze_embedding
		if freeze_embedding:
			# trainable embedding for the four special tokens <pad>, <unk>, <start>, <stop>
			weight = torch.tensor(word_indexer.emb_matrix)
			if gpu:
				weight = weight.cuda()
			self.special_embedding = nn.Embedding.from_pretrained(weight, freeze=False)
			# non-trainable embedding for all tokens, including the four special tokens (though they will not be accessed)
			weight = torch.tensor(word_indexer.emb_matrix)
			if gpu:
				weight = weight.cuda()
			self.normal_embedding = nn.Embedding.from_pretrained(weight, freeze=True)
		else:
			weight = torch.tensor(word_indexer.emb_matrix)
			if gpu:
				weight = weight.cuda()
			self.embedding = nn.Embedding.from_pretrained(weight, freeze=False)

	def forward(self, input):
		'''input is a tensor, return output of the same shape but with one more dimension of embedding'''
		if self.freeze_embedding:
			special = self.special_embedding(input).permute(2,1,0) * (input<4).float().permute(1,0)
			normal = self.normal_embedding(input).permute(2,1,0) * (input>=4).float().permute(1,0)
			padded_embeddings = (special + normal).permute(2,1,0)
		else:
			padded_embeddings = self.embedding(input)
		return padded_embeddings

class Encoder(nn.Module):
	'''an encoder that transforms from sentence to hidden representation with LSTM network'''
	def __init__(self, word_indexer, freeze_embedding, encoding_dim, gpu):
		super(Encoder, self).__init__()
		assert word_indexer.index_of('<pad>')==0, '<pad> must have an word index of 0'
		self.gpu = gpu
		self.embedding = AugmentedEmbedding(word_indexer, freeze_embedding, gpu)
		self.lstm = nn.LSTM(input_size=word_indexer.num_emb_dim(), hidden_size=encoding_dim)
		self.word_indexer = word_indexer

	def forward(self, input):
		'''
		input is a list of strings
		return the array of encodings of shape batch_size x embedding_dim
		'''
		tokenss = [nltk.word_tokenize(l.lower()) for l in input] # a list of list of tokens
		idxss = [torch.tensor([self.word_indexer.index_of(w) for w in tks]) for tks in tokenss]
		sent_lens = map(len, idxss)
		assert min(sent_lens) > 0 # no empty sentences
		sent_end_idxs = [l-1 for l in sent_lens]
		padded_idx_array = rnn_utils.pad_sequence(idxss)
		if self.gpu:
			padded_idx_array = padded_idx_array.cuda()
		padded_embeddings = self.embedding(padded_idx_array)
		output, (h_n, c_n) = self.lstm(padded_embeddings)
		selected_output = output[sent_end_idxs, range(len(input)), :]
		return selected_output

	def check_padding_unchanged(self):
		print 'checking padding embedding...'
		idx = torch.tensor([[0]])
		if self.gpu:
			idx = idx.cuda()
		pad_sum = self.embedding(idx).detach().cpu().numpy().__abs__().sum()
		assert pad_sum < 1e-6, pad_sum

class BagEncoder(nn.Module):
	'''an encoder that transforms from sentence to representation by a simple embedding average'''
	def __init__(self, word_indexer, freeze_embedding, gpu):
		super(BagEncoder, self).__init__()
		assert word_indexer.index_of('<pad>')==0, '<pad> must have an word index of 0'
		self.gpu = gpu
		self.embedding = AugmentedEmbedding(word_indexer, freeze_embedding, gpu)

	def forward(self, input):
		'''
		input is a list of strings
		return the array of encodings of shape batch_size x embedding_dim
		'''
		tokenss = [nltk.word_tokenize(l.lower()) for l in input] # a list of list of tokens
		idxss = [torch.tensor([self.word_indexer.index_of(w) for w in tks]) for tks in tokenss]
		sent_lens = map(len, idxss)
		assert min(sent_lens) > 0 # no empty sentences
		if self.gpu:
			idxss = [idxs.cuda() for idxs in idxss]
		embeddings = [self.embedding(idxs) for dixs in idxss]
		avg_embeddings = [emb.mean(dim=0) for emb in embeddings]
		return torch.stack(avg_embeddings)

	def check_padding_unchanged(self):
		print 'checking padding embedding...'
		idx = torch.tensor(0)
		if self.gpu:
			idx = idx.cuda()
		pad_sum = self.embedding(idx).detach().cpu().numpy().__abs__().sum()
		assert pad_sum < 1e-6, pad_sum

class ParentChildClassifier(nn.Module):
	def __init__(self, parent_dim, child_short_dim, child_full_dim, hidden_dim):
		super(ParentChildClassifier, self).__init__()
		if child_full_dim is not None:
			self.hidden = nn.Linear(parent_dim+child_short_dim+child_full_dim, hidden_dim)
		else:
			self.hidden = nn.Linear(parent_dim+child_short_dim, hidden_dim)
		self.relu = nn.ReLU()
		self.out = nn.Linear(hidden_dim, 2)
		self.logsoftmax = nn.LogSoftmax(dim=1)

	def forward(self, parent_encodings, child_short_encodings, child_full_encodings):
		if child_full_encodings is not None:
			encodings = torch.cat([parent_encodings, child_short_encodings, child_full_encodings], dim=1)
		else:
			encodings = torch.cat([parent_encodings, child_short_encodings], dim=1)
		log_probs = self.logsoftmax(self.out(self.relu(self.hidden(encodings))))
		return log_probs

class StepRankerMargin(nn.Module):
	def __init__(self, parent_dim, child_short_dim, child_full_dim, hidden_dim):
		super(StepRankerMargin, self).__init__()
		if child_full_dim is not None:
			self.hidden = nn.Linear(parent_dim+child_short_dim+child_full_dim, hidden_dim)
		else:
			self.hidden = nn.Linear(parent_dim+child_short_dim, hidden_dim)
		self.relu = nn.ReLU()
		self.out = nn.Linear(hidden_dim, 1) # this layer contains the ranker w

	def forward(self, parent_encodings, child_short_encodings_1, child_short_encodings_2, child_full_encodings_1, child_full_encodings_2):
		'''return the ranking scores for child 1 and child 2 in which child 1 should come temporally before child 2'''
		assert (child_full_encodings_1 is None) == (child_full_encodings_2 is None)
		if child_full_encodings_1 is not None:
			encodings_1 = torch.cat([parent_encodings, child_short_encodings_1, child_full_encodings_1], dim=1)
			encodings_2 = torch.cat([parent_encodings, child_short_encodings_2, child_full_encodings_2], dim=1)
		else:
			encodings_1 = torch.cat([parent_encodings, child_short_encodings_1], dim=1)
			encodings_2 = torch.cat([parent_encodings, child_short_encodings_2], dim=1)
		score_1 = self.out(self.relu(self.hidden(encodings_1)))
		score_2 = self.out(self.relu(self.hidden(encodings_2)))
		return score_1.view(score_1.numel()), score_2.view(score_2.numel())

	def score(self, parent_encodings, child_encodings):
		'''return the score of multiple child encodings each with respective parent encoding'''
		encodings = torch.cat([parent_encodings, child_encodings], dim=1)
		scores = self.out(self.relu(self.hidden(encodings)))
		return scores.view(scores.numel())

class StepRankerLogistic3(nn.Module):
	'''a logistic ranker that includes a don't care token'''
	def __init__(self, parent_dim, child_short_dim, child_full_dim, hidden_dim):
		super(StepRankerLogistic3, self).__init__()
		if child_full_dim is not None:
			self.hidden = nn.Linear(parent_dim+child_short_dim+child_full_dim+child_short_dim+child_full_dim, hidden_dim)
		else:
			self.hidden = nn.Linear(parent_dim+child_short_dim+child_short_dim, hidden_dim)
		self.relu = nn.ReLU()
		self.out = nn.Linear(hidden_dim, 3)
		self.logsoftmax = nn.LogSoftmax(dim=1)

	def forward(self, parent_encodings, child_short_encodings_1, child_short_encodings_2, child_full_encodings_1, child_full_encodings_2):
		'''return the predicted probability that child 1 should come temporally before child 2'''
		assert (child_full_encodings_1 is None) == (child_full_encodings_2 is None)
		if child_full_encodings_1 is not None:
			input = torch.cat([parent_encodings, child_short_encodings_1, child_full_encodings_1, 
				child_short_encodings_2, child_full_encodings_2], dim=1)
		else:
			input = torch.cat([parent_encodings, child_short_encodings_1, child_short_encodings_2], dim=1)
		return self.logsoftmax(self.out(self.relu(self.hidden(input))))

class StepRankerLogistic(nn.Module):
	'''a logistic ranker'''
	def __init__(self, parent_dim, child_short_dim, child_full_dim, hidden_dim):
		super(StepRankerLogistic, self).__init__()
		if child_full_dim is not None:
			self.hidden = nn.Linear(parent_dim+child_short_dim+child_full_dim+child_short_dim+child_full_dim, hidden_dim)
		else:
			self.hidden = nn.Linear(parent_dim+child_short_dim+child_short_dim, hidden_dim)
		self.relu = nn.ReLU()
		self.out = nn.Linear(hidden_dim, 2)
		self.logsoftmax = nn.LogSoftmax(dim=1)

	def forward(self, parent_encodings, child_short_encodings_1, child_short_encodings_2, child_full_encodings_1, child_full_encodings_2):
		'''return the predicted probability that child 1 should come temporally before child 2'''
		assert (child_full_encodings_1 is None) == (child_full_encodings_2 is None)
		if child_full_encodings_1 is not None:
			input = torch.cat([parent_encodings, child_short_encodings_1, child_full_encodings_1, 
				child_short_encodings_2, child_full_encodings_2], dim=1)
		else:
			input = torch.cat([parent_encodings, child_short_encodings_1, child_short_encodings_2], dim=1)
		return self.logsoftmax(self.out(self.relu(self.hidden(input))))
