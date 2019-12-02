
from __future__ import division

from gensim.models import KeyedVectors
from collections import Counter
from StringIO import StringIO
from article_repr import *
import pickle, random, nltk, gensim
import numpy as np


def count_article_type(articles):
	step = 0
	method = 0
	part = 0
	na = 0
	for a in articles:
		if a.format=='steps':
			step += 1
		elif a.sub_sec_type=='METHODS':
			method += 1
		elif a.sub_sec_type=='STEPS':
			part += 1
		elif a.sub_sec_type=='NA':
			na += 1
	assert step + method + part + na == len(articles)
	# print step, method, part, na
	print step/len(articles), method/len(articles), part/len(articles), na/len(articles)

def load_split_articles(parsed_article_path):
	random.seed(0)
	parsed_articles = pickle.load(open(parsed_article_path, 'rb'))
	all_article_titles = parsed_articles.keys()
	random.shuffle(all_article_titles)

	idx1 = int(0.8*len(all_article_titles))
	idx2 = int(0.9*len(all_article_titles))

	train_article_titles = all_article_titles[:idx1]
	dev_article_titles = all_article_titles[idx1:idx2]
	test_article_titles = all_article_titles[idx2:]

	train_articles = [parsed_articles[t] for t in train_article_titles]
	dev_articles = [parsed_articles[t] for t in dev_article_titles]
	test_articles = [parsed_articles[t] for t in test_article_titles]

	split_articles = {'train': train_articles, 'dev': dev_articles, 'test': test_articles}
	return split_articles

def to_unicode(s):
	if isinstance(s, unicode):
		return s
	return unicode(s, 'utf-8')

def rand_embedding(dim, norm=8):
	vec = np.array([random.random()-0.5 for _ in xrange(dim)])
	vec = vec / np.linalg.norm(vec) * norm
	return vec

class WordIndexer:	
	'''the class for converting between word, index, and embedding'''
	def __init__(self, split_articles, pretrained_embedding_fn, occur_cutoff):
		'''
		learn vocabulary using split_articles, removing words that occur less than occur_cutoff times
		use the pretrained_embedding_fn to initialize an embedding array with kept words and special
		characters <pad>, <unk>, <start> and <stop>, whose indices are 0, 1, 2, and 3
		'''
		self.split_articles = split_articles
		all_text_buf = StringIO()
		for ds_label in ['train', 'dev', 'test']:
			articles = split_articles[ds_label]
			for a in articles:
				all_text_buf.write(to_unicode(a.title) + u' ')
				if a.format=='steps':
					for s in a.steps:
						all_text_buf.write(to_unicode(s.short) + u' ')
				else:
					for sub in a.subsections:
						all_text_buf.write(to_unicode(sub.title) + u' ')
						for s in sub.steps:
							all_text_buf.write(to_unicode(s.short) + u' ')
		all_text = all_text_buf.getvalue().lower()
		all_text_buf.close()
		tokens = nltk.word_tokenize(all_text)
		token_counter = Counter(tokens)
		keep_word_count = []

		embedding = KeyedVectors.load(pretrained_embedding_fn)
		dim = len(embedding['the'])
		for k, v in token_counter.iteritems():
			if v < occur_cutoff:
				continue
			if k not in embedding:
				continue
			keep_word_count.append((v, k))
		keep_word_count = sorted(keep_word_count)[::-1]
		keep_words_sorted = ['<pad>', '<unk>', '<start>', '<stop>'] + [e[1] for e in keep_word_count]
		
		self.idx2word = keep_words_sorted
		self.word2idx = {w:idx for idx, w in enumerate(keep_words_sorted)}

		self.emb_matrix = np.zeros((len(keep_words_sorted), dim), 'float32')
		self.emb_matrix[1] = rand_embedding(300) # <unk>
		self.emb_matrix[2] = rand_embedding(300) # <start>
		self.emb_matrix[3] = rand_embedding(300) # <stop>
		for i in xrange(4, len(keep_words_sorted)):
			self.emb_matrix[i] = embedding[keep_words_sorted[i]]

	def num_emb_dim(self):
		return self.emb_matrix.shape[1]

	def num_words(self):
		return self.emb_matrix.shape[0]

	def get_pretrained_embedding(self):
		'''
		return the pre-trained embedding as an N x d array, 
		where N is the number of words, and d is the embedding dim
		the first four vectors are the embeddings for <pad>, <unk>, <start>, and <stop>
		'''
		return self.emb_matrix

	def embedding_of(self, word):
		'''return the embedding of the given word, or the embedding of the <unk>'''
		print 'WARNING: You probably shouldn\'t be calling this function'
		return self.emb_matrix[self.index_of(word)]

	def index_of(self, word):
		'''return the index of the given word, or the index of <unk>'''
		# if word=='<pad>':
		# 	return 0
		# else:
		# 	return 666
		if word not in self.word2idx:
			return self.word2idx['<unk>']
		else:
			return self.word2idx[word]

	def word_of(self, index):
		'''return the word for the given index'''
		return self.idx2word[index]

class ParentChildDataset:
	'''
	the dataset for training parent-child relation classification
	currently we have: 
	* article title --> steps (for flat articles)
	* article title --> steps (for 2-level hierarchies)
	* sub_sec title --> steps (for 2-level hierarchies)
	'''

	def __init__(self, split_articles):
		self.split_articles = split_articles
		self.data_dict = dict() # from parent string to a set of child strings
		self.data_pairs = dict() # a list of (parent, child) pairs (with duplicate parent)
		self.all_childs = dict() # a list of all child strings
		for ds_label in ['train', 'dev', 'test']:
			cur_data_pairs = []
			cur_data_dict = dict()
			articles = split_articles[ds_label]
			for a in articles:
				if a.format=='steps':
					cur_data_pairs += [(a.title, s.short, s.full) for s in a.steps]
					if a.title not in cur_data_dict:
						cur_data_dict[a.title] = [s.short for s in a.steps]
					else:
						cur_data_dict[a.title] += [s.short for s in a.steps]
				else:
					subsec_titles = [s.title for s in a.subsections]
					subsec_stepss = [[st.short for st in s.steps] for s in a.subsections]
					subsec_stepss_full = [[st.full for st in s.steps] for s in a.subsections]
					assert len(subsec_titles)==len(subsec_stepss)
					for sub_t, sub_sts, sub_sts_full in zip(subsec_titles, subsec_stepss, subsec_stepss_full):
						assert len(sub_sts)==len(sub_sts_full)
						cur_data_pairs += [(sub_t, sub_st, sub_st_full) for sub_st, sub_st_full in zip(sub_sts, sub_sts_full)]
						if sub_t not in cur_data_dict:
							cur_data_dict[sub_t] = sub_sts
						else:
							cur_data_dict[sub_t] += sub_sts
					all_steps = sum(subsec_stepss, [])
					all_steps_full = sum(subsec_stepss_full, [])
					cur_data_pairs += [(a.title, s, s_full) for (s, s_full) in zip(all_steps, all_steps_full)]
					if a.title not in cur_data_dict:
						cur_data_dict[a.title] = all_steps
					else:
						cur_data_dict[a.title] += all_steps
			for k, v in cur_data_dict.iteritems():
				cur_data_dict[k] = set(v)
			cur_data_pairs = [p for p in cur_data_pairs if p[0]!='' and p[1]!='' and p[2]!='']
			for p in cur_data_pairs:
				assert isinstance(p[0], basestring) and isinstance(p[1], basestring) and isinstance(p[2], basestring)
			self.data_pairs[ds_label] = cur_data_pairs
			self.data_dict[ds_label] = cur_data_dict
			self.all_childs[ds_label] = [(p[1], p[2]) for p in cur_data_pairs]

	def sample_pos_data(self, ds_label, N):
		return random.sample(self.data_pairs[ds_label], N)

	def sample_neg_data(self, ds_label, N):
		'''get the parent by sampling positive data, and then substitute in negative data'''
		pos_data = self.sample_pos_data(ds_label, N)
		parents, _, _ = zip(*pos_data)
		childs = []
		fulls = []
		for p in parents:
			c, full = random.choice(self.all_childs[ds_label])
			while c in self.data_dict[ds_label][p]:
				c, full = random.choice(self.all_childs[ds_label])
			childs.append(c)
			fulls.append(full)
		return zip(parents, childs, fulls)

	def sample_batch_data(self, ds_label, N):
		'''sample half positive and half negative data'''
		assert N%2==0, 'batch size must be even'
		pos_data = self.sample_pos_data(ds_label, int(N/2))
		pos_labels = [1] * int(N/2)
		neg_data = self.sample_neg_data(ds_label, int(N/2))
		neg_labels = [0] * int(N/2)
		return pos_data + neg_data, np.array(pos_labels + neg_labels).astype('int64')

class StepRankingDataset:
	'''
	the dataset for training step rankings
	pairwise ranking data is returned. currently we have
	* (article title, step 1, step 2) from a flat article, where step 1 occurs before step 2
	* (article title, step 1, step 2) from a 2-level hierarchy article, where step 1 occurs before step 2 in the same subsection
	* (sub_sec title, step 1, step 2) from a subsection, where step 1 occurs before step 2
	each article or subsection has the same probability of being sampled
	'''
	def __init__(self, split_articles):
		self.split_articles = split_articles
		self.rankings = dict() # a dict from title to a list of possible lists of steps
		for ds_label in ['train', 'dev', 'test']:
			cur_rankings = dict()
			articles = split_articles[ds_label]
			for a in articles:
				if a.format=='steps':
					steps = [(s.short, s.full) for s in a.steps]
					self.add_rankings(cur_rankings, a.title, steps)
				else:
					subsec_titles = [s.title for s in a.subsections]
					subsec_stepss = [[(st.short, st.full) for st in s.steps] for s in a.subsections]
					assert len(subsec_titles)==len(subsec_stepss)
					for sub_t, sub_sts in zip(subsec_titles, subsec_stepss):
						self.add_rankings(cur_rankings, sub_t, sub_sts)
						self.add_rankings(cur_rankings, a.title, sub_sts)
			if '' in cur_rankings:
				del cur_rankings['']
			self.rankings[ds_label] = cur_rankings

	def add_rankings(self, rankings, title, steps):
		if title not in rankings:
			rankings[title] = []
		rankings[title].append(steps)

	def sample_ranking_data(self, ds_label, N):
		'''return a list of N tuples in the format of (title, step 1, step 2)'''
		data = []
		while len(data)!=N:
			title = random.choice(self.rankings[ds_label].keys())
			assert title!='', 'Empty titles should have already been removed'
			steps = random.choice(self.rankings[ds_label][title])
			n = len(steps)
			if n==1:
				continue
			i, j = random.sample(range(n), 2)
			mi = min(i, j)
			ma = max(i, j)
			if steps[mi]=='' or steps[ma]=='':
				continue
			data.append((title, steps[mi], steps[ma]))
		return data

	def get_random_article(self, ds_label):
		'''
		return the title of a random article (or subsection) and the list of ordered steps
		'''
		title = random.choice(self.rankings[ds_label].keys())
		assert title!='', 'Empty titles should have already been removed'
		steps = random.choice(self.rankings[ds_label][title])
		return title, steps
