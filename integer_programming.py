
from __future__ import division

import torch
import numpy as np
import cvxpy as cp
import argparse, random, pickle
from nn_models import *
from article_repr import *
from load_data import load_split_articles, WordIndexer, ParentChildDataset, StepRankingDataset
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def add_argument(parser, argument, default):
	typ = type(default)
	if typ==bool:
		parser.add_argument('--'+argument, dest=argument.replace('-', '_'), action='store_true')
		parser.add_argument('--no-'+argument, dest=argument.replace('-', '_'), action='store_false')
		parser.set_defaults(**{argument.replace('-', '_'): default})
	else:
		parser.add_argument('--'+argument, nargs='?', type=type(default), default=default)

arg_parser = argparse.ArgumentParser()
add_argument(arg_parser, 'article-path', 'articles/parsed_articles.pkl')
add_argument(arg_parser, 'embedding-path', 'glove/glove.model')
add_argument(arg_parser, 'occur-cutoff', 5)
add_argument(arg_parser, 'model-folder', 'models')
add_argument(arg_parser, 'context-encoder', 'LSTM')
add_argument(arg_parser, 'child-short-enc-template', 'child_short_encoder_%i.model')
add_argument(arg_parser, 'child-full-enc-template', 'child_full_encoder_%i.model')
add_argument(arg_parser, 'parent-enc-template', 'parent_encoder_%i.model')
add_argument(arg_parser, 'pc-classifier-template', 'pc_classifier_%i.model')
add_argument(arg_parser, 'step-ranker-template', 'step_ranker_%i.model')
add_argument(arg_parser, 'iter', -1)

args = arg_parser.parse_args()
for k, v in vars(args).iteritems():
	print k, '=', v

assert args.iter!=-1, 'Must specify --iter argument'

split_articles = pickle.load(open('articles/split_articles.pkl', 'rb'))

print 'loading word indexer'
word_indexer = WordIndexer(split_articles, args.embedding_path, args.occur_cutoff)
print 'loading parent child dataset'
parent_child_dataset = ParentChildDataset(split_articles)
print 'loading step ranking dataset'
step_ranking_dataset = StepRankingDataset(split_articles)

print 'loading models'

parent_encoder = torch.load(args.model_folder+'/'+args.parent_enc_template%args.iter)
parent_encoder.lstm.flatten_parameters()
child_short_encoder = torch.load(args.model_folder+'/'+args.child_short_enc_template%args.iter)
child_short_encoder.lstm.flatten_parameters()
assert args.context_encoder in ['None', 'LSTM', 'Bag'], 'Context encoder must be one of None, LSTM or Bag'
if args.context_encoder=='None':
	child_full_encoder = None
else:
	child_full_encoder = torch.load(args.model_folder+'/'+args.child_full_enc_template%args.iter)
	child_full_encoder.lstm.flatten_parameters()
pc_classifier = torch.load(args.model_folder+'/'+args.pc_classifier_template%args.iter)
step_ranker = torch.load(args.model_folder+'/'+args.step_ranker_template%args.iter)

def solve(W, max_dc):
	N = W.shape[0]
	X_list = []
	X = [[None for _ in xrange(N)] for _ in xrange(N)]
	for i in xrange(N):
		for j in xrange(N):
			if i==j:
				continue
			X[i][j] = cp.Variable(boolean=True)
			X_list.append(X[i][j])
	constraints = []
	for i in xrange(N):
		for j in xrange(N):
			if i==j:
				continue
			constraints.append(X[i][j]+X[j][i]<=1)
	for i in xrange(N):
		for j in xrange(N):
			for k in xrange(N):
				if i==j or i==k or j==k:
					continue
				constraints.append(X[i][j]+X[j][k]-X[i][k]-1<=0)
	constraints.append(sum(X_list)>=N*(N-1)/2-max_dc)
	f0 = 0
	for i in xrange(N):
		for j in xrange(N):
			if i==j:
				continue
			f0 = f0 + W[i][j] * X[i][j]
	obj = cp.Maximize(f0)
	prob = cp.Problem(obj, constraints)
	prob.solve()
	if prob.status=='optimal':
		return X
	else:
		raise Exception(prob.status)

def retrieve_result(X):
	N = len(X)
	ct = 0
	for i in xrange(N):
		for j in xrange(N):
			if i==j or X[i][j].value<1e-3:
				continue
			assert X[i][j].value > 1 - 1e-3, str(X[i][j].value)
			if i>j:
				ct += 1
	return ct

def rank_steps(parent, short_steps, full_steps):
	'''rank a list of steps for a parent'''
	N = len(short_steps)
	W = np.zeros((N, N))
	parent_enc = parent_encoder([parent])
	child_short_encs = child_short_encoder(short_steps)
	child_full_encs = child_full_encoder(full_steps)
	print child_short_encs.shape
	print child_full_encs.shape
	for i in xrange(N):
		for j in xrange(N):
			s1 = child_short_encs[i].view(1, -1)
			s2 = child_short_encs[j].view(1, -1)
			f1 = child_full_encs[i].view(1, -1)
			f2 = child_full_encs[j].view(1, -1)
			log_probs1 = step_ranker(parent_enc, s1, s2, f1, f2).detach().cpu().numpy()
			log_probs2 = step_ranker(parent_enc, s2, s1, f2, f1).detach().cpu().numpy()
			W[i, j] = (log_probs1[0, 0]+log_probs2[0,1]) / 2
	mistakes = []
	for dc in xrange(int(N*(N-1)/2+1)):
		X = solve(W, dc)
		mistake = retrieve_result(X)
		mistakes.append(mistake/(N*(N-1)/2))
	# plt.plot(mistakes)
	# plt.show()
	return mistakes

def top3_dcs(parent, short_steps, full_steps):
	'''rank a list of steps for a parent'''
	N = len(short_steps)
	W = np.zeros((N, N))
	parent_enc = parent_encoder([parent])
	child_short_encs = child_short_encoder(short_steps)
	child_full_encs = child_full_encoder(full_steps)
	# print child_short_encs.shape
	# print child_full_encs.shape
	for i in xrange(N):
		for j in xrange(N):
			s1 = child_short_encs[i].view(1, -1)
			s2 = child_short_encs[j].view(1, -1)
			f1 = child_full_encs[i].view(1, -1)
			f2 = child_full_encs[j].view(1, -1)
			log_probs1 = step_ranker(parent_enc, s1, s2, f1, f2).detach().cpu().numpy()
			log_probs2 = step_ranker(parent_enc, s2, s1, f2, f1).detach().cpu().numpy()
			W[i, j] = (log_probs1[0, 0]+log_probs2[0,1]) / 2
	dc = 3
	X = solve(W, dc)
	dcs = []
	for i in xrange(N):
		for j in xrange(i+1, N):
			if X[i][j].value < 1e-3 and X[j][i].value < 1e-3:
				dcs.append((i, j))
	return dcs

for _ in xrange(100):
	t, sts = step_ranking_dataset.get_random_article('dev')
	if len(sts)<=3:
		continue
	shorts, fulls = zip(*sts)
	dcs = top3_dcs(t, shorts, fulls)
	print 'Title:', t
	print 'Steps:'
	print '\n'.join(shorts)
	print ''
	for i, j in dcs:
		print shorts[i]
		print shorts[j]
		print ''
	print '\n\n\n'
	raw_input()
