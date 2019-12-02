
from __future__ import division

import torch
import numpy as np
import cvxpy as cp
import argparse, random
from nn_models import *
from article_repr import *
from load_data import load_split_articles, WordIndexer, ParentChildDataset, StepRankingDataset
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from sklearn.manifold import TSNE

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

random.seed(1)
split_articles = load_split_articles(args.article_path)

print 'loading word indexer'
word_indexer = WordIndexer(split_articles, args.embedding_path, args.occur_cutoff)
print 'loading parent child dataset'
parent_child_dataset = ParentChildDataset(split_articles)
print 'loading step ranking dataset'
step_ranking_dataset = StepRankingDataset(split_articles)

print 'loading models'
# parent_encoder = torch.load(args.model_folder+'/'+args.parent_enc_template%args.iter)
# child_encoder = torch.load(args.model_folder+'/'+args.child_enc_template%args.iter)


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

N = 50
steps = []
titles = []
for _ in xrange(N):
	t, sts = step_ranking_dataset.get_random_article('dev')
	steps.append(random.choice(sts))
	titles.append(t)
shorts, fulls = zip(*steps)
short_codes = child_short_encoder(shorts)
parent_codes = parent_encoder(titles)
tsne = TSNE()
fitted = tsne.fit_transform(parent_codes.detach().cpu().numpy())

xs, ys = zip(*fitted)
plt.plot(xs, ys, '.')

import textwrap

for i in xrange(N):
	txt = titles[i]
	x = xs[i]
	y = ys[i]
	lines = textwrap.wrap(txt, width=20)
	txt = '\n'.join(lines)
	plt.annotate(txt, xy=(x, y), horizontalalignment='center', verticalalignment='center')

plt.show()
