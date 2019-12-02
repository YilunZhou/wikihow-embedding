
from __future__ import division

import torch
from torch.optim import Adam, SGD
from article_repr import *
import random, argparse, os, pickle
import numpy as np
from collections import Counter
from load_data import load_split_articles, WordIndexer, ParentChildDataset, StepRankingDataset
from nn_models import *

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
add_argument(arg_parser, 'embedding-dim', 300)
add_argument(arg_parser, 'freeze-embedding', True)
add_argument(arg_parser, 'use-parent', True)
add_argument(arg_parser, 'context-encoder', 'None')
add_argument(arg_parser, 'context-freeze-embedding', True)
add_argument(arg_parser, 'encoding-dim', 200)
add_argument(arg_parser, 'parent-child-hidden-dim', 200)
add_argument(arg_parser, 'step-ranking-hidden-dim', 200)
add_argument(arg_parser, 'auto-encoder-dim', 50)
add_argument(arg_parser, 'ranking-margin', 1)
add_argument(arg_parser, 'batch-size', 128)
add_argument(arg_parser, 'occur-cutoff', 5)
add_argument(arg_parser, 'gpu', True)
add_argument(arg_parser, 'train-log', '/dev/null')
add_argument(arg_parser, 'validate-every', 50)
add_argument(arg_parser, 'validate-log', '/dev/null')
add_argument(arg_parser, 'test-log', '/dev/null')
add_argument(arg_parser, 'save-every', 200)
add_argument(arg_parser, 'save-dir', '/dev/null')
add_argument(arg_parser, 'child-short-enc-template', 'child_short_encoder_%i.model')
add_argument(arg_parser, 'child-full-enc-template', 'child_full_encoder_%i.model')
add_argument(arg_parser, 'parent-enc-template', 'parent_encoder_%i.model')
add_argument(arg_parser, 'pc-classifier-template', 'pc_classifier_%i.model')
add_argument(arg_parser, 'step-ranker-template', 'step_ranker_%i.model')
add_argument(arg_parser, 'max-iter', 10000)
add_argument(arg_parser, 'dont-care-penalization', 0)

args = arg_parser.parse_args()
for k, v in vars(args).iteritems():
	print k, '=', v



def zero_grad_all(models):
	for m in models:
		if m is not None:
			m.zero_grad()

def step_all(optimizers):
	for m in optimizers:
		if m is not None:
			m.step()

def clean_full(full):
	full = full.replace('#', '').replace('*', '')
	return full

def forward_parent_child(pc_data, p_encoder, c_short_encoder, c_full_encoder, pc_classifier, pc_loss_func):
	'''return symbolic loss and numerical correct classification ratio'''
	X_texts, y = pc_data
	y_tensor = torch.tensor(y.astype('int64'))
	if args.gpu:
		y_tensor = y_tensor.cuda()
	parents, child_shorts, child_fulls = zip(*X_texts)
	child_fulls = map(clean_full, child_fulls)
	parent_encs = p_encoder(parents)
	child_short_encs = c_short_encoder(child_shorts)
	if c_full_encoder is not None:
		child_full_encs = c_full_encoder(child_fulls)
	else:
		child_full_encs = None
	log_probs = pc_classifier(parent_encs, child_short_encs, child_full_encs)
	parent_child_loss = pc_loss_func(log_probs, y_tensor)
	y_pred = log_probs.detach().cpu().numpy().argmax(axis=1)
	pc_acc = (y_pred==y).mean()
	return parent_child_loss, pc_acc

def forward_step_ranking(sr_data, p_encoder, c_short_encoder, c_full_encoder, s_ranker, sr_loss_func):
	parents, child1s, child2s = zip(*sr_data)
	child1s_short, child1s_full = zip(*child1s)
	child2s_short, child2s_full = zip(*child2s)
	child1s_full = map(clean_full, child1s_full)
	child2s_full = map(clean_full, child2s_full)
	parent_encs = p_encoder(parents)
	child1_short_encs = c_short_encoder(child1s_short)
	child2_short_encs = c_short_encoder(child2s_short)
	ranking_target = [random.choice([0, 1]) for _ in xrange(args.batch_size)]
	A_short_encs = torch.zeros(child1_short_encs.shape)
	B_short_encs = torch.zeros(child2_short_encs.shape)
	if args.gpu:
		A_short_encs = A_short_encs.cuda()
		B_short_encs = B_short_encs.cuda()
	for i in xrange(args.batch_size):
		if ranking_target[i]==0:
			A_short_encs[i] = child1_short_encs[i]
			B_short_encs[i] = child2_short_encs[i]
		else:
			A_short_encs[i] = child2_short_encs[i]
			B_short_encs[i] = child1_short_encs[i]
		
	if c_full_encoder is not None:
		child1_full_encs = c_full_encoder(child1s_full)
		child2_full_encs = c_full_encoder(child2s_full)
		A_full_encs = torch.zeros(child1_full_encs.shape)
		B_full_encs = torch.zeros(child2_full_encs.shape)
		if args.gpu:
			A_full_encs = A_full_encs.cuda()
			B_full_encs = B_full_encs.cuda()
		for i in xrange(args.batch_size):
			if ranking_target[i]==0:
				A_full_encs[i] = child1_full_encs[i]
				B_full_encs[i] = child2_full_encs[i]
			else:
				A_full_encs[i] = child2_full_encs[i]
				B_full_encs[i] = child1_full_encs[i]
	else:
		# child1_full_encs = None
		# child2_full_encs = None
		A_full_encs = None
		B_full_encs = None
	
	log_probs = s_ranker(parent_encs, A_short_encs, B_short_encs, A_full_encs, B_full_encs)
	ranking_target = torch.tensor(ranking_target).type(torch.LongTensor)
	# inverted_ranking_target = 1 - ranking_target
	if args.gpu:
		ranking_target = ranking_target.cuda()
		# inverted_ranking_target = inverted_ranking_target.cuda()
	# step_ranking_loss = sr_loss_func(log_probs, inverted_ranking_target)
	step_ranking_loss = sr_loss_func(log_probs, ranking_target)
	log_probs = log_probs.detach().cpu().numpy()
	rank_pred = log_probs.argmax(axis=1)
	# pred_frac = (rank_pred!=2).mean()
	correct = [pred==true or pred==2 for pred, true in zip(rank_pred, ranking_target.detach().cpu().numpy())]
	rk_acc = sum(correct) / len(correct)
	probs = np.exp(log_probs)
	# print 'Pred frac:', pred_frac
	return step_ranking_loss, rk_acc, probs

def train_single_batch(pc_dataset, rk_dataset, 
	p_encoder, c_short_encoder, c_full_encoder, pc_classifier, s_ranker, pc_loss_func, sr_loss_func, 
	p_enc_opt, c_short_enc_opt, c_full_enc_opt, pc_opt, sr_opt):
	zero_grad_all([p_encoder, c_short_encoder, c_full_encoder, pc_classifier, s_ranker])

	parent_child_data = pc_dataset.sample_batch_data('train', args.batch_size)
	parent_child_loss, pc_acc = forward_parent_child(
		parent_child_data, p_encoder, c_short_encoder, c_full_encoder, pc_classifier, pc_loss_func)
	pc_loss = parent_child_loss.detach().cpu().numpy()
	
	ranking_data = rk_dataset.sample_ranking_data('train', args.batch_size)
	step_ranking_loss, rk_acc, probs = forward_step_ranking(
		ranking_data, p_encoder, c_short_encoder, c_full_encoder, s_ranker, sr_loss_func)
	rk_loss = step_ranking_loss.detach().cpu().numpy()
	
	total_loss = 1.0 * parent_child_loss + 1.0 * step_ranking_loss
	total_loss.backward()
	step_all([p_enc_opt, c_short_enc_opt, c_full_enc_opt, pc_opt, sr_opt])

	return pc_loss, rk_loss, pc_acc, rk_acc, probs

def evaluate_model(ds_label, pc_dataset, rk_dataset, 
	p_encoder, c_short_encoder, c_full_encoder, pc_classifier, s_ranker, pc_loss_func, sr_loss_func):
	zero_grad_all([p_encoder, c_short_encoder, c_full_encoder, pc_classifier, s_ranker])

	parent_child_data = pc_dataset.sample_batch_data(ds_label, args.batch_size)
	parent_child_loss, pc_acc = forward_parent_child(
		parent_child_data, p_encoder, c_short_encoder, c_full_encoder, pc_classifier, pc_loss_func)
	pc_loss = parent_child_loss.detach().cpu().numpy()
	
	ranking_data = rk_dataset.sample_ranking_data(ds_label, args.batch_size)
	step_ranking_loss, rk_acc, probs = forward_step_ranking(
		ranking_data, p_encoder, c_short_encoder, c_full_encoder, s_ranker, sr_loss_func)
	rk_loss = step_ranking_loss.detach().cpu().numpy()

	return pc_loss, rk_loss, pc_acc, rk_acc


# random.seed(1)
# split_articles = load_split_articles(args.article_path)

split_articles = pickle.load(open('articles/split_articles.pkl', 'rb'))

print 'loading parent child dataset'
parent_child_dataset = ParentChildDataset(split_articles)
print 'loading step ranking dataset'
step_ranking_dataset = StepRankingDataset(split_articles)
print 'loading word indexer'
word_indexer = WordIndexer(split_articles, args.embedding_path, args.occur_cutoff)

if args.use_parent:
	parent_encoder = Encoder(word_indexer, args.freeze_embedding, args.encoding_dim, args.gpu)
else:
	parent_encoder = RandEncoder(args.encoding_dim, args.gpu)
child_short_encoder = Encoder(word_indexer, args.freeze_embedding, args.encoding_dim, args.gpu)
assert args.context_encoder in ['None', 'LSTM', 'Bag'], 'Context encoder must be one of None, LSTM or Bag'
if args.context_encoder=='None':
	child_full_dim = None
	child_full_encoder = None
elif args.context_encoder=='LSTM':
	child_full_dim = args.encoding_dim
	child_full_encoder = Encoder(word_indexer, args.context_freeze_embedding, args.encoding_dim, args.gpu)
elif args.context_encoder=='Bag':
	child_full_dim = args.embedding_dim
	child_full_encoder = BagEncoder(word_indexer, args.context_freeze_embedding, args.gpu)

parent_child_classifier = ParentChildClassifier(args.encoding_dim, args.encoding_dim, child_full_dim, args.parent_child_hidden_dim)
step_ranker = StepRankerLogistic(args.encoding_dim, args.encoding_dim, child_full_dim, args.step_ranking_hidden_dim)

parent_child_loss_func = nn.NLLLoss()
step_ranking_loss_func = nn.NLLLoss()

parent_encoder_optimizer = Adam(parent_encoder.parameters())
child_short_encoder_optimizer = Adam(child_short_encoder.parameters())
if child_full_encoder is not None:
	child_full_encoder_optimizer = Adam(child_full_encoder.parameters())
else:
	child_full_encoder_optimizer = None
parent_child_optimizer = Adam(parent_child_classifier.parameters())
step_ranking_optimizer = Adam(step_ranker.parameters())

if args.gpu:
	parent_encoder.cuda()
	child_short_encoder.cuda()
	if child_full_encoder is not None:
		child_full_encoder.cuda()
	parent_child_classifier.cuda()
	step_ranker.cuda()
	parent_child_loss_func.cuda()
	step_ranking_loss_func.cuda()

train_log = open(args.train_log, 'w')
validate_log = open(args.validate_log, 'w')
test_log = open(args.test_log, 'w')
for iter in xrange(args.max_iter):
	
	### validate
	if iter%args.validate_every==0:
		parent_encoder.check_padding_unchanged()
		child_short_encoder.check_padding_unchanged()
		if child_full_encoder is not None:
			child_full_encoder.check_padding_unchanged()

		pc_loss, rk_loss, pc_acc, rk_acc = evaluate_model('dev', parent_child_dataset, step_ranking_dataset,
			parent_encoder, child_short_encoder, child_full_encoder, 
			parent_child_classifier, step_ranker, parent_child_loss_func, step_ranking_loss_func)
		print pc_loss, rk_loss, pc_acc, rk_acc
		validate_log.write('%f %f %f %f\n'%(pc_loss, rk_loss, pc_acc, rk_acc))

		pc_loss, rk_loss, pc_acc, rk_acc = evaluate_model('test', parent_child_dataset, step_ranking_dataset,
			parent_encoder, child_short_encoder, child_full_encoder, 
			parent_child_classifier, step_ranker, parent_child_loss_func, step_ranking_loss_func)
		print pc_loss, rk_loss, pc_acc, rk_acc
		test_log.write('%f %f %f %f\n'%(pc_loss, rk_loss, pc_acc, rk_acc))
		
		train_log.flush()
		validate_log.flush()
		test_log.flush()
	
	### save
	if args.save_every!=-1 and iter%args.save_every==0:
		if not os.path.isdir(args.save_dir):
			os.makedirs(args.save_dir)
		torch.save(parent_encoder, args.save_dir+'/'+args.parent_enc_template%iter)
		torch.save(child_short_encoder, args.save_dir+'/'+args.child_short_enc_template%iter)
		if child_full_encoder is not None:
			torch.save(child_full_encoder, args.save_dir+'/'+args.child_full_enc_template%iter)
		torch.save(parent_child_classifier, args.save_dir+'/'+args.pc_classifier_template%iter)
		torch.save(step_ranker, args.save_dir+'/'+args.step_ranker_template%iter)

	### train
	pc_loss, rk_loss, pc_acc, rk_acc, probs = train_single_batch(parent_child_dataset, step_ranking_dataset, 
		parent_encoder, child_short_encoder, child_full_encoder, 
		parent_child_classifier, step_ranker, parent_child_loss_func, step_ranking_loss_func, 
		parent_encoder_optimizer, child_short_encoder_optimizer, child_full_encoder_optimizer, 
		parent_child_optimizer, step_ranking_optimizer)
	avg_probs = probs.mean(axis=0)
	std_probs = np.std(probs, axis=0)
	print pc_loss, rk_loss, pc_acc, rk_acc
	print 'Avg class probs:', avg_probs
	print 'Std class probs:', std_probs
	# print 'Min DC prob:', min(probs[:, 2])
	print 'avg class diff:', abs(probs[:, 0]-probs[:, 1]).mean()
	# print probs
	train_log.write('%f %f %f %f\n'%(pc_loss, rk_loss, pc_acc, rk_acc))
