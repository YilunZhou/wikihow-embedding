
from __future__ import division

import argparse
import numpy as np
import matplotlib.pyplot as plt

smooth = lambda x, n:np.convolve(x, np.ones(n)/n, 'valid')

def add_argument(parser, argument, default, multi=False):
	typ = type(default)
	if typ==bool:
		parser.add_argument('--'+argument, dest=argument.replace('-', '_'), action='store_true')
		parser.add_argument('--no-'+argument, dest=argument.replace('-', '_'), action='store_false')
		parser.set_defaults(**{argument.replace('-', '_'): default})
	elif multi is False:
		parser.add_argument('--'+argument, nargs='?', type=type(default), default=default)
	else:
		parser.add_argument('--'+argument, nargs='+', type=type(default), default=default)

arg_parser = argparse.ArgumentParser()
add_argument(arg_parser, 'train-log', '/dev/null', multi=True)
add_argument(arg_parser, 'validate-every', 50)
add_argument(arg_parser, 'validate-log', '/dev/null', multi=True)

args = arg_parser.parse_args()
for k, v in vars(args).iteritems():
	print k, '=', v

train_datas = [np.loadtxt(f) for f in args.train_log]

dev_datas = [np.loadtxt(f) for f in args.validate_log]

for train_data, dev_data in zip(train_datas, dev_datas):
	train_smooth = 100
	dev_smooth = 10
	train_xs = range(len(smooth(train_data[:,0], train_smooth)))
	dev_xs = [i*args.validate_every for i in range(len(smooth(dev_data[:,0], dev_smooth)))]

	plt.subplot(2,2,1)
	plt.plot(train_xs, smooth(train_data[:,0], train_smooth))
	plt.plot(dev_xs, smooth(dev_data[:,0], dev_smooth))
	plt.title('relevance loss, training vs dev')

	plt.subplot(2,2,2)
	plt.plot(train_xs, smooth(train_data[:,1], train_smooth))
	plt.plot(dev_xs, smooth(dev_data[:,1], dev_smooth))
	plt.title('ordering loss, training vs dev')

	plt.subplot(2,2,3)
	plt.plot(train_xs, smooth(train_data[:,2], train_smooth))
	plt.plot(dev_xs, smooth(dev_data[:,2], dev_smooth))
	plt.title('relevance acc, training vs dev')

	plt.subplot(2,2,4)
	plt.plot(train_xs, smooth(train_data[:,3], train_smooth))
	plt.plot(dev_xs, smooth(dev_data[:,3], dev_smooth))
	plt.title('ordering acc, training vs dev')

plt.tight_layout()

plt.savefig('result.png', bbox_inches='tight')
plt.show()
