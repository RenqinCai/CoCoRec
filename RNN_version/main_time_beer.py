import argparse
import torch
# import lib
import numpy as np
import os
import datetime
from dataset_time import *
# from dataset import *
from loss import *
from network import *
from optimizer import *
from trainer import *
from torch.utils import data
import pickle
import sys
import logger
import torch.nn as nn

from sampledSoftmax import *

parser = argparse.ArgumentParser()
parser.add_argument('--hidden_size', default=50, type=int)
parser.add_argument('--num_layers', default=1, type=int)
parser.add_argument('--batch_size', default=100, type=int)
parser.add_argument('--dropout_input', default=0, type=float)
parser.add_argument('--dropout_hidden', default=.2, type=float)

# parse the optimizer arguments
parser.add_argument('--optimizer_type', default='Adagrad', type=str)
parser.add_argument('--final_act', default='tanh', type=str)
parser.add_argument('--lr', default=.05, type=float)
parser.add_argument('--weight_decay', default=0, type=float)
parser.add_argument('--momentum', default=0.1, type=float)
parser.add_argument('--eps', default=1e-6, type=float)

parser.add_argument("-seed", type=int, default=7,
					 help="Seed for random initialization")
parser.add_argument("-sigma", type=float, default=None,
					 help="init weight -1: range [-sigma, sigma], -2: range [0, sigma]")
parser.add_argument("--embedding_dim", type=int, default=-1,
					 help="using embedding")
parser.add_argument('--loss_type', default='XE', type=str)
parser.add_argument('--topk', default=5, type=int)
parser.add_argument('--bptt', default=1, type=int)
parser.add_argument('--test_observed', default=5, type=int)
parser.add_argument('--window_size', default=30, type=int)
parser.add_argument('--cate_window_size', default=5, type=int)
parser.add_argument('--warm_start', default=5, type=int)
parser.add_argument('--n_epochs', default=20, type=int)
parser.add_argument('--save_dir', default='models', type=str)
parser.add_argument('--data_folder', default='../Data/movielen/1m/', type=str)
parser.add_argument('--data_action', default='item.pickle', type=str)
parser.add_argument('--data_cate', default='cate.pickle', type=str)
parser.add_argument('--data_time', default='time.pickle', type=str)
parser.add_argument("--is_eval", action='store_true')
parser.add_argument('--load_model', default=None,  type=str)
parser.add_argument('--checkpoint_dir', type=str, default='checkpoint')
parser.add_argument('--data_name', default=None, type=str)
parser.add_argument('--shared_embedding', default=None, type=int)
parser.add_argument('--patience', default=1000, type=int)
parser.add_argument('--negative_num', default=1000, type=int)
parser.add_argument('--valid_start_time', default=0, type=int)
parser.add_argument('--test_start_time', default=0, type=int)
parser.add_argument('--model_name', default="topkCascadeCategoryRNN", type=str)
parser.add_argument('--cate_embedding_dim', type=int, default=64)
parser.add_argument('--cate_hidden_size', type=int, default=64)
parser.add_argument('--cate_shared_embedding', default=1, type=int)
parser.add_argument('--beam_size', type=int, default=10)

# Get the arguments
args = parser.parse_args()
args.cuda = torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(7)
random.seed(args.seed)

if args.cuda:
	print("gpu")
	torch.cuda.manual_seed(args.seed)
else:
	print("cpu")

def make_checkpoint_dir():
	print("PARAMETER" + "-"*10)
	now = datetime.datetime.now()
	S = '{:02d}{:02d}{:02d}{:02d}'.format(now.month, now.day, now.hour, now.minute)
	checkpoint_dir = "../log/"+args.model_name+"/"+args.checkpoint_dir
	args.checkpoint_dir = checkpoint_dir
	save_dir = os.path.join(args.checkpoint_dir, S)

	if not os.path.exists("../log"):
		os.mkdir("../log")
	
	if not os.path.exists("../log/"+args.model_name):
		os.mkdir("../log/"+args.model_name)

	if not os.path.exists(args.checkpoint_dir):
		os.mkdir(args.checkpoint_dir)

	if not os.path.exists(save_dir):
		os.mkdir(save_dir)

	args.checkpoint_dir = save_dir

	with open(os.path.join(args.checkpoint_dir, 'parameter.txt'), 'w') as f:
		for attr, value in sorted(args.__dict__.items()):
			print("{}={}".format(attr.upper(), value))
			f.write("{}={}\n".format(attr.upper(), value))

	print("---------" + "-"*10)

def init_model(model):
	if args.sigma is not None:
		for p in model.parameters():
			if args.sigma != -1 and args.sigma != -2:
				sigma = args.sigma
				p.data.uniform_(-sigma, sigma)
			elif len(list(p.size())) > 1:
				sigma = np.sqrt(6.0 / (p.size(0) + p.size(1)))
				if args.sigma == -1:
					p.data.uniform_(-sigma, sigma)
				else:
					p.data.uniform_(0, sigma)

def count_parameters(model):
	parameter_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
	print("parameter_num", parameter_num) 

def main():
	
	hidden_size = args.hidden_size
	num_layers = args.num_layers
	batch_size = args.batch_size
	dropout_input = args.dropout_input
	dropout_hidden = args.dropout_hidden
	embedding_dim = args.embedding_dim
	final_act = args.final_act
	loss_type = args.loss_type
	topk = args.topk
	optimizer_type = args.optimizer_type
	lr = args.lr
	weight_decay = args.weight_decay
	momentum = args.momentum
	eps = args.eps

	n_epochs = args.n_epochs

	window_size = args.window_size
	cate_window_size = args.cate_window_size

	### initialize a log object
	log = logger.Logger()
	log.addIOWriter(args)

	shared_embedding = args.shared_embedding
	message = "main_time.py shared_embedding: "+str(shared_embedding)
	log.addOutput2IO(message)

	if embedding_dim == -1:
		message = "embedding dim not -1 "+str(embedding_dim)
		log.addOutput2IO(message)
		raise AssertionError()

	message = "*"*10
	log.addOutput2IO(message)
	message = "train load, valid, test load"
	log.addOutput2IO(message)

	data_action = args.data_folder+args.data_action
	data_cate = args.data_folder+args.data_cate
	data_time = args.data_folder+args.data_time

	valid_start_time = args.valid_start_time
	test_start_time = args.test_start_time

	observed_threshold = args.test_observed

	st = datetime.datetime.now()
	data_obj = MYDATA(data_action, data_cate, data_time, valid_start_time, test_start_time, observed_threshold, window_size, cate_window_size)
	et = datetime.datetime.now()
	print("duration ", et-st)

	train_data = data_obj.train_dataset
	valid_data = data_obj.test_dataset
	test_data = data_obj.test_dataset
	
	input_size = data_obj.items()
	output_size = input_size

	negative_num = args.negative_num
	print("negative num", negative_num)

	message = "input_size "+str(input_size)
	log.addOutput2IO(message)

	cate_input_size = data_obj.cates()
	cate_output_size = cate_input_size
	cate_embedding_dim = args.cate_embedding_dim
	cate_hidden_size = args.cate_hidden_size

	beam_size = args.beam_size

	train_data_loader = MYDATALOADER(train_data, batch_size)
	valid_data_loader = MYDATALOADER(valid_data, batch_size)

	if not args.is_eval:
		make_checkpoint_dir()

	if not args.is_eval:

		### for negative sampling
		ss = SampledSoftmax(output_size, negative_num, embedding_dim, None)

		network = GatedLongRec(log, ss, input_size, hidden_size, output_size, cate_embedding_dim=cate_embedding_dim, cate_input_size=cate_input_size, cate_output_size=cate_output_size, cate_hidden_size=cate_hidden_size,
							final_act=final_act,
							num_layers=num_layers,
							use_cuda=args.cuda,
							dropout_input=dropout_input,
							dropout_hidden=dropout_hidden,
							embedding_dim=embedding_dim, 
							shared_embedding=shared_embedding
							)
		
		count_parameters(network)

		init_model(network)
		optimizer = Optimizer(network.parameters(),
								  optimizer_type=optimizer_type,
								  lr=lr,
								  weight_decay=weight_decay,
								  momentum=momentum,
								  eps=eps)

		loss_function = LossFunction(loss_type=loss_type, use_cuda=args.cuda)

		cate_loss_function = LossFunction(loss_type=loss_type, use_cuda=args.cuda)

		trainer = Trainer(log, network, beam_size, train_data=train_data_loader,
							  eval_data=valid_data_loader,
							  optim=optimizer,
							  use_cuda=args.cuda,
							  loss_func=loss_function,
							  cate_loss_func=cate_loss_function,
							  topk = args.topk, input_size=input_size,
							  args=args)

		trainer.train(0, n_epochs - 1, batch_size)
	
if __name__ == '__main__':
	main()