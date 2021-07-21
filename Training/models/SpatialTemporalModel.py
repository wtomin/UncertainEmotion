from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm2d, PReLU, ReLU, Sigmoid, Dropout2d, Dropout, AvgPool2d, MaxPool2d, AdaptiveAvgPool2d, Sequential, Module, Parameter
import torch.nn.functional as F
import torch
import torch.nn as nn
from .TemporalModel import RNNModel, TransformerModel

class SpatialTemporalModel(nn.Module):
	def __init__(self, backbone, tmodels, tasks):
		super(SpatialTemporalModel, self).__init__()
		self.backbone = backbone
		self.tasks = tasks
		tmodels = [tmodels[t] for t in self.tasks]
		self.tmodels = nn.ModuleList(tmodels)

	def eval(self,):
		self.backbone.eval()
		self.tmodels.eval()
	def train(self,):
		self.backbone.train()
		self.tmodels.train()
	def cuda(self,):
		self.backbone.cuda()
		self.tmodels.cuda()

	def forward(self, input_seq, hiddens=None):
		bs, seq_len = input_seq.size(0), input_seq.size(1)
		input_new = input_seq.view((bs * seq_len,)+ input_seq.size()[2:])
		features = self.backbone(input_new)
		features = features.view((bs, seq_len, -1))
		outputs = {}
		output_hiddens = {}
		for i, t in enumerate(self.tasks):
			if self.tmodels[i].model_type.lower() == 'transformer':
			    outputs[t] = self.tmodels[i](features)
			    output_hiddens[t] = None
			else:
				# rnn models will return initial state and hidden state
				if hiddens is not None:
					assert isinstance(hiddens, dict)
					hidden = hiddens[t]
				else:
					hidden = None
				out, hidden = self.tmodels[i](features, hidden)
				outputs[t] = out
				output_hiddens[t] = hidden
		return outputs, output_hiddens




