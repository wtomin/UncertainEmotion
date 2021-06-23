from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm2d, PReLU, ReLU, Sigmoid, Dropout2d, Dropout, AvgPool2d, MaxPool2d, AdaptiveAvgPool2d, Sequential, Module, Parameter
import torch.nn.functional as F
import torch
import torch.nn as nn
#from .TemporalModel import RNNModel, TransformerModel, LinearDecoder
class SpatialTemporalModel(nn.Module):
	def __init__(self, backbone, tmodels, tasks, dropout = 0.5):
		super(SpatialTemporalModel, self).__init__()
		self.backbone = backbone
		self.tasks = tasks
		tmodels = [tmodels[t] for t in self.tasks]
		self.dropout = nn.Dropout(dropout)
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

	def forward(self, input_signal, input_signal_length):
		features = self.backbone(input_signal = input_signal, input_signal_length=input_signal_length)
		# features.shape (Bs, N_C, T)
		B, N, T = features.size()
		outputs = {}
		for i, t in enumerate(self.tasks):
			features = self.dropout(features)
			out  =  self.tmodels[i](encoder_output= features)
			outputs[t] = out
		return outputs 




