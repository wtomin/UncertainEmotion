from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm2d, PReLU, ReLU, Sigmoid, AvgPool2d, MaxPool2d, AdaptiveAvgPool2d, Sequential, Module, Parameter
import torch.nn.functional as F
import torch
import torch.nn as nn
#from .TemporalModel import RNNModel, TransformerModel, LinearDecoder
class SpatialTemporalModel(nn.Module):
    def __init__(self, video_backbone, audio_backbone, tmodels, tasks):
        super(SpatialTemporalModel, self).__init__()
        self.video_backbone = video_backbone
        self.audio_backbone = audio_backbone
        self.tasks = tasks
        tmodels = [tmodels[t] for t in self.tasks]
        self.tmodels = nn.ModuleList(tmodels)

    def eval(self,):
        self.video_backbone.eval()
        self.audio_backbone.eval()
        self.tmodels.eval()
    def train(self,):
        self.video_backbone.train()
        self.audio_backbone.train()
        self.tmodels.train()
    def cuda(self,):
        self.video_backbone.cuda()
        self.audio_backbone.cuda()
        self.tmodels.cuda()

    def forward(self, video_input, audio_input, input_hiddens = None):
        Bs, Len = video_input.size(0), video_input.size(1)
        reshape = [Bs*Len,] + list(video_input.size())[2:]
        video_features = self.video_backbone(video_input.view(*reshape))
        video_features = video_features.view(Bs, Len, -1)
        
        input_signal, input_signal_length = audio_input
        Bs, Len = input_signal.size(0), input_signal.size(1)
        reshape =  [Bs*Len, ] + list(input_signal.size())[2:]
        audio_features = self.audio_backbone(input_signal = input_signal.view(*reshape),  input_signal_length=input_signal_length.view(Bs*Len,))
        audio_features = audio_features.mean(-1) # average on the time steps
        audio_features = audio_features.view(Bs, Len, -1)
        # features.shape (Bs, N_C, T)
        features = torch.cat([video_features, audio_features], dim=-1)
        B, N, T = features.size()
        outputs = {}
        output_hiddens = {}
        for i, t in enumerate(self.tasks):
            h = None if input_hiddens is None else input_hiddens[t]
            out, hiddens  =  self.tmodels[i](features, h)
            outputs[t] = out
            output_hiddens[t] = hiddens
        return outputs, output_hiddens




