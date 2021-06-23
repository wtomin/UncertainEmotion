import torch
import numpy as np
import os
from nemo.collections.asr.models import EncDecClassificationModel
class Identity(torch.nn.Module):
    def forward(self, encoder_output):
        return encoder_output

def map_location(cuda):
    if torch.cuda.is_available() and cuda:
        map_location=lambda storage, loc: storage.cuda()
    else:
        map_location='cpu'

def rename(newname):
    def decorator(f):
        f.__name__ = newname
        return f
    return decorator

def get_MarbleNet_config():
    MODEL_CONFIG = "marblenet_3x2x64.yaml"

    if not os.path.exists(MODEL_CONFIG):
        cmd = "wget https://raw.githubusercontent.com/NVIDIA/NeMo/v1.0.2/examples/asr/conf/marblenet/{}".format(MODEL_CONFIG)
        os.system(cmd)

    from omegaconf import OmegaConf
    config_path = MODEL_CONFIG
    config = OmegaConf.load(config_path)
    config = OmegaConf.to_container(config, resolve=True)
    config = OmegaConf.create(config)
    return config

class VAD_MarbleNet(EncDecClassificationModel):
    def __init__(self, *args, **wargs):
        super(EncDecClassificationModel, self).__init__(*args, **wargs)

    def forward(self, input_signal, input_signal_length):
        forward_without_type_check = super().forward.__wrapped__
        logits = forward_without_type_check(input_signal=input_signal, input_signal_length=input_signal_length)
        return logits
        
def read_AU(txt_file):
    with open(txt_file, 'r') as f:
        lines = f.readlines()
    lines = lines[1:] # skip first line
    lines = [x.strip() for x in lines]
    lines = [x.split(',') for x in lines]
    lines = [[float(y) for y in x ] for x in lines]
    return np.array(lines)
def read_Expr(txt_file):
    with open(txt_file, 'r') as f:
        lines = f.readlines()
    lines = lines[1:] # skip first line
    lines = [x.strip() for x in lines]
    lines = [int(x) for x in lines]
    return np.array(lines)
def read_VA(txt_file):
    with open(txt_file, 'r') as f:
        lines = f.readlines()
    lines = lines[1:] # skip first line
    lines = [x.strip() for x in lines]
    lines = [x.split(',') for x in lines]
    lines = [[float(y) for y in x ] for x in lines]
    return np.array(lines)