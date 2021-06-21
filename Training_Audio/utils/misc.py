import torch
import numpy as np
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


def mobile_facenet(pretrained=True, cuda=True):
    from models.mobilefacenet import MobileFaceNet
    model = MobileFaceNet([112, 112], 136) 
    checkpoint = 'checkpoint/mobilefacenet_model_best.pth.tar'
    if pretrained:
        checkpoint = torch.load(checkpoint, map_location=map_location(cuda))
        model.load_state_dict(checkpoint['state_dict'])
    if cuda:
        model.cuda()
    return model

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