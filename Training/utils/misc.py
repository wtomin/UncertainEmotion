import torch
from scipy.stats import entropy
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


def cal_uncertainty(stacked_probas):
    N_models = stacked_probas.shape[0]
    N_classes = stacked_probas.shape[-1]
    total_uncerainty = entropy(stacked_probas.mean(0), base=2, axis=-1)/np.log2(N_classes)
    alea_uncertainty = np.stack([entropy(prob, base=2, axis=-1) for prob in stacked_probas], axis=0).mean(0)/np.log2(N_classes)
    epi_uncertainty = total_uncerainty - alea_uncertainty
    if isinstance(epi_uncertainty, float):
        assert epi_uncertainty >=0, "expected to be non-negative"
    else:
        assert epi_uncertainty.min() >=0, "expected to be non-negative"
    return alea_uncertainty, epi_uncertainty


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