import torch
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