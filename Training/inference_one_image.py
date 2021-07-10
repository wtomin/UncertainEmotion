import time
from options.test_options import TestOptions
#from data.custom_dataset_data_loader import Multitask_DatasetDataLoader
from data.test_video_dataset import Test_dataset
from models import ModelsFactory
from collections import OrderedDict
from torch.autograd import Variable
import os
import numpy as np
from PATH import PATH
import pickle
from copy import deepcopy, copy
import pandas as pd
from tqdm import tqdm
import pickle
from utils.transforms import test_transforms
from utils.validation import sigmoid, softmax
from scipy.stats import entropy

#################RuntimeError: received 0 items of ancdata ###########################
import torch
torch.multiprocessing.set_sharing_strategy("file_system")
################# RuntimeError: unable to open shared memory object </torch_29841_2933458171> in read-write mode ############
import sys
import torch
from torch.utils.data import dataloader
from torch.multiprocessing import reductions
from multiprocessing.reduction import ForkingPickler
import torch.nn.functional as F
default_collate_func = dataloader.default_collate
import cv2
from PIL import Image
def default_collate_override(batch):
    dataloader._use_shared_memory = False
    return default_collate_func(batch)

setattr(dataloader, 'default_collate', default_collate_override)

for t in torch._storage_classes:
  if sys.version_info[0] == 2:
    if t in ForkingPickler.dispatch:
        del ForkingPickler.dispatch[t]
  else:
    if t in ForkingPickler._extra_reducers:
        del ForkingPickler._extra_reducers[t]
#########################################################################
args = TestOptions().parse()

if args.auxillary:
    from utils.misc import mobile_facenet
    print("Training model with an auxillary task: face alignment.")
    args.tasks = args.tasks + ['FA']
    FA_teacher = mobile_facenet(pretrained=True, cuda=args.cuda)
    FA_teacher.eval()
else:
    FA_teacher = None

EPS = 1e-8
PRESET_VARS = PATH()

test_image_path = '/media/Samsung/Aff-wild2-Challenge/cropped_aligned/90-30-1080x1920/00081.jpg'

img = Image.open(test_image_path)
img = img.resize((224, 224))
img.show()
out_size = img.size[0]
transform_func = test_transforms(args.image_size)
test_image = transform_func(img)
test_image = test_image.unsqueeze(0).unsqueeze(0).cuda() #(1, 1, 3, 112, 112)

class Tester(object):
    def __init__(self):
        args.load_epoch = args.load_epochs[0]
        args.name = args.names[0]
        model = ModelsFactory.get_by_name(args,
                    is_train= False,
                    dropout = 0.5,
                    pretrained=False)
        self._model = model

        estimates, uncertainties = self._test()
        self.print_resutls(estimates, uncertainties)
    def print_resutls(self, estimates, uncertainties):
        for task in estimates:
            if task == 'FA':
                landmark = estimates['FA']
                landmark = landmark.reshape(-1, 2)
                image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)  # PIL转cv2
                for x, y in landmark:
                    cv2.circle(image, (int(x*out_size), int(y*out_size)), 2, (0,255,0), -1)
                Image.fromarray(image[:, :, ::-1]).show()
            elif task == 'AU':
                categories = PRESET_VARS.Aff_wild2.categories['AU']
                output = ' '.join(["{}: {} ({:.1f}%)".format(categories[i], estimates[task][i], uncertainties[task][i]*100) for i in range(len(estimates[task]))])
                print(output)
            elif task == 'EXPR':
                categories = PRESET_VARS.Aff_wild2.categories['EXPR']
                o = estimates[task]
                p = uncertainties[task]
                output = "{}: {:.1f}%".format(categories[o], p*100)
                print(output)
            elif task == 'Valence' or task == 'Arousal':
                N = 20
                o = estimates[task]
                p = uncertainties[task]
                print("{}: {:.2f}".format(task, o))
                edges = np.linspace(-1, 1, num= N+1)
                dig = np.digitize(o, edges) - 1
                if dig==N:
                    dig  = N-1
                start, end = edges[dig], edges[dig+1]
                print("{:.2f} ~ {:.2f}: {:.2f}%".format(start, end, p))
    def _test(self):
        example_task = 'AU'
        hiddens = dict([t, None] for t in args.tasks)
        with torch.no_grad():
            input_image = Variable(test_image)
            output, _ = self._model._model(input_image, hiddens)
        estimates, uncertainties = self.get_uncertaintyand_estimates_from_multitask_output(output)
        return estimates, uncertainties
    def get_uncertaintyand_estimates_from_multitask_output(self, output):
        estimates = {}
        uncertainties = {}
        for task in output.keys():
            if task == 'AU':
                probas = torch.sigmoid(output['AU'].squeeze().cpu())
                o = (probas>0.5).type(torch.LongTensor)
                estimates['AU'] = o.numpy()
                uncertainties['AU'] = []
                for i_o, o in enumerate( estimates['AU']):
                    if o ==0:
                        uncertainties['AU'].append(1- probas.numpy()[i_o])
                    else:
                        uncertainties['AU'].append(probas.numpy()[i_o])
                uncertainties['AU'] = np.array(uncertainties['AU'])

            elif task == 'EXPR':
                probas = F.softmax(output['EXPR'].squeeze().cpu(), dim=-1)
                o = probas.argmax(-1).type(torch.LongTensor)
                estimates['EXPR'] = o.numpy()
                uncertainties['EXPR'] = probas[o]
            elif task == 'VA':
                N = 20
                pv = F.softmax(output['VA'][:,:, :N].cpu(), dim=-1).squeeze().numpy()
                pa = F.softmax(output['VA'][:,:, N:].cpu(), dim=-1).squeeze().numpy()
                bins = np.linspace(-1, 1, num=N)
                v = (bins * pv).sum(-1)
                a = (bins * pa).sum(-1)
                estimates['Valence'] = v
                estimates['Arousal'] = a
                # discretize the predictions
                edges = np.linspace(-1, 1, num= N+1)
                v_dig = np.digitize(v, edges) - 1
                if v_dig==N:
                    v_dig = N -1
                a_dig = np.digitize(a, edges) - 1
                if a_dig ==N:
                    a_dig = N- 1
                uncertainties['Valence'] = pv[v_dig]
                uncertainties['Arousal'] = pa[a_dig]
            elif task == 'FA':
                estimates['FA'] = output['FA'].cpu().numpy()
                uncertainties['FA'] = None
        return estimates, uncertainties



if __name__=="__main__":
    Tester()
