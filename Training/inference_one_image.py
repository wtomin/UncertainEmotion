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
import matplotlib
import matplotlib.pyplot as plt
fontsize = 24
font = {
        'size'   : fontsize, 'family': 'sans-serif',
        'serif': 'Helvetica',
        'weight': 'normal'}
#plt.rcParams['figure.dpi'] = 300
matplotlib.rc('font', **font)
plt.rcParams.update({'figure.autolayout': True})
#plt.rcParams["font.family"] = "serif"
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

if args.auxiliary:
    from utils.misc import mobile_facenet
    print("Training model with an auxiliary task: face alignment.")
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
        annt_img, results = self.print_results(estimates, uncertainties)
        self.plot_results(img, results)

    def plot_results(self, annt_img, results):
        fig, axes = plt.subplots(1, 2, figsize=(12, 8))
        axes[0].imshow(np.array(annt_img))
        axes[0].set_axis_off()

        group_names = ['AU1', 'AU2', 'AU4', 'AU6', 'AU7', 'AU10', 'AU12', 'AU15','AU23','AU24', 'AU25','AU26']
        group_data = [ results[t][1] for t in group_names] 
        group_names = ['{}: {}'.format(t, results[t][0]) for t in group_names] 

        group_names += [results['EXPR'][0]]
        group_data += [results['EXPR'][1]]
        group_names += ['Valence: {:.1f}'.format(results['Valence'][0]), 'Arousal: {:.1f}'.format(results['Arousal'][0])]
        group_data += [results['Valence'][1], results['Arousal'][1]]
        group_names.reverse()
        group_data.reverse()
        axes[1].barh(group_names+group_names, group_data+group_data,  height=0.2)
        axes[1].set_xlim([0, 1])
        axes[1].get_xaxis().set_ticks([0, 1])
        axes[1].set_xlabel("Emotion Uncertainty")

        #plt.figtext(0.7,0.03,"Emotion Uncertainty", va="center", ha="center")
        plt.figtext(0.2,0.1,"Facial Image", va="center", ha="center")
        plt.tight_layout()
        plt.savefig('inference_img.pdf', dpi=300)
        plt.show()

    def print_results(self, estimates, uncertainties):
        results = {}
        for task in estimates:
            if task == 'FA':
                landmark = estimates['FA']
                landmark = landmark.reshape(-1, 2)
                image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR) 
                for x, y in landmark:
                    cv2.circle(image, (int(x*out_size), int(y*out_size)), 2, (0,255,0), -1)
                annt_img = Image.fromarray(image[:, :, ::-1])

            elif task == 'AU':
                categories = PRESET_VARS.Aff_wild2.categories['AU']
                output = ' '.join(["{}: {} ({:.1f}%)".format(categories[i], estimates[task][i], uncertainties[task][i]*100) for i in range(len(estimates[task]))])
                print(output)
                for i in range(len(categories)):
                    results[categories[i]] = [estimates[task][i], uncertainties[task][i]]
            elif task == 'EXPR':
                categories = PRESET_VARS.Aff_wild2.categories['EXPR']
                o = estimates[task]
                p = uncertainties[task]
                output = "{}: {:.1f}%".format(categories[o], p*100)
                print(output)
                results['EXPR'] = [categories[o], p]
            elif task == 'Valence' or task == 'Arousal':
                N = 20
                o = estimates[task]
                p = uncertainties[task]
                print("{}: {:.2f}".format(task, o))
                # edges = np.linspace(-1, 1, num= N+1)
                # dig = np.digitize(o, edges) - 1
                # if dig==N:
                #     dig  = N-1
                # start, end = edges[dig], edges[dig+1]
                print("uncertainty: {:.2f}%".format( p*100))
                results[task] = [o, p]
        return annt_img, results
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
                    uncertainties['AU'].append(entropy([probas.numpy()[i_o], 1- probas.numpy()[i_o]], base=2)/np.log2(2))
                uncertainties['AU'] = np.array(uncertainties['AU'])

            elif task == 'EXPR':
                probas = F.softmax(output['EXPR'].squeeze().cpu(), dim=-1)
                o = probas.argmax(-1).type(torch.LongTensor)
                estimates['EXPR'] = o.numpy()
                uncertainties['EXPR'] = entropy(probas, base=2)/np.log2(7)
            elif task == 'VA':
                N = 20
                pv = F.softmax(output['VA'][:,:, :N].cpu(), dim=-1).squeeze().numpy()
                pa = F.softmax(output['VA'][:,:, N:].cpu(), dim=-1).squeeze().numpy()
                bins = np.linspace(-1, 1, num=N)
                v = (bins * pv).sum(-1)
                a = (bins * pa).sum(-1)
                estimates['Valence'] = v
                estimates['Arousal'] = a
                # # discretize the predictions
                # edges = np.linspace(-1, 1, num= N+1)
                # v_dig = np.digitize(v, edges) - 1
                # if v_dig==N:
                #     v_dig = N -1
                # a_dig = np.digitize(a, edges) - 1
                # if a_dig ==N:
                #     a_dig = N- 1
                uncertainties['Valence'] = entropy(pv, base=2)/np.log2(N)
                uncertainties['Arousal'] = entropy(pa, base=2)/np.log2(N)
            elif task == 'FA':
                estimates['FA'] = output['FA'].cpu().numpy()
                uncertainties['FA'] = None
        return estimates, uncertainties



if __name__=="__main__":
    Tester()
