from mobilefacenet import MobileFaceNet
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import pickle
import time
import cv2
import os
import argparse
import numpy as np
from scipy.stats import entropy
from PIL import Image
import time
import torchvision.transforms as transforms

parser = argparse.ArgumentParser()
parser.add_argument('--use_cuda',  action='store_true',
    help='whether to use GPU device')
parser.add_argument('--face_detector', type=str, default = 'cascade',
    choices = ['mtcnn', 'cascade'],
    help='select the face detector')
parser.add_argument('--save_file', type=str, default=None, 
    help='if set, save to video file')
args = parser.parse_args()
# face detector
if args.face_detector == 'cascade':
    cascPathface = os.path.dirname(
     cv2.__file__) + "/data/haarcascade_frontalface_default.xml"
    # load the harcaascade in the cascade classifier
    detector = cv2.CascadeClassifier(cascPathface)
else:
    from mtcnn import MTCNN
    detector = MTCNN()

if args.save_file is not None:
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    assert '.avi' in args.save_file, "support avi file only"
    out_video = cv2.VideoWriter(filename=args.save_file, 
        fourcc=fourcc, 
        frameSize=(600, 400),
        fps=30)
categories = {'AU': ['AU1','AU2','AU4','AU6','AU7','AU10','AU12','AU15','AU23','AU24','AU25','AU26'],
                'EXPR':['Neutral','Anger','Disgust','Fear','Happiness','Sadness','Surprise'],
                'VA':['valence', 'arousal']}
EPS =  1e-8
def sigmoid(x):
    return np.exp(x)/(1+np.exp(x))
def softmax(x, axis=-1):
    return np.exp(x)/np.expand_dims(np.sum(np.exp(x), axis=axis), axis)
########### Model Definition #############
class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ninp, nhid, nlayers, nout, dropout=0.5):
        super(RNNModel, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.model_type = rnn_type
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError( """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)
        self.decoder = nn.Linear(nhid, nout)
        #self.init_weights()
        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        nn.init.zeros_(self.decoder.weight)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, input, hidden = None):
        output, hidden = self.rnn(input, hidden)
        output = self.dropout(output)
        decoded = self.decoder(output)
        #decoded = decoded.view(-1, self.nout)
        return decoded, hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                    weight.new_zeros(self.nlayers, bsz, self.nhid))
        else:
            return weight.new_zeros(self.nlayers, bsz, self.nhid)

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

def map_location(cuda):
    if torch.cuda.is_available() and cuda:
        map_location=lambda storage, loc: storage.cuda()
    else:
        map_location='cpu'

def load_model_from_path(checkpoint_path, 
    tasks = ['EXPR', 'AU', 'VA'],
    nhid = 128,
    nlayers=1,
    dropout=0.5,
    use_cuda=True):
    
    bk = MobileFaceNet([112, 112], 136) 
    bk.remove_output_layer()
    ninp = 512 # 512 is the feature dimension for each task, produced by GDC module

    temporal_models = {}

    for t in tasks:
        if t == 'EXPR':
            dim = 7
        elif t == 'AU':
            dim = 12
        else:
            dim = 40 # digitize_num
        temporal_models[t] = RNNModel("GRU", ninp, nhid, nlayers, dim, dropout)

    STmodel = SpatialTemporalModel(bk, temporal_models, tasks)

    checkpoint = torch.load(checkpoint_path, map_location=map_location(use_cuda))
    checkpoint = checkpoint['state_dict']
    keys = ['tmodels.3.rnn.weight_ih_l0', 'tmodels.3.rnn.weight_hh_l0', 'tmodels.3.rnn.bias_ih_l0', 
    'tmodels.3.rnn.bias_hh_l0', 'tmodels.3.decoder.weight', 'tmodels.3.decoder.bias']
    for k in keys:
        del checkpoint[k]
    STmodel.load_state_dict(checkpoint)
    device = torch.device('cuda:0' if use_cuda else 'cpu')
    STmodel.to(device)
    STmodel.eval()
    return STmodel
def test_transforms(img_size):
    transform_list = [transforms.Resize(img_size),
                    transforms.ToTensor(),
                    ]
    return transforms.Compose(transform_list)

def logits_2_probas(preds, task, T=1):
    preds = preds/T
    if task == 'EXPR':
        p = softmax(preds, axis=-1)
    elif task =='AU':
        p = sigmoid(preds)
    elif task =='VA':
        p = [softmax(preds[..., :20], axis=-1), softmax(preds[..., 20:], axis=-1)]
        p = np.concatenate(p, axis=-1)
    return p
def probas_2_estimates(probas, task):
    if task == 'EXPR':
        est = probas.argmax(axis=-1).astype(np.int)
    elif task =='AU':
        est = (probas > 0.5).astype(np.int)
    elif task == 'VA':
        v, a = probas[..., :20], probas[..., 20:]
        bins = np.linspace(-1, 1, 20)
        v = (bins * v).sum(-1)
        a = (bins * a).sum(-1)
        est = np.stack([v, a], axis = -1)
    return est

def uncertainty_bar(probas, fill = '█', length=10):
    p = int(entropy(probas)/np.log2(probas.shape[-1]) * length)
    bar = fill * p + '-'*(length - p)
    return bar

def uncertainty(probas):
    return entropy(probas)/np.log2(probas.shape[-1])

def format_vis_text(probas):
    text = [("Task:  Prediction Uncertainty", -1)]
    for i, au_id in enumerate(categories['AU']):
        out = int(probas['AU'][i]>0.5)
        p = np.array([1-probas['AU'][i], probas['AU'][i]])
        text+=[("{}: {} ".format(au_id, out), uncertainty(p))]

    out = probas['EXPR'].argmax()
    out = categories['EXPR'][out]
    text+=[("{}: {}".format("EXPR", out), uncertainty(probas['EXPR']))]

    va = probas_2_estimates(probas['VA'], task='VA')
    v_u = uncertainty(probas['VA'][:20])
    a_u = uncertainty(probas['VA'][20:])
    text+=[("{}: {:.1f}".format("Valence", va[0]), v_u)]
    text+=[("{}: {:.1f}".format("Arousal", va[1]), a_u)]
    return text


####### main #######
def main():
    print("Models are loading...")
    ensemble = []
    tasks = ['EXPR', 'AU', 'VA']

    model_paths = ["../Training/checkpoints/student_round_3_exp_0/net_epoch_4_student_round_3_exp_0.pth"]
    for ckp_path in model_paths:
        ensemble.append(load_model_from_path(ckp_path, use_cuda= args.use_cuda))
    device = torch.device('cuda:0' if args.use_cuda else 'cpu')
    print("{} are loaded.".format(len(ensemble)))
    transform = test_transforms(112)
    print("Streaming started")
    video_capture = cv2.VideoCapture(0)
    # loop over frames from the video file stream
    hiddens= dict([(t, None) for t in tasks] )
    prev_faces = None
    while True:
        # grab the frame from the threaded video stream
        ret, frame = video_capture.read()
        frame = cv2.resize(frame, (600, 400))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        start_time = time.time()
        if args.face_detector =='cascade':
            faces = detector.detectMultiScale(gray, 
                                              scaleFactor= 1.1,
                                              minNeighbors= 5,
                                              minSize=(60, 60),
                                              flags=cv2.CASCADE_SCALE_IMAGE)
        else:
            boxes = detector.detect_faces(frame)
            if boxes:
                faces = (boxes[0]['box'],)
            else:
                faces = ()

        
        # convert the input frame from BGR to RGB 
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        if len(faces)>0:
            detected = True 
        else:
            if prev_faces is not None:
                faces = prev_faces
                detected = True 
            else:
                detected = False 
        detection_time = time.time() - start_time
        if detected:
            x, y, w, h = faces[0]
            #prev_faces = faces
            face_image = transform(Image.fromarray(frame[y:y+h, x:x+w]))
            #s = time.time()
            if args.use_cuda:
                face_image = torch.FloatTensor(face_image).cuda()
            #load_time = time.time() -s 
            face_image = face_image.unsqueeze(0).unsqueeze(0) # (1, 1, 3, 112, 112)
            
            preds = {}

            for model in ensemble:
                with torch.no_grad():
                    pred, _ = model(face_image)
                #recognition_time = time.time() - s - load_time
                for task in tasks:
                    if not args.use_cuda:
                        pred[task] = pred[task].squeeze().numpy()
                    else:
                        pred[task] = pred[task].squeeze().cpu().numpy()
                    if task not in preds:
                        preds[task] = []
                    preds[task].append(logits_2_probas(pred[task], task))
            
            #estimates = {}
            for task in tasks:
                preds[task] = np.stack(preds[task], axis=0).mean(0)
                #estimates[task] = probas_2_estimates(preds[task])
            processing_time = time.time() - start_time
            #recognition_time = processing_time - detection_time

            # visualize
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            text = format_vis_text(preds)
            dy = 20
            for i, t in enumerate(text):
                t, un = t
                cv2.putText(frame, t, (10, 25+i*dy), cv2.FONT_HERSHEY_SIMPLEX,
                     0.75, (0, 255, 0), 2)
                if not un==-1:
                    l = int(un * 100)
                    rl = 100 -l
                    dx = 175
                    cv2.line(frame, (10+dx, 25+i*dy), (10+dx+l, 25+i*dy), (0, 255, 0), 5)
                    cv2.line(frame, (10+dx+l, 25+i*dy), (10+dx+l+rl, 25+i*dy), (211, 211, 211), 5)

            cv2.putText(frame, "FPS:{} ".format(int(1/processing_time)), (450, 25),
               cv2.FONT_HERSHEY_SIMPLEX,  0.75, (0, 255, 0), 2 )
            if args.use_cuda:
                cv2.putText(frame, "gpu", (500, 380),
                   cv2.FONT_HERSHEY_SIMPLEX,  0.75, (0, 255, 0), 2 )
            else:
                cv2.putText(frame, "cpu", (500, 380),
                   cv2.FONT_HERSHEY_SIMPLEX,  0.75, (0, 255, 0), 2 )

        cv2.imshow("Frame", frame)
        if args.save_file is not None:
            out_video.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video_capture.release()
    cv2.destroyAllWindows()
    if args.save_file is not None:
        out_video.release()


if __name__ == '__main__':
    main()



