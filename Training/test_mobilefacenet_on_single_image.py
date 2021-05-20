from models.mobilefacenet import MobileFaceNet
import torch
from PIL import Image
import numpy as np
import cv2
if torch.cuda.is_available():
    map_location=lambda storage, loc: storage.cuda()
else:
    map_location='cpu'

backbone = 'MobileFaceNet'
out_size = 112
img = cv2.imread('example_image/00014.jpg')


img = cv2.resize(img, (out_size, out_size))
face = img.transpose((2, 0, 1))
face = face.reshape((1,) + face.shape)/255.

if backbone=='MobileFaceNet':
    model = MobileFaceNet([112, 112],136)   
    checkpoint = torch.load('checkpoint/mobilefacenet_model_best.pth.tar', map_location=map_location)      
    print('Use MobileFaceNet as backbone') 
model.load_state_dict(checkpoint['state_dict'])
if torch.cuda.is_available():
    model.cuda()
model.eval()
input = torch.from_numpy(face).float()
input= torch.autograd.Variable(input)
input = input.cuda() if torch.cuda.is_available() else input
landmark = model(input)[0].cpu().data.numpy()[0]

landmark = landmark.reshape(-1, 2)
for x, y in landmark:
    cv2.circle(img, (int(x*out_size), int(y*out_size)), 2, (0,255,0), -1)

Image.fromarray(img[:, :, ::-1]).show()
