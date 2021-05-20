
# The Backbone

The backbone is a facial landmark detection model from this [repo](https://github.com/cunjian/pytorch_face_landmark). The MobileNet model architecture is defined in this [script](https://github.com/cunjian/pytorch_face_landmark/blob/master/models/mobilefacenet.py), and the checkpoint is downloaded [here](https://github.com/cunjian/pytorch_face_landmark/blob/master/checkpoint/mobilefacenet_model_best.pth.tar).

# The algorithm

"Face alignment" is an auxillary task, which we keep it during training, but won't put too much attention on it. It means that we don't need uncertainty estimation for this task.

The dataset we use is only Aff-wild2 dataset, since it is large enough.

The emotion tasks (EXPR, FAU, VA) all need to predict the actual prediction and the uncertainty prediction, but this is for the student model. 

# The architecture

I will choose MobileFaceNet as the backbone, and feed the feature from the last conv layer to four Transformers or four RNNs. We can compare their efficiency and performance.

To be specific, the backbone is a CNN model, 
