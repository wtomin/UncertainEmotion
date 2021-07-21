# Efficient Multitask Emotion Network (EMENet)

This is a repository for our solution for the [ABAW2021 Challenge](https://ibug.doc.ic.ac.uk/resources/iccv-2021-2nd-abaw/). Our team name is NISL-2021.

We trained unified models to predict three types of emotion labels, i.e., 12 facial action units, 7 basic emotions, valence and arousal. The 12 action units are AU1, AU2, AU4, AU6, AU7, AU10, AU12, AU15, AU23, AU24, AU25, AU26. The seven basic emotions are neutral, anger, disgust, fear, happiness, sadness and surprise. Valence and arousal are both continuous values in the range [-1, 1].

Our models have efficient CNN-RNN architectures. We list the number of parameters of our visual model and visual-audio model as follows:

| Model | # Param. | FLOPs|
| --- | ---| ---|
|EMENet-V| 1.68M| 228M|
|EMENet-VA|1.91M | 234M| 

Note that the FLOPs are the number of floating-point operations when the visual input is one RGB image (112x112) and audio input is one mel spectrogram (64x64). Our model can accept a sequence of facial images and a sequence of spectrograms.

We not only trained single models, but also trained deep ensembles. A deep ensemble consists of several models with the same architecture, but different random initialization. It has been proved to be robust in uncertainty estimation ([Ovidia et al., 2019](https://arxiv.org/abs/1906.02530)). We applied deep ensembles for emotion uncertainty estimation.

# Requirements

1. Python3.9
2. CUDA 11.0
3. Install other requirements using

```
pip install requirements.txt
```
4. When using audio-visual models, [NeMo](https://github.com/NVIDIA/NeMo) is required. Install it using:

```
apt-get update && apt-get install -y libsndfile1 ffmpeg
pip install Cython
pip install nemo_toolkit['all']
```

# Usage

# Cite



