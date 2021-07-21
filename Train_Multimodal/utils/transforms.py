import torchvision.transforms as transforms
from torchaudio.transforms import AmplitudeToDB, MelSpectrogram
import math 
def get_audio_transforms():
    audio_transform = transforms.Compose([
      AmplitudeToDB('power', 80),
      transforms.Normalize(mean=[-14.8], std=[19.895]),
      ])
    return audio_transform

def get_meltransform(seq_len_secs, sample_rate, window_size = 20e-3, window_stride = 2e-3):
    num_fft = 2 ** math.ceil(math.log2(window_size * sample_rate))
    meltransform = MelSpectrogram(sample_rate=sample_rate, n_mels=64,
                  n_fft=num_fft,
                  win_length=int(window_size * sample_rate),
                  hop_length=int(window_stride
                                 * sample_rate))
    return meltransform

def train_transforms(img_size):
    transform_list = [transforms.Resize(img_size),
                      transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
                      transforms.RandomHorizontalFlip(),
                      transforms.ToTensor(),
                    ]
    return transforms.Compose(transform_list)

def test_transforms(img_size):
    transform_list = [transforms.Resize(img_size),
                    transforms.ToTensor(),
                    ]
    return transforms.Compose(transform_list)