import torchaudio

def read_audio(audio_file):
    if torchaudio.get_audio_backend() in ['soundfile', 'sox_io']:
        out, sr = torchaudio.load(audio_file,
                                  frame_offset=0,
                                  num_frames=-1)
    else:
        out, sr = torchaudio.load(audio_file, offset=0, num_frames=-1)
    return out, sr