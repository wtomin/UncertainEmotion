import torchaudio

def read_audio(audio_file, offset, num_frames, normalization = True):
    if torchaudio.get_audio_backend() in ['soundfile', 'sox_io']:
        out, sr = torchaudio.load(audio_file,
                                  frame_offset=offset,
                                  num_frames=num_frames)
    else:
        out, sr = torchaudio.load(audio_file, offset=offset, num_frames=num_frames)
    return out, sr