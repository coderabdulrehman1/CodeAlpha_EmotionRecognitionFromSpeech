
import numpy as np
import librosa


SR = 22050          # sampling rate every clip is resampled to
DURATION = 3        # seconds of audio to keep
OFFSET = 0.5        # skip the first 0.5 s (usually silence)
N_MFCC = 40         # number of MFCC coefficients per frame
HOP_LENGTH = 512    # librosa default: 512 samples between frames
MAX_FRAMES = 130    # 3 s * 22050 / 512 = ~129 frames, so 130 covers it


def mfcc_from_signal(y, sr=SR):
    """Turn a waveform (1D array) into a fixed-size MFCC matrix of shape (N_MFCC, MAX_FRAMES)."""
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, hop_length=HOP_LENGTH)  # (40, frames)

    if mfcc.shape[1] < MAX_FRAMES:                      
        pad_width = MAX_FRAMES - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)))

    return mfcc[:, :MAX_FRAMES].astype(np.float32)       


def extract_mfcc(path):
    """Load an audio file and return its MFCC matrix of shape (N_MFCC, MAX_FRAMES)."""
    y, _ = librosa.load(path, sr=SR, duration=DURATION, offset=OFFSET)
    return mfcc_from_signal(y)
