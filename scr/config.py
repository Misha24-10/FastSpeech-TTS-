from dataclasses import dataclass

class ModelConfig:
    src_vocab: int = 38
    N: int = 6
    fft_hidden: int = 384
    blockconv_filtersize: int = 1024
    len_red_filtersize: int = 256
    mel_size: int = 80
    head: int = 2
    dropout: float = 0.1

@dataclass
class MelSpectrogramConfig:
    sr: int = 22050
    win_length: int = 1024
    hop_length: int = 256
    n_fft: int = 1024
    f_min: int = 0
    f_max: int = 8000
    n_mels: int = 80
    power: float = 1.0

    # value of melspectrograms if we fed a silence into `MelSpectrogram`
    pad_value: float = -11.5129251

pathtoweights = '/content/drive/MyDrive/AUDIO_DLA/TTS/Full DS/light-leaf-11.pt'
datapath = '.'
batchSize = 10