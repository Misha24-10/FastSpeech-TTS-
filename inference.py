import config
from scr.Dataset.dataset import LJSpeechDataset
from scr.colator.colator import LJSpeechCollator
from scr.featurizer.featurizer import MelSpectrogram
from config import ModelConfig, MelSpectrogramConfig
from scr.vocoder.Vocoder import Vocoder
from scr.aligner.aligner import GraphemeAligner
import wandb
from torch.utils.data import  Subset,DataLoader
from scr.model.model import make_model, WarmupWrapper

import torch.optim as optim
from itertools import islice
from tqdm import tqdm
from matplotlib import pyplot as plt
import torch.nn.functional as F
import torch
from torch import nn


def to_tokens(sentence):
    tokenizer = torchaudio.pipelines.TACOTRON2_GRIFFINLIM_CHAR_LJSPEECH.get_text_processor()
    tokens, token_lengths = tokenizer(sentence)
    return tokens, token_lengths 

def train(sentence):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataloader_train, dataloader_valid = get_dataloader()
    featurizer = MelSpectrogram(MelSpectrogramConfig()).to(device)
    vocoder = Vocoder().to(device).eval()
    aligner = GraphemeAligner().to(device)
    modelconfig = ModelConfig()

    wandb.config = {
        "model_config": modelconfig,
        "mel_config": MelSpectrogramConfig,
        "batch_size": config.batchSize
    }
    model = make_model(modelconfig.src_vocab, modelconfig.N, modelconfig.fft_hidden, modelconfig.blockconv_filtersize,
                       modelconfig.len_red_filtersize, modelconfig.mel_size, modelconfig.head, modelconfig.dropout).to(device)
    model.load_state_dict(torch.load(config.pathtoweights))

    aligner.eval()

    model.eval()
    tec = to_tokens(sentence)[0]
    torch.cat((tec,tec))
    spec_out, dur_out = model(torch.cat((tec,tec)).cuda(), )
    plt.imshow((spec_out[0]).cpu().detach().numpy())
    torchaudio.save("output_waveglow.wav", vocoder.inference(spec_out[0].unsqueeze(dim=0).cuda()).cpu(), sample_rate=22050)


if __name__ == '__main__':
    sentence = "A defibrillator is a device that gives a high energy electric shock to the heart of someone who is in cardiac arrest"
    train(sentence)

