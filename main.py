import scr.config
from scr.Dataset.dataset import LJSpeechDataset
from scr.colator.colator import LJSpeechCollator
from scr.featurizer.featurizer import MelSpectrogram
from scr.config import ModelConfig, MelSpectrogramConfig
from scr.vocoder.Vocoder import Vocoder
from scr.aligner.aligner import GraphemeAligner
import torch
import wandb
from torch.utils.data import  Subset,DataLoader
from scr.model.model import *
import torch.optim as optim
from itertools import islice
from tqdm import tqdm


def get_dataloader(datapath='.', batchSize=10):
    dataset = LJSpeechDataset(datapath)
    train_ratio = 0.9
    torch.manual_seed(41)
    train_size = int(len(dataset) * train_ratio)
    indexes = torch.randperm(len(dataset))
    train_indexes = indexes[:train_size]
    validation_indexes = indexes[train_size:]
    train_dataset = Subset(dataset, train_indexes)
    validation_dataset = Subset(dataset, validation_indexes)
    dataloader_train = DataLoader(train_dataset, batch_size = batchSize, collate_fn = LJSpeechCollator())
    dataloader_valid = DataLoader(validation_dataset, batch_size = batchSize, collate_fn = LJSpeechCollator())
    return dataloader_train, dataloader_valid


def train():
    wandb.login(key='358c4114387c5c7ca207c32ba4343e7c86efc182')
    wandb.init(project='TTS', entity='mishaya') # username in wandb
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataloader_train, dataloader_valid = get_dataloader()
    featurizer = MelSpectrogram(MelSpectrogramConfig()).to(device)
    vocoder = Vocoder().to(device).eval()
    aligner = GraphemeAligner().to(device)
    modelconfig = ModelConfig()

    wandb.config = {
        "model_config": modelconfig,
        "mel_config": MelSpectrogramConfig,
        "batch_size": scr.config.batchSize
    }
    model = make_model(modelconfig.src_vocab, modelconfig.N, modelconfig.fft_hidden, modelconfig.blockconv_filtersize,
                       modelconfig.len_red_filtersize, modelconfig.mel_size, modelconfig.head, modelconfig.dropout).to(device)
    loss_one = nn.MSELoss()
    loss_second = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), 0.0003)
    optimizer = WarmupWrapper(1000, optimizer, 0.0006)
    model.load_state_dict(torch.load(scr.config.pathtoweights))

    aligner.eval()
    for epoch in range(10):  # loop over the dataset multiple times
        model.train()
        for i, data in islice(enumerate(tqdm(dataloader_train)),len(dataloader_train)):
            durations = aligner(data.waveform.to("cuda"), data.waveforn_length, data.transcript).to(device)
            inputs_text = data.tokens.to(device)
            inputs_text_len = data.token_lengths.to(device)
            inputs_wavs = data.waveform.to(device)
            inputs_wavs_len = data.waveforn_length.to(device)

            specs_GT = featurizer(inputs_wavs)

            if durations.shape != inputs_text.shape:
                print(" log unequal dim")
            durations = (durations[:,:inputs_text.shape[-1]])
            durations = (durations / durations.sum(dim=1)[:, None])
            durations = durations * inputs_wavs_len[:, None]
            durations = durations.cumsum(dim=1).int()
            durations = torch.ceil(durations * specs_GT.shape[-1] / inputs_wavs_len.max())
            for j in range(len(durations)):
                row = torch.unique_consecutive(durations[j])
                durations[j] = F.pad(row, (0, durations.shape[-1] - row.shape[-1]), "constant", 0)
            for j in range(durations.shape[0]):
                pred = 0
                for k in range(durations.shape[-1]):
                    curr = durations[j][k]
                    if pred != 0 and curr != 0:
                        durations[j][k] -= pred
                    pred += curr


            optimizer.zero_grad()
            spec_out, dur_out = model(inputs_text, durations)
            groud_true = torch.log(durations.clamp(1e-5))
            try:
                loss = loss_one(dur_out.squeeze(), groud_true) + loss_second(spec_out, specs_GT)
            except:
                print("continue\n\n")
                continue
            print(" ",loss.cpu())
            wandb.log({"loss": loss})

            if ( i  % 50 == 0  ):
                wandb.log({
                    "Audio valid every n step": [wandb.Audio(vocoder.inference(spec_out[0].unsqueeze(dim=0)).cpu()[0], caption="Audio in train", sample_rate=22050)],
                    "Spec train ": [wandb.Image(plt.imshow((spec_out[0]).cpu().detach().numpy()), caption="Spec train ")],
                    "Spec train GroundTrue": [wandb.Image(plt.imshow((specs_GT[0].cpu())), caption="Spec train GroundTrue")]
                })
            loss.backward()
            optimizer.step()
        model.eval()

        for i, data in islice(enumerate(tqdm(dataloader_valid)),50):
            durations = aligner(data.waveform.to("cuda"), data.waveforn_length, data.transcript).to(device)
            inputs_text = data.tokens.to(device)
            inputs_text_len = data.token_lengths.to(device)
            inputs_wavs = data.waveform.to(device)
            inputs_wavs_len = data.waveforn_length.to(device)
            specs_GT = featurizer(inputs_wavs)

            if durations.shape != inputs_text.shape:
                print(" log unequal dim")

            durations = (durations[:,:inputs_text.shape[-1]])
            durations = (durations / durations.sum(dim=1)[:, None])
            durations = durations * inputs_wavs_len[:, None]
            durations = durations.cumsum(dim=1).int()
            durations = torch.ceil(durations * specs_GT.shape[-1] / inputs_wavs_len.max())

            for j in range(len(durations)):
                row = torch.unique_consecutive(durations[j])
                durations[j] = F.pad(row, (0, durations.shape[-1] - row.shape[-1]), "constant", 0)
            for j in range(durations.shape[0]):
                pred = 0
                for k in range(durations.shape[-1]):
                    curr = durations[j][k]
                    if pred != 0 and curr != 0:
                        durations[j][k] -= pred
                    pred += curr

            spec_out, dur_out = model(inputs_text, durations)
            groud_true = torch.log(durations.clamp(1e-5))
            try:
                loss = loss_one(dur_out.squeeze(), groud_true) + loss_second(spec_out, specs_GT)
            except:
                print("continue\n\n")
                continue

            print(" ",float(loss))
            wandb.log({"loss In validation": loss})

            if ( i  % 8 == 0  ):
                wandb.log({
                    "Audio In validation": [wandb.Audio(vocoder.inference(spec_out[0].unsqueeze(dim=0)).cpu()[0], caption="Audio In validation", sample_rate=22050)],
                    "Spec In validation": [wandb.Image(plt.imshow((spec_out[0]).cpu().detach().numpy()), caption="Spec In validation")],
                    "Spec GroundTrue In validation": [wandb.Image(plt.imshow((specs_GT[0].cpu())), caption="Spec GroundTrue In validation")]
                })
if __name__ == '__main__':
    train()

