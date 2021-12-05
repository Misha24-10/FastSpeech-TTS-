from torch import nn
import torch
import torch.nn.functional as F
import copy
import math



class Embed(nn.Module):
    def __init__(self,num_embeddings, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings,embedding_dim)
    def forward(self, batch):
        return self.embedding(batch)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

def clones(module, N):
    "Копирут блоки N раз"
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class LayerNorm(nn.Module):
    """Модуль для нормироки слоя"""

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class Encoder(nn.Module):
    "Базовый кодировщик представляет собой стек из N слоев."

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x):
        "Пропускает входные данные через каждый слой"
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


def attention(query, key, value, dropout=None):
    "Вычисляет 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    p_attn = nn.functional.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Создант self-attention с h головами"
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value):
        "Implements Figure 2"
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class Conv1d(nn.Module):
    "Вычисляет FFN блок, часть со свертками."

    def __init__(self, d_model=384, hiden=1536, kernal_size=3, dropout=0.1):
        super(Conv1d, self).__init__()
        self.conv1 = nn.Conv1d(d_model, hiden, kernal_size, padding=1)
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(hiden, d_model, kernal_size, padding=1)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        x = torch.permute(x, (0, 2, 1))
        x = self.dropout1(F.relu(self.conv1(x)))
        x = self.dropout2(F.relu(self.conv2(x)))
        x = torch.permute(x, (0, 2, 1))
        return x


class lenRegulator(nn.Module):
    "Модуль для регулировки длиины"

    def __init__(self, in_channels=384, filter_size=256, kernal_size=3, dropout=0.1):
        super(lenRegulator, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, filter_size, kernal_size, padding=1)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = LayerNorm(filter_size)
        self.conv2 = nn.Conv1d(filter_size, filter_size, kernal_size, padding=1)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = LayerNorm(filter_size)
        self.linear = nn.Linear(filter_size, 1)

    def forward(self, x):
        x = torch.permute(x, (0, 2, 1))
        x = self.conv1(x)
        x = self.norm1(torch.permute(x, (0, 2, 1)))
        x = self.dropout1(F.relu(x))
        x = self.conv2(torch.permute(x, (0, 2, 1)))
        x = self.norm2(torch.permute(x, (0, 2, 1)))
        x = self.dropout2(F.relu(torch.permute(x, (0, 2, 1))))
        x = torch.permute(x, (0, 2, 1))
        x = self.linear(x)
        return x


class Linear_Layer(nn.Module):
    "Последний слой модели, выдает блоки [B, n_mels, Time]"

    def __init__(self, in_channels=384, filter_size=80, dropout=0.1):
        super(Linear_Layer, self).__init__()
        self.linear = nn.Linear(in_channels, filter_size)

    def forward(self, x):
        x = self.linear(x)
        return x


class SublayerConnection(nn.Module):
    """
    Модуль перебрски весов( residual connection)
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        out = self.norm(x)
        out = sublayer(out)
        out = self.dropout(out)
        return x + out


class EncoderLayer(nn.Module):
    "один FFT Block "

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x):
        "-----------"
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x))
        return self.sublayer[1](x, self.feed_forward)


class Model(nn.Module):
    """
    Feed-Forward Transformer модель
    """

    def __init__(self, embedding, position_emb, encoder, duration_predictor, decoder, position_emb2, generator):
        super(Model, self).__init__()

        self.embedding = embedding
        self.position_emb = position_emb

        self.encoder = encoder
        self.duration_predictor = duration_predictor
        self.decoder = decoder
        self.position_emb2 = position_emb2
        self.generator = generator

    def forward(self, src, input_dur_inframe=None, device="cuda"):
        "-----"
        encode = self.encode(src)

        if input_dur_inframe is not None:
            input_dur_inframe = input_dur_inframe.long()
            outp = torch.zeros((src.shape[0], torch.sum(input_dur_inframe, dim=-1).max(), encode.shape[-1])).to(
                device)  # [5, Time, 384]
            N, L = input_dur_inframe.shape
            for j in range(N):
                count = 0
                for k in range(L):
                    kol = input_dur_inframe[j][k]
                    if kol != 0:
                        outp[j, count:count + kol] = encode[j, k]
                        count += kol
            decoder = self.decoder(outp)
        else:
            decoder = self.decoder(encode)
        duration_predictor = self.duration_predictor(encode)
        mel_spec = torch.permute(self.generator(decoder), (0, 2, 1))
        return mel_spec, duration_predictor  # reruen both mel_pred and duration_pred

    def encode(self, src):
        x = self.position_emb(self.embedding(src))
        return self.encoder(x)

    def decode(self, src):
        return self.decoder(self.position_emb2(src))

def make_model(src_vocab = 38, N=2,
               fft_hidden=192, blockconv_filtersize = 768, len_red_filtersize = 128, mel_size = 80 ,head=2, dropout=0.1): #Параметры для обучения на 1 батче
    "Helper: Construct a model from hyperparameters."
    copy_of_this_layer = copy.deepcopy
    attn = MultiHeadedAttention(head, fft_hidden)
    conv1d = Conv1d(d_model = fft_hidden, hiden = blockconv_filtersize)
    embedding  = Embed(src_vocab, fft_hidden)
    position_emb = PositionalEncoding(fft_hidden, dropout)
    position_emb2 = PositionalEncoding(fft_hidden, dropout)
    model = Model(
        embedding,
        position_emb,
        Encoder(EncoderLayer(fft_hidden, copy_of_this_layer(attn), copy_of_this_layer(conv1d), dropout), N),
        lenRegulator(fft_hidden, len_red_filtersize),
        Encoder(EncoderLayer(fft_hidden, copy_of_this_layer(attn), copy_of_this_layer(conv1d), dropout), N),
        position_emb2,
        Linear_Layer(fft_hidden, mel_size),
    )
    # for p in model.parameters():
    #     if p.dim() > 1:
    #         nn.init.xavier_uniform(p)
    return model

class WarmupWrapper:
    def __init__(self, warmup: int, optimizer: torch.optim.Optimizer, max_lr: float) -> None:
        self.optimizer = optimizer
        self._step = 0
        self._rate = 0
        self.max_lr = max_lr
        self.warmup = warmup
        self._lrs = (torch.arange(start=0, end=warmup) / warmup) * max_lr

    def state_dict(self):
        return {key: value for key, value in self.dict.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        self.dict.update(state_dict)

    def step(self):
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step = None):
        if step is None:
            step = self._step
        if step >= self.warmup:
            return self.max_lr
        return self._lrs[step]

    def zero_grad(self):
        self.optimizer.zero_grad()