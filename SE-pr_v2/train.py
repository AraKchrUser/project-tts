from typing import *
from timeit import default_timer as timer
from pathlib import Path
from string import punctuation

import librosa
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchaudio.transforms import Resample
from transformers import HubertModel
import whisperx

from models import TextEncoder, TransformerDecoder, Seq2Seq, WhisperX
from dataset import Text2PseudoPhonemes, CoquiTTSTokenizer
from cutils import load_checkpoint, wip_memory


def path(_path):
    return "./examples/" / Path(_path)


BATCH_SIZE = 32
DEVICE     = "cuda" #"cpu"
NUM_EPOCHS = 200
HUBERT_SR  = 16_000
HUBERT_PRETRAIN  = "/mnt/storage/kocharyan/hfmodels/content-vec-best" #"facebook/hubert-large-ls960-ft"




def calculate_wer_with_alignment(reference_text: str, recognized_text: str):
    
    remove_punctuation = lambda string: ''.join(filter(lambda sym: sym not in punctuation, string.lower().strip())).split()
    reference_words = remove_punctuation(reference_text)
    recognized_words = remove_punctuation(recognized_text)

    # расстояние Левенштейна 
    
    # Инициализация матрицы для подсчета расстояния между словами
    distance_matrix = [[0] * (len(recognized_words) + 1) for _ in range(len(reference_words) + 1)]
    # Наполнение первой строки матрицы
    for i in range(len(reference_words) + 1):
        distance_matrix[i][0] = i

    # Наполнение первого столбца матрицы
    for j in range(len(recognized_words) + 1):
        distance_matrix[0][j] = j

    # Заполнение матрицы расстояний методом динамического программирования
    for i in range(1, len(reference_words) + 1):
        for j in range(1, len(recognized_words) + 1):
            if reference_words[i - 1] == recognized_words[j - 1]:
                distance_matrix[i][j] = distance_matrix[i - 1][j - 1]
            else:
                insert = distance_matrix[i][j - 1] + 1
                delete = distance_matrix[i - 1][j] + 1
                substitute = distance_matrix[i - 1][j - 1] + 1
                distance_matrix[i][j] = min(insert, delete, substitute)

    # Расчет WER  (в процентах)
    wer = distance_matrix[-1][-1] / len(reference_words) * 100
    
    ali = [[] for _ in range(3)]
    correct = 0
    insertion = 0
    substitution = 0
    deletion = 0
    i, j = len(reference_words), len(recognized_words)
    while True:
        if i == 0 and j == 0:
            break
        elif (i >= 1 and j >= 1
              and distance_matrix[i][j] == distance_matrix[i - 1][j - 1] 
              and reference_words[i - 1] == recognized_words[j - 1]):
            ali[0].append(reference_words[i - 1])
            ali[1].append(recognized_words[j - 1])
            ali[2].append('C')
            correct += 1
            i -= 1
            j -= 1
        elif j >= 1 and distance_matrix[i][j] == distance_matrix[i][j - 1] + 1:
            ali[0].append("***")
            ali[1].append(recognized_words[j - 1])
            ali[2].append('I')
            insertion += 1
            j -= 1
        elif i >= 1 and j >= 1 and distance_matrix[i][j] == distance_matrix[i - 1][j - 1] + 1:
            ali[0].append(reference_words[i - 1])
            ali[1].append(recognized_words[j - 1])
            ali[2].append('S')
            substitution += 1
            i -= 1
            j -= 1
        else:
            ali[0].append(reference_words[i - 1])
            ali[1].append("***")
            ali[2].append('D')
            deletion += 1
            i -= 1
    
    ali[0] = ali[0][::-1]
    ali[1] = ali[1][::-1]
    ali[2] = ali[2][::-1]
    
    assert len(ali[0]) == len(ali[1]) == len(ali[2]), f"wrong ali {ali}"
    
    return {"wer" : wer,
            "cor": correct, 
            "del": deletion,
            "ins": insertion,
            "sub": substitution,
            "ali": ali,
            "reference_words": reference_words,
            "recognized_words": recognized_words,
            }


def train_epoch(model: Seq2Seq, optimizer: Any, 
                loss_fn: Any, dataset: Dataset):
    model.train()
    losses = []

    dataset_loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, collate_fn=dataset.collate_fn
        )
    for batch in dataset_loader:
        
        tokens_padded = batch["tokens_padded"].to(DEVICE)
        text_lens     = batch["text_lens"].to(DEVICE)
        lables        = batch['lables'].to(DEVICE)

        logits = model(tokens_padded, text_lens, lables[:, :-1])

        optimizer.zero_grad()
        # print(lables.shape, lables[:, :-1].shape, lables[:, 1:].shape, logits.shape)
        loss = loss_fn(
            logits.reshape(-1, logits.shape[-1]), 
            lables[:, 1:].reshape(-1)
        )
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
    
    return losses


def init_textencoder(dataset):
    N_VOCAB         = len(dataset.tokenizer.characters.vocab) # 177
    inter_channels  = 192
    hidden_channels = 192
    filter_channels = 768
    n_heads         = 2
    n_layers        = 10 # 6/10 for YourTTS
    kernel_size     = 3
    p_dropout       = .1
    return TextEncoder(
        N_VOCAB, inter_channels, hidden_channels, filter_channels, 
        n_heads, n_layers, kernel_size, p_dropout,
    )


def init_decoder(dataset):
    num_layers     = 3
    emb_size       = 192
    dim_ff         = 300
    nhead          = 1
    tgt_vocab_size = dataset.gen_pad + 1 #TODO CHECKME
    dropout        = .1
    gen_pad        = dataset.gen_pad
    gen_bos        = dataset.gen_bos
    gen_eos        = dataset.gen_eos
    return TransformerDecoder(
        num_layers, emb_size, dim_ff, nhead, tgt_vocab_size, 
        dropout, gen_pad, gen_bos, gen_eos
    )


def init_dataset():
    return Text2PseudoPhonemes(
        path("rudevices_chunk"), path("extracted_contents"), path("clusters/clusters.pt"), 
        None, None, "ckpts/yourrtts_config.json",
    )

def train():

    #TODO: CHENGEME
    dataset = init_dataset()

    prior_encoder = init_textencoder(dataset)
    prior_encoder, _ = load_checkpoint(prior_encoder,
                                "ckpts/yourtts_ruslan.pth", 
                                "text_encoder", False)
    for param in prior_encoder.parameters():
        param.requires_grad = True

    decoder = init_decoder(dataset)
    
    model = Seq2Seq(prior_encoder, decoder)
    model.to(DEVICE)
    
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9
        )
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=dataset.gen_pad) #TODO: PAD

    losses = []
    for epoch in range(NUM_EPOCHS):
        start_time = timer()
        _losses = train_epoch(model, optimizer, loss_fn, dataset)
        end_time = timer()
        if epoch % 50 == 0:
            print(f"Epoch: {epoch+1}, Train loss: {np.mean(_losses)}, Time: {(end_time - start_time)/60:.3f} min")
        losses.append(np.mean(_losses))
    
    return losses, model, dataset

def speech_editing(audio_f, src_text, target_text, dataset, model=None): #TODO
    # Idea:
    # 1)
    # берем все слова которые были заменены
    # считаем для них контент вектора
    # через виспер заменяем их
    # 2) 
    # Все переводим в центроиды 
    # Потом выделяем слова, которые были заменены 
    

    # get contents
    hubert = HubertModel.from_pretrained(HUBERT_PRETRAIN).to(DEVICE)
    audio, sr = librosa.load(audio_f, mono=True)
    audio     = torch.from_numpy(audio).float().to(DEVICE)
    if sr != HUBERT_SR:
            audio = Resample(sr, HUBERT_SR).to(audio.device)(audio).to(DEVICE)
    if audio.ndim == 1: audio = audio.unsqueeze(0)
    with torch.no_grad():
        contents = hubert(audio).last_hidden_state
    print("DEBUG", contents.shape)
    torch.cuda.empty_cache()
    wip_memory(hubert)

    # get word timestamps
    audio  = WhisperX.load_audio(audio_f)
    whisperx_model = WhisperX() 
    out = whisperx_model(audio)
    alignment = WhisperX.postprocess_out(out, by='words')
    timesteps = WhisperX.formed_timesteps(alignment)
    print("DEBUG", timesteps)

    if src_text is None:
        src_text = out['segments'][0]['text']
    
    if dataset is None:
        dataset = init_dataset()

    #use wer for ali

    # wer_info = calculate_wer_with_alignment(src_text, target_text)
    # ali       = wer_info["ali"][2]
    # tgt_words = wer_info["recognized_words"]
    # for i in range(len(ali)):
    #     if not ali[i] == "S":
    #         continue


    # preprocessing text ?
    token_ids = dataset.encode(target_text)
    text_len  = len(target_text)

    if isinstance(model, str):
        model_p = model
        encoder, decoder = init_textencoder(dataset), init_decoder(dataset)
        model = Seq2Seq(encoder, decoder)
        model.load_state_dict(torch.load(model_p))
        model.eval()
    preds = greedy_decoding(model, token_ids, text_len, len(token_ids) + 200, dataset.gen_bos, dataset.gen_eos)
    preds = preds.cpu().numpy()[1:-1]

    # TODO: replace idx by embeds
    return preds


def greedy_decoding(model, src, src_len, max_len, start_symbol, end_symbol):  #CHECK ME
    memory = model.only_encode(src, src_len)

    # TODO: Gen subsequent_mask
    preds = torch.ones(1, 1) #tensor([[1.]])
    preds = preds.fill_(start_symbol).type(torch.long).to(DEVICE)
    for i in range(max_len-1):
        memory.to(DEVICE)
        out = model.only_decode(preds, memory) # tgt mask gen
        # out = out.transpose(0, 1) #?
        prob = model.decoder.generator(out[:,-1])
        _, next_word = torch.max(prob, dim=1) #?
        next_word = next_word.item()

        added = torch.ones(1, 1).type_as(src.data).fill_(next_word)
        preds = torch.cat([preds, added], dim=0)

        if next_word == end_symbol:
            break
    
    return preds


if __name__ == "main":
    train()
    
