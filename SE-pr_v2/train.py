from typing import *
from timeit import default_timer as timer
from pathlib import Path
from string import punctuation
import enum

import librosa
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchaudio.transforms import Resample
from transformers import HubertModel
import whisperx

from models import TextEncoder, TransformerDecoder, Seq2Seq, WhisperX, SimpleSeq2SeqTransformer
from dataset import Text2PseudoPhonemes, CoquiTTSTokenizer, Text2SemanticCode
from cutils import load_checkpoint, wip_memory
from se_infer import SVCInfer
from calc_content import calc_hubert_content, get_hubert_model


def path(_path):
    return "./examples/" / Path(_path)


class Exp(enum.Enum):
    '''Use: Exp.simple_transformer == EXP_CODE'''
    yourtts_encoder = 1
    simple_transformer = 2


LR = 0.0001
DATASET_SR = 16_000
BATCH_SIZE = 320
DEVICE     = "cuda" #"cpu"
NUM_EPOCHS = 2_000
HUBERT_SR  = 16_000
HUBERT_PRETRAIN  = "/mnt/storage/kocharyan/hfmodels/content-vec-best" #"facebook/hubert-large-ls960-ft"
MODEL_TYPE = 1
EXP_CODE = Exp.simple_transformer

if Exp.simple_transformer == EXP_CODE:
    print("You run simple transformer exp")
else:
    print("You run yourtts-encoder exp")



def calculate_wer_with_alignment(reference_text: str, recognized_text: str):
    '''Функция вычисления замененных слов'''
    
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


def create_masks(src, tgt):
    src_seq_len = src.shape[1]
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=DEVICE)
    tgt_seq_len = tgt.shape[1]
    tgt_mask = (torch.triu(
        torch.ones((tgt_seq_len, tgt_seq_len), device=DEVICE)
        ) == 1).float()
    tgt_mask = tgt_mask.masked_fill(tgt_mask == 0, float('-inf'))
    tgt_mask = tgt_mask.masked_fill(tgt_mask == 1, float(0.0))
    return src_mask.type(torch.bool), tgt_mask.transpose(0,1)

def create_padding(src, tgt, t_pad_id, s_pad_id):
    # t_pad_id = dataset.text_tokenizer.pad_id
    # s_pad_id = dataset.semantic_codes_clusters.pad_id
    src_padding_mask = (src == t_pad_id)#.transpose(0,1)
    tgt_padding_mask = (tgt == s_pad_id)#.transpose(0,1)
    return src_padding_mask, tgt_padding_mask


def train_epoch(model: Union[Seq2Seq, SimpleSeq2SeqTransformer], 
                optimizer: Any, loss_fn: Any, dataset: Dataset):
    '''Одна эпоха обучения.'''
    model.train()
    losses = []

    dataset_loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, collate_fn=dataset.collate_fn,
        )
    for batch in dataset_loader: #TODO: CHECKME
        
        if EXP_CODE == Exp.yourtts_encoder:
            
            tokens_padded = batch["tokens_padded"].to(DEVICE)
            text_lens     = batch["text_lens"].to(DEVICE)
            lables        = batch['lables'].to(DEVICE)
            
            #TODO: check model
            tgt_input = lables[:, :-1]
            _, tgt_mask = create_masks(tokens_padded, tgt_input)
            # t_pad_id = s_pad_id = model.gen_pad
            # src_padding_mask, tgt_padding_mask  = create_padding(
            #     tokens_padded, tgt_input, 
            #     t_pad_id, s_pad_id
            #     )
            logits = model(
                tokens_padded=tokens_padded, text_lens=text_lens, 
                lables=tgt_input, tgt_mask=tgt_mask,
                )
            logits = logits.reshape(-1, logits.shape[-1])
            lables = lables[:, 1:].reshape(-1)
        
        if EXP_CODE == Exp.simple_transformer:
            
            tokens_padded = batch["tokens_padded"].to(DEVICE)
            lables = batch['lables'].to(DEVICE)
            
            tgt_input = lables[:, :-1]
            src_mask, tgt_mask = create_masks(tokens_padded, tgt_input)
            t_pad_id = dataset.text_tokenizer.pad_id
            s_pad_id = dataset.semantic_codes_clusters.pad_id
            src_padding_mask, tgt_padding_mask  = create_padding(
                tokens_padded, tgt_input, 
                t_pad_id, s_pad_id,
                )
            logits = model(
                src=tokens_padded, tgt=tgt_input,
                src_padding_mask=src_padding_mask, tgt_padding_mask=tgt_padding_mask,
                src_mask=src_mask, tgt_mask=tgt_mask,
                )
            logits = logits.reshape(-1, logits.shape[-1])
            lables = lables[:, 1:].reshape(-1)

        optimizer.zero_grad()
        # print(lables.shape, lables[:, :-1].shape, lables[:, 1:].shape, logits.shape)
        loss = loss_fn(logits, lables)
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
    assert tgt_vocab_size == 103
    return TransformerDecoder(
        num_layers, emb_size, dim_ff, nhead, tgt_vocab_size, 
        dropout, gen_pad, gen_bos, gen_eos
    )


def init_dataset():
    if EXP_CODE == Exp.simple_transformer:
        print("Text2SemanticCode dataset prepared")
        return Text2SemanticCode(
            texts_path=path("rudevices_chunk"), contents_path=path("extracted_contents"), 
            clusters_path=path("clusters/clusters.pt"), tokenizer_conf=None, dsrate=DATASET_SR,
        )
    elif EXP_CODE == Exp.yourtts_encoder:
        return Text2PseudoPhonemes(
            path("rudevices_chunk"), path("extracted_contents"), path("clusters/clusters.pt"), 
            None, None, "ckpts/yourrtts_config.json",
        )

def init_simple_transformer(dataset):
    num_enc = 3
    num_dec = 3
    emb_size = 128
    nhead = 4
    dim_ff = 256
    src_vocab = dataset.text_tokenizer.size
    tgt_vocab = dataset.semantic_codes_clusters.vocab_size
    assert src_vocab == 42
    assert tgt_vocab == 103
    model = SimpleSeq2SeqTransformer(
        num_enc=num_enc, num_dec=num_dec, emb_size=emb_size, nhead=nhead, dim_ff=dim_ff,
        src_vocab=src_vocab, tgt_vocab=tgt_vocab,
    )
    model = model.to(DEVICE)
    pad = dataset.semantic_codes_clusters.pad_id
    eos = dataset.semantic_codes_clusters.eos_id
    bos = dataset.semantic_codes_clusters.bos_id
    return model, pad, eos, bos

def train():

    dataset = init_dataset()

    if EXP_CODE == Exp.yourtts_encoder:
        prior_encoder = init_textencoder(dataset)
        prior_encoder, _ = load_checkpoint(prior_encoder,
                                    "ckpts/yourtts_ruslan.pth", 
                                    "text_encoder", False)
        for param in prior_encoder.parameters(): #TODO: CHANGEME
            param.requires_grad = True

        decoder = init_decoder(dataset)
        
        model = Seq2Seq(prior_encoder, decoder)
        model = model.to(DEVICE)

        pad = dataset.gen_pad
        print("YourTTS encoder prepared")
    
    elif EXP_CODE == Exp.simple_transformer:
        model, pad, eos, bos = init_simple_transformer(dataset=dataset)
        print("SimpleSeq2SeqTransformer prepared")
    
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LR, betas=(0.9, 0.98), eps=1e-9,
        )
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=pad) #TODO: PAD

    losses = []
    for epoch in range(NUM_EPOCHS):
        start_time = timer()
        _losses = train_epoch(model, optimizer, loss_fn, dataset)
        end_time = timer()
        if epoch % 25 == 0:
            print(f"Epoch: {epoch+1}, Train loss: {np.mean(_losses)}, Time: {(end_time - start_time)/60:.3f} min")
        losses.append(np.mean(_losses))
    
    return losses, model, dataset

def speech_editing(audio_f, src_text, target_text, dataset, model=None): #TODO
    # Idea:
    # 1)
    # берем все слова которые были заменены
    # считаем для них центры - контент вектора
    # через виспер определяем границы и заменяем их
    # 2) 
    # Все переводим в центроиды 
    # Потом выделяем слова, которые были заменены 
    # Делаем патч

    # get contents
    #V1
    # hubert = HubertModel.from_pretrained(HUBERT_PRETRAIN).to(DEVICE)
    # audio, sr = librosa.load(audio_f, mono=True)
    # audio = torch.from_numpy(audio).float().to(DEVICE)
    # if sr != HUBERT_SR:
    #         audio = Resample(sr, HUBERT_SR).to(audio.device)(audio).to(DEVICE)
    # if audio.ndim == 1: audio = audio.unsqueeze(0)
    # with torch.no_grad():
    #     contents = hubert(audio).last_hidden_state
    # print(f"DEBUG {contents.shape=}")
    # torch.cuda.empty_cache()
    # wip_memory(hubert)
    #V2
    hubert = get_hubert_model(HUBERT_PRETRAIN, DEVICE, True)
    contents = calc_hubert_content(hubert, audio_f, DEVICE, None, None)
    torch.cuda.empty_cache()
    wip_memory(hubert)

    # get word timestamps
    audio  = WhisperX.load_audio(audio_f)
    whisperx_model = WhisperX()
    out = whisperx_model(audio)
    alignment = WhisperX.postprocess_out(out, by='words')
    timesteps = WhisperX.formed_timesteps(alignment)
    print(f"DEBUG {timesteps=}")

    if src_text is None:
        # Если самомго эталонного текста нет
        src_text = out['segments'][0]['text']
    
    if dataset is None:
        dataset = init_dataset()

    #use wer for ali
    all_preds = dict()

    wer_info = calculate_wer_with_alignment(src_text, target_text)
    ali       = wer_info["ali"][2]
    tgt_words = wer_info["recognized_words"]
    src_words = wer_info["reference_words"]
    for i in range(len(ali)):
        if not ali[i] == "S":
            continue
        else:
            target_text = tgt_words[i]

            # preprocessing text ?
            token_ids = dataset.tokenizer_encode(target_text)
            print(target_text, ":", token_ids)
            text_len  = len(target_text)

            if isinstance(model, str):
                if EXP_CODE == Exp.simple_transformer:
                    model_p = model
                    model, pad, eos, bos = init_simple_transformer(dataset=dataset)
                    model.load_state_dict(torch.load(model_p))
                    model.eval()
                    model = model.to(DEVICE)
                if EXP_CODE == Exp.yourtts_encoder:
                    # Инициализируем обученную модель из пути
                    model_p = model
                    encoder, decoder = init_textencoder(dataset), init_decoder(dataset)
                    model = Seq2Seq(encoder, decoder)
                    model.load_state_dict(torch.load(model_p))
                    model.eval()
                    model = model.to(DEVICE)
                    pad, eos, bos = dataset.gen_pad, dataset.gen_eos, dataset.gen_bos,
            
            # Декодируем
            if EXP_CODE == Exp.simple_transformer:
                
                print(f"Decoding: {target_text=}")
                num_tokens = len(token_ids)
                token_ids = torch.LongTensor([token_ids])
                print(f"{token_ids.shape=}, {num_tokens=}")
                src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
                tgt_words_preds = new_greedy_decoding(
                    model, src=token_ids, src_mask=src_mask, pad_symbol=pad,
                    max_len=len(token_ids)+20, start_symbol=bos, end_symbol=eos,
                    ).cpu().numpy()
                
                ref_text = src_words[i]
                print(f"Decoding: {ref_text=}")
                ref_text = dataset.tokenizer_encode(ref_text)
                num_tokens = len(ref_text)
                src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
                ref_text = torch.LongTensor([ref_text])
                src_words_preds = new_greedy_decoding(
                    model, src=ref_text, src_mask=src_mask, pad_symbol=pad,
                    max_len=len(ref_text)+20, start_symbol=bos, end_symbol=eos,
                    ).cpu().numpy()
                

                ref_text = timesteps[i][0]
                ref_text_content = [*timesteps[i][1]]
                print(f"Decoding: {ref_text=} (GT), range={timesteps[i][1]}")
                gt = dataset.get_contents_centers(contents, ref_text_content)
                gt = np.array(gt)
                
            if EXP_CODE == Exp.yourtts_encoder:
                preds = greedy_decoding(
                    model, torch.LongTensor([token_ids]).to(DEVICE), torch.LongTensor([text_len]).to(DEVICE), 
                    len(token_ids) + 20, bos, eos,
                    )
                preds = preds.cpu().numpy()#[1:-1]
            all_preds[i] = {}
            all_preds[i][f'tgt_preds ({tgt_words[i]})'] = tgt_words_preds
            all_preds[i][f'src_gt ({src_words[i]})']    = gt
            all_preds[i][f'src_preds ({src_words[i]})'] = src_words_preds


    CONFIG_PATH = "/mnt/storage/kocharyan/so-vits-svc-fork/ruslana/configs/44k/config.json"
    NUM = 1225
    MODEL_PATH = f"/mnt/storage/kocharyan/so-vits-svc-fork/ruslana/logs/44k/G_{NUM}.pth"
    OUT_DIR = "examples/res/"
    INPUT = [audio_f]
    infer_vc = SVCInfer(
        model_path=MODEL_PATH, conf_path=CONFIG_PATH, 
        auto_predict_f0=True, f0_method="dio",
        device=DEVICE, noise_scale=.4,
        )
    print("=====> SVCInfer works ...")
    infer_vc.inference(input_paths=INPUT, output_dir=OUT_DIR, speaker=None, )

    return all_preds


def new_greedy_decoding(model, src, src_mask, max_len, start_symbol, end_symbol, pad_symbol):
    src = src.to(DEVICE)
    src_mask = src_mask.to(DEVICE)

    mem = model.encode(src, src_mask)
    preds = torch.ones(1, 1).fill_(start_symbol)
    preds = preds.type(torch.long).to(DEVICE)

    for i in range(max_len-1):
        mem = mem.to(DEVICE)

        tgt_seq_len = preds.shape[1]
        tgt_mask = (torch.triu(
            torch.ones((tgt_seq_len, tgt_seq_len), device=DEVICE)
            ) == 1).float()
        tgt_mask = tgt_mask.masked_fill(tgt_mask == 0, float('-inf'))
        tgt_mask = tgt_mask.masked_fill(tgt_mask == 1, float(0.0))
        tgt_mask = tgt_mask.type(torch.bool)
        tgt_mask = tgt_mask.to(DEVICE)

        logits = model.decode(preds, mem, tgt_mask)
        # print(f"{logits.shape=}")
        logits = model.gen(logits[:, -1, :]) #TODO: CHECKME
        # print(f"{logits.shape=}")
        _, next_symb = torch.max(logits, dim=-1) #TODO: CHECKME
        next_symb = next_symb.item()

        next_symb = torch.ones(1, 1).type_as(src.data).fill_(next_symb)
        preds = torch.cat([preds, next_symb], dim=-1)
        
        if next_symb == end_symbol or next_symb == pad_symbol:
            break

    return preds


#CHECK ME ! Не работает - почему то выводит константное предсказание 
def greedy_decoding(model, src, src_len, max_len, start_symbol, end_symbol):
    
    print(f"DEBUG: {model.decoder.target_vocab_size=}")
    memory = model.only_encode(src, src_len)
    memory = memory.permute((0, 2, 1))
    print(f"DEBUG: {memory.shape=}")

    # TODO: Gen subsequent_mask
    preds = torch.ones(1, 1) #tensor([[1.]])
    preds = preds.fill_(start_symbol).type(torch.long).to(DEVICE)
    for i in range(max_len-1):
        print(f"DEBUG: {i=}")
        memory = memory.to(DEVICE)
        prob = model.only_decode(preds, memory) # tgt mask gen | prob = model.generator(out[:, -1])
        print(f"DEBUG: {prob.shape=}")
        # print(f"DEBUG: {prob[:, 0, :]=}")
        prob = model.decoder.generator(prob[:, -1, :]) #[:, -1]
        # out = out.permute((0, 2, 1)).squeeze() #?
        # print(f"DEBUG: {out.shape=}")
        # out = out[:,-1]
        # print(f"DEBUG: {out.shape=}")
        # prob = model.decoder.generator(out)
        # print(f"DEBUG: {out.shape=}")
        _, next_symb = torch.max(prob, dim=-1) #? 
        next_symb = next_symb.item()
        print(f"DEBUG: {next_symb=}")

        added = torch.ones(1, 1).type_as(src.data).fill_(next_symb)
        preds = torch.cat([preds, added], dim=-1)

        # print(preds)

        if next_symb == end_symbol:
            break
    
    return preds


if __name__ == "main":
    train()
    
