from utils import *
from datasets import get_dataset
from clustering import CharClusters
from inferencers import *
from ali import *
from whisper_proc import _batch_whisper_infer

from functools import partial
from typing import Union
from pathlib import Path
from joblib import Parallel, delayed, cpu_count
import tqdm
from typing import *

import numpy as np
import librosa 

import torch
from torch.utils.data import DataLoader, Dataset
from torchaudio.transforms import Resample
from transformers import AutoProcessor, HubertModel


HUBERT_SR = 16_000


def get_hubert(model_id: str = "facebook/hubert-large-ls960-ft", device: str = "cuda"):
    processor    = AutoProcessor.from_pretrained(model_id)
    hubert_model = HubertModel.from_pretrained(model_id).to(device)
    hubert_inf   = HuBERTInference(hubert_model).to(device)
    return processor, hubert_model, hubert_inf

def calc_hubert_content(model: Union[HuBERTInference, HubertModel], 
                        audio: Union[torch.Tensor, str, Path], 
                        device: str, sr: int, processor: Any) -> torch.Tensor:
    if isinstance(audio, str) or isinstance(audio, Path):
        contents = model(audio, model, processor, device)
    else:
        # audio, sr = librosa.load(filepath, sr=16_000, mono=True)
        # audio = torch.from_numpy(audio).float().to("cuda")
        # audio = torch.as_tensor(audio)
        if sr != HUBERT_SR:
            audio = Resample(sr, HUBERT_SR).to(audio.device)(audio).to(device)
        if audio.ndim == 1: audio = audio.unsqueeze(0)
        with torch.no_grad():
            contents = model(audio)
            contents = contents.last_hidden_state #.transpose(1, 2) #["last_hidden_state"].transpose(1, 2) #
    return contents


def _one_item_hubert_infer(file, hubert_model, hps): #TODO
    
    audio, sr = librosa.load(file, sr=hps["data_sr"], mono=True)
    audio     = torch.from_numpy(audio).float().to(hps["device"])
    # https://github.com/voicepaw/so-vits-svc-fork/blob/main/src/so_vits_svc_fork/preprocessing/preprocess_hubert_f0.py#L65
    content  = calc_hubert_content(hubert_model, audio, hps["device"], sr).to("cpu") #repeat_expand_2d, fixed len
    torch.cuda.empty_cache()
    hps["out_dir"] = Path(hps["out_dir"])

    # print(content.shape) #DEBUG
    
    for char, content_ids in hps["ali"][file].items():
        
        char_dir = hps["out_dir"] / Path(char)
        check_and_create_dir(char_dir)
        
        # content_path = char_dir / (Path(file).name + ".content.pt")
        content_path = char_dir / (".".join(file.split("/")) + ".content.pt")
        with content_path.open("wb") as f:
            hiddens = []
            for content_id in content_ids:
                hiddens.append(content[:, content_id]) # [bs=1, idx=content_id, hidden=1024]
            hiddens = torch.cat(hiddens, 1)
            
            # print(hiddens.shape) #DEBUG
            
            torch.save({"content": hiddens}, f)


def calc_hubert_ali_for_one_char(alignment):
    char = alignment['char'].lower()
    if not (is_cyrillic(char) or char == " "): #TODO: CHECK
        raise Exception()
    start = int(alignment['start'] * 1000 // 20)
    end   = int(alignment['end']   * 1000 // 20) + 1
    return char if char != " " else "bspace", [*range(start, end)]


def _batch_hubert_infer(files, pbar, hps): #TODO
    proc, hubert_model, _ = get_hubert(model_id="facebook/hubert-large-ls960-ft", device=hps["device"])
    for file in tqdm(files, position=pbar):
        _one_item_hubert_infer(file, hubert_model, hps)
    wip_memory(hubert_model)


def create_hubert_content(data_dir: Union[str, Path] = "RuDevices", sr: int = 16_000,
                          out_dir: str = "./ali", device: str = "cuda") -> dict:
    # Для каждой аудиозаписи сохраняем словарь из букв и контент-вектров

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True) # TODO: CHECK

    dataset = get_dataset(data_dir)
    # whisper = WhisperXInference("float16", device, "ru")
    # dataset = AudioDataset(dataset)
    # loader  = DataLoader(dataset, batch_size=1, collate_fn=lambda batch: batch)

    audio_files = dataset #TOOD: CHECK | CHECK space symb !!!!!
    n_jobs      = 3 #cpu_count() - 4
    file_chunks = np.array_split(audio_files, n_jobs)
    # files_to_whispout = Parallel(n_jobs=-1)(delayed(_whisper_inf)( # make batch inf ?!
    #     data[0], dataset.dataset[i]
    # ) for i, data in enumerate(loader))
    # files_to_whispout = [_whisper_inf(data[0], dataset.dataset[i]) for i, data in enumerate(loader)]
    
    files_to_whispout = Parallel(n_jobs=n_jobs)(delayed(_batch_whisper_infer)(
        chunk, pbar
    ) for (pbar, chunk) in enumerate(file_chunks))
    # files_to_whispout = [_batch_whisper_infer(chunk, pbar) for (pbar, chunk) in enumerate(file_chunks)]

    files_to_whispout = flatten(files_to_whispout)
    resault = dict(files_to_whispout)

    all_alignments = get_all_alignments(resault, calc_hubert_ali_for_one_char)
    with (out_dir / "align.pt").open("wb") as f:
        torch.save({"ali": dict(all_alignments)}, f)
    #TODO: cuda memory clear
    # wip_memory(whisper)
    
    audio_files = list(all_alignments.keys()) #TOOD: CHECK | CHECK space symb !!!!!
    n_jobs      = 3 #cpu_count() - 4
    file_chunks = np.array_split(audio_files, n_jobs)

    # print(file_chunks) # DEGUG

    hps            = {}
    hps["data_sr"] = sr
    hps["out_dir"] = out_dir # "./preprocessed_dataset"
    hps["device"]  = device
    hps["ali"]     = all_alignments
    
    Parallel(n_jobs=n_jobs)(delayed(_batch_hubert_infer)( #TODO
        chunk, pbar, hps
    ) for (pbar, chunk) in enumerate(file_chunks)) #TODO: CHECK (...) for other
    # [_batch_hubert_infer(chunk, pbar, hps) for (pbar, chunk) in enumerate(file_chunks)]
    # add torch.cuda.empty_cache()
    # use hubert for get content
    # Создать папки с именами букв
    # в каждом сохранить для каждого аудио вектора

    return