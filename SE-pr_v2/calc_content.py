from pathlib import Path
from typing import * 
from tqdm import tqdm

from joblib import Parallel, delayed, cpu_count
import numpy as np
import librosa 
import torch
from torchaudio.transforms import Resample
from transformers import HubertModel

from cutils import get_dataset_from_dir, wip_memory


def calc_hubert_content(model: HubertModel, 
                        audio: Union[torch.Tensor, str, Path], 
                        device: str, sr: int, processor: Optional[Any]) -> torch.Tensor:
    HUBERT_SR = 16_000
    if isinstance(audio, str) or isinstance(audio, Path):
        contents = model(audio, model, processor, device)
    else:
        if sr != HUBERT_SR:
            audio = Resample(sr, HUBERT_SR).to(audio.device)(audio).to(device)
        if audio.ndim == 1: audio = audio.unsqueeze(0)
        with torch.no_grad():
            contents = model(audio)
            contents = contents.last_hidden_state #TODO checking:.transpose(1, 2) #["last_hidden_state"].transpose(1, 2) #
    return contents


def _one_item_hubert_infer(file, hubert_model, hps):
    '''Вычисление контент-векторов для файла и сохранение'''
    
    audio, sr = librosa.load(file, sr=hps["data_sr"], mono=True)
    audio     = torch.from_numpy(audio).float().to(hps["device"])
    content   = calc_hubert_content(hubert_model, audio, hps["device"], sr, None)
    content   = content.to("cpu") #TODO: repeat_expand_2d, fixed len
    # print(content.shape) 
    # DEBUG: 
    # torch.Size([1, 59, 768])
    # torch.Size([1, 290, 768])
    # torch.Size([1, 98, 768]) ...
    
    torch.cuda.empty_cache()

    file = Path(file).relative_to(hps['rel_to']).as_posix()
    content_path = Path(hps["out_dir"]) / (".".join(file.split("/")) + ".content.pt")
    with content_path.open("wb") as f:
        torch.save({"content": content}, f)

    return


def _batch_hubert_infer(files, pbar, hps):
    '''Вичисление контент векторов для N-файлов в одном запущенном потоке'''
    hubert_model = HubertModel.from_pretrained(hps["hmodel_id"]).to(hps["device"])
    for file in tqdm(files, position=pbar):
        _one_item_hubert_infer(file, hubert_model, hps)
    wip_memory(hubert_model)


def create_hubert_content(data_dir: Union[str, Path] = "RuDevices", srate: int = 16_000, pretrain_path: str = "./",
                          out_dir: str = "./extracted_contents", device: str = "cuda", njobs: Optional[int] = 1) -> dict:
    '''Многопоточная обработка датасета - вычисляем контент-вектора и сохраняем'''
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    relative_to = Path(data_dir)
    dataset     = get_dataset_from_dir(data_dir, pattern="*.wav")
    n_jobs      = njobs if njobs else (cpu_count() - 1)
    file_chunks = np.array_split(dataset, n_jobs)
    print(f"{n_jobs=}")

    hps              = {}
    hps["hmodel_id"] = pretrain_path #"facebook/hubert-large-ls960-ft" #TODO: check so-vits
    hps["data_sr"]   = srate
    hps["out_dir"]   = out_dir
    hps["device"]    = device
    hps["rel_to"]    = relative_to

    print(hps)
    
    Parallel(n_jobs=n_jobs)(delayed(_batch_hubert_infer)( #TODO
        chunk, pbar, hps
    ) for (pbar, chunk) in enumerate(file_chunks))

    return