from pprint import pprint
from pathlib import Path
import random

import torch
from tqdm import tqdm
from cm_time import timer
from joblib import Parallel, delayed, cpu_count
import numpy as np

from cutils import get_dataset_from_dir, wip_memory, del_folder
from calc_content import _one_item_hubert_infer, get_hubert_model
from models import WhisperX


def timesteps2content(timesteps, content, f0, save_to):
    # assert content.shape[-1] == f0.shape[-1] == max([max(idxs) for _, idxs in timesteps.values()])
    for i in timesteps:
        idxs = [*timesteps[i][1]]
        # Дичь (как правильно отмапить?):
        try:
            word_repr = content[..., idxs] # content: [h, t]
            # print(f"{timesteps[i][0]} {content[..., idxs].shape=} {timesteps[i][1]}")
        except IndexError:
            continue
        word = timesteps[i][0]
        word_dir = (Path(save_to) / word)
        word_dir.mkdir(parents=True, exist_ok=True)
        file = word_dir / f"{random.getrandbits(128)}"
        with file.open("wb") as f:
            torch.save({"word_repr": word_repr}, f)


def _batch_whisper_infer(files, pbar, hps):
    whisperx_model = WhisperX()
    hubert_model = (
        get_hubert_model(conf=hps["hmodel_id"], device=hps["device"], 
                         final_proj=True) if not hps["content_p"] else None
    )
    assert hubert_model is None # local checking 
    
    for file in tqdm(files, position=pbar):
        audio  = WhisperX.load_audio(file)
        out = whisperx_model(audio)
        
        try:
            alignment = WhisperX.postprocess_out(out, by='words')
            timesteps = WhisperX.formed_timesteps(alignment)
        except KeyError:
            continue
        
        if not hps["content_p"]:
            contents = _one_item_hubert_infer(file, hubert_model, hps, True)
        else:
            f = Path(hps["content_p"]) / Path(Path(file).name + ".content.pt")
            contents = torch.load(f) #.permute(1, 0)
            assert len(contents["content"].shape) == 2
            assert contents["content"].shape[0] == 256 # [h, t]
        
        timesteps2content(timesteps, contents["content"], contents["f0"], hps["out_dir"])

    wip_memory(whisperx_model)


def create_words_dataset(
    data_dir: str, srate: int, out_dir: str,
    njobs: int, pretrain: str, 
    ):
    
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = get_dataset_from_dir(data_dir, pattern="*.wav")
    file_chunks = np.array_split(dataset, njobs)

    hps = {}
    hps["model_path"] = pretrain  #TODO: Захардкожено в модель
    hps["data_sr"] = srate #TODO: определяется автоматом?
    hps["hmodel_id"] = "../../hfmodels/content-vec-best"
    hps["out_dir"] = out_dir
    hps["device"] = "cuda"
    hps["f0_method"] = "dio"
    hps["hop_len"]   = 512
    hps["content_p"] = "../../NIR/ruslan_contents/" # | None

    Parallel(n_jobs=njobs)(delayed(_batch_whisper_infer)(
        chunk, pbar, hps
    ) for (pbar, chunk) in enumerate(file_chunks))

    return


class ConcatVITSInfer:
    pass


if __name__ == "__main__":
    del_folder("../../NIR/ruslan_word_database/")
    create_words_dataset(
        "../../sambashare/ruslan_ds/RUSLAN/", 22_000, 
        "../../NIR/ruslan_word_database/", 3, "../../hfmodels/content-vec-best",
    )

    # Изначально 638/7400 [14:53<2:39:50 (3 потока)