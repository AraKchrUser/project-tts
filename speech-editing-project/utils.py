from pathlib import Path
from typing import Union, Any
import unicodedata
import random
import shutil

import wget
from directory_tree import display_tree

import gc
import numpy as np
import torch


def wip_memory(model):
    del model
    _wip_memory()

def _wip_memory():
    gc.collect()
    torch.cuda.empty_cache()


def check_and_create_dir(dir: Path):
    dir.mkdir(parents=True, exist_ok=True)


def is_cyrillic(char):
    return 'CYRILLIC' in unicodedata.name(char) #TODO: Add space symb


def del_folder(path):
    path = Path(path)
    if not path.exists():
        return
    for sub in path.iterdir():
        if sub.is_dir(): del_folder(sub)
        else : sub.unlink()
    path.rmdir()


def flatten(xss):
    return [x for xs in xss for x in xs]


def load_checkpoint(model: Any, ckpt_path: Union[str, Path], mname: str, download: bool=False):
    
    ckpt_path = Path(ckpt_path)
    
    if download:
        urls = [
            "https://huggingface.co/spaces/sayashi/vits-uma-genshin-honkai/resolve/main/model/G_0.pth",
            "https://huggingface.co/spaces/sayashi/vits-uma-genshin-honkai/resolve/main/model/D_0.pth",
        ]
        if not ckpt_path.parent.exists():
            ckpt_path.parent.mkdir(exist_ok=True, parents=True)
        for url in urls:
            wget.download(url, out=ckpt_path.parent)
    
    with ckpt_path.open("rb") as f:
        ckpt_dict = torch.load(f, map_location="cpu", weights_only=True)
    
    ckpt_dict = dict([(key[len(mname)+1:], ckpt_dict['model'][key]) 
                      for key in ckpt_dict['model'].keys() if mname in key])
    model_dict = model.state_dict()
    
    new_state_dict = {}
    for k, v in model_dict.items():
        # https://github.com/jaywalnut310/vits/blob/main/utils.py#L34
        new_state_dict[k] = ckpt_dict[k]
    model.load_state_dict(new_state_dict)

    return model


def create_chunk_dataset(
        src_dataset: Union[Path, str]="RuDevices", k: int=2, 
        out_dataset: Union[Path, str]="rudevices_chunk", display: bool=False):
    
    chunk_dirs = {}
    res_files = []
    src_path = Path(src_dataset)
    choice_dir = random.choices([*src_path.iterdir()], k=k)
    out_path = Path(out_dataset)
    if out_path.exists():
        del_folder(out_path)
    out_path.mkdir(parents=True, exist_ok=True)

    for _dir in choice_dir:
        chunk_subdirs = random.sample([*_dir.iterdir()], k=k)
        for _subdir in chunk_subdirs:
            chunk_dirs[_subdir] = random.choices([*_subdir.rglob("*.wav")], k=2*k)
    
    files = np.concatenate(list(chunk_dirs.values()))
    for src_file in files:
        dist_file  = src_file.as_posix().replace(src_path.as_posix(), out_path.as_posix())
        res_files.append(dist_file)
        Path(dist_file).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src_file, Path(dist_file))
    
    if display:
        display_tree(out_path.as_posix())

    return