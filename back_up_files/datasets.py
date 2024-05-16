from typing import *
import json
from functools import partial

# from inferencers import *
from clustering import *
from utils import *
from typing import *
from pathlib import Path

import whisperx

import torch
from torch.utils.data import Dataset, DataLoader


class TextFromAudioDataset(Dataset):
    # TODO: make collate
    
    def __init__(self, data_path: Union[str, Path], config_path=None, processor=None, hubert_inf=None, clusters_path="clusters/clusters.pt"):
        
        self.dataset = sorted(get_dataset(data_path, "*.txt"))
        # self.audios = sorted(get_dataset(data_path, "*.wav"))
        
        self.__init_symbols(config_path)
        self._symbol_to_id = {s: i for i, s in enumerate(self.symbols)}

        self.processor = processor
        self.hubert_inf = hubert_inf

        self.char_clusters = CharClusters(clusters_path)

    def __getitem__(self, index):
        
        text = open(self.dataset[index], "r").read().strip()
        seq  = [self._symbol_to_id[symb] for symb in text]
        seq  = torch.LongTensor(seq)
        x = seq

        y = get_label_for_file(
            self.dataset[index].replace(".txt", ".wav"),
            self.char_clusters,
            self.processor,
            self.hubert_inf, True
        )

        # y = [self._symbol_to_id[symb] for symb in y[1]]
        
        return text, x, y

    def __len__(self):
        return len(self.dataset)

    def __init_symbols(self, config_path):
        if config_path:
            with open(config_path, 'r') as f: #e.g. "YourTTS-RU-RUSLAN-April-30-2023_03+48PM-0000000/config.json"
                data = json.load(f)
            data = data['characters']
            self.symbols = [data["pad"]] + list(data["punctuations"]) + list(data["characters"]) + list(data["phonemes"]) + ["<BLNK>"]
        else:
            _pad         = '_'
            _punctuation = ' ' #';:,.!?¡¿—…"«»“” '
            _letters     = 'абвгдежзийклмнопрстуфхцчшщъыьэюяё' #"ЁАБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдежзийклмнопрстуфхцчшщъыьэюяё"
            #_letters     = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
            # _letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"
            self.symbols = [_pad] + list(_punctuation) + list(_letters) # + list(_letters_ipa)

class AudioDataset(Dataset):
    
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        waveform = whisperx.load_audio(self.dataset[idx])
        return waveform


class AudioLoad(Dataset):
    
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        waveform = whisperx.load_audio(self.dataset[idx])
        return waveform

def get_dataset(dataset_dir: Union[Path, str] = "RuDevices", pattern: str = "*.wav"):
    dataset_dir   = Path(dataset_dir)
    audio_dataset = list(dataset_dir.rglob(pattern))
    audio_dataset = list(map(lambda x: x.as_posix(), audio_dataset))
    return audio_dataset


def hubert_processor_collate(batch, processor):
    # wave_forms, _ = zip(*batch) if batch consist of multiple items, e.g. (batch, sr)
    input_values = processor(batch, sampling_rate=16_000, return_tensors="pt", padding=True).input_values
    return input_values

def hubert_inference(file, hubert_inf, processor, device):
    collate = partial(hubert_processor_collate, processor=processor)
    hubert_audio_loader = DataLoader(AudioLoad([file]), batch_size=1, collate_fn=collate)
    hubert_inf.eval()
    hubert_inf = hubert_inf.to(device)
    return hubert_inf(next(iter(hubert_audio_loader)).to(device))


def calculates_labels(
    file: str, ali: Dict[str, List[List]], clusters: CharClusters, 
    hubert_model: Any, hubert_preproc: Any, replace_spase:bool=True,
    ):
    
    def get_char(i):
        for ids in idx2char:
            for _id in ids:
                if _id == tuple(i): return idx2char[ids], _id
        return 
    
    idx2char = {tuple(tuple(x) for x in ids): char for char, ids in ali.items()}
    idss     = [get_char(i) for i in sorted(flatten(ali.values()))]
    text     = "".join([i for i, j in idss])
    if replace_spase:
        text = text.replace("bspace", " ")

    res1 = []
    res2 = []
    res = []
    hubert_contents = hubert_inference(file, hubert_model, hubert_preproc, "cuda") #TODO make unifiy for preprocessing
    src_len = len(hubert_contents[0])
    for char, ids in idss:
        inf_res = hubert_contents[:, ids]
        inf_res = inf_res.cpu().numpy().squeeze(0)
        pred = clusters.predict_cluster_center(
                char, inf_res.astype(np.float32)
            )
        res.append("".join([*map(lambda x: char+f'({x})', pred)]))
        res1.append(pred)
        res2.append(char)
    return text, "-".join(res), src_len


def get_label_for_file(file: str, char_clusters: CharClusters, processor=None, hubert_inf=None, replace_spase=True):
    # if not processor or not hubert_inf:
    #     processor, hubert_model, hubert_inf = get_hubert()
    clac_label_res = calculates_labels(
        file, torch.load("ali/align.pt")['ali'][file], char_clusters, hubert_inf, processor, replace_spase
    )
    return clac_label_res


def whisper_inference_for_file(file, whisper):
    whisper_audio_loader = DataLoader(AudioDataset([file]), batch_size=1, collate_fn=lambda batch: batch)
    whisper.eval()
    return whisper(next(iter(whisper_audio_loader))[0])