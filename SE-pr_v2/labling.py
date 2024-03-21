from typing import * 
from pathlib import Path
import json

import librosa
import numpy as np
import torch
from torch.utils.data import Dataset
from torchaudio.transforms import Resample
from transformers import HubertModel

from cutils import get_dataset_from_dir
from clustering import PseudoPhonemes

symbols_dict = {
    "_pad" : '_',
    "_punctuation" : ' !+,-.:;?«»—',
    "_letters" : 'абвгдежзийклмнопрстуфхцчшщъыьэюяё',
    "_letters_ipa" : "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ",
}


class Text2PseudoPhonemes(Dataset):
    '''
        Тут есть пару идей:
            1) Можно токенизировать в формате YourTTS и использовать TextEncoder для генерации последовательности псевдо-фонем
               Этот подход должен лучше работать, так как уже несет в себе информацию о фонемах
            2) Можно просто токенизировать с помощью T5 и генерировать псевдо-фонемы. 
    '''

    def __init__(self, texts_path: str, contents_path: str, clusters_path: str, 
                 pretrain_path: str, lm_tokenizer: Optional[str]=None,
                 config: Optional[Union[str, dict]]=symbols_dict, dsrate: int=16_000) -> None:
        super().__init__()
        
        self.texts    = sorted(get_dataset_from_dir(texts_path, "*.txt"))
        self.contents = [Path(file).relative_to(texts_path) for file in self.texts]
        self.contents = [file.with_suffix(".wav.content.pt").as_posix() for file in self.contents]
        self.contents = [".".join(file.split("/")) for file in self.contents]
        self.contents = [(Path(contents_path) / file).as_posix() for file in self.contents]
            
        
        self.pseudo_phonem_clusters = PseudoPhonemes(clusters_path)
        self.pseudo_phonem_clusters.build_clusters()

        self.hubert = HubertModel.from_pretrained(pretrain_path)
        self.srate = dsrate
        
        self.config_path = config
        self.lm_tokenizer = lm_tokenizer
        if config:
            self.__init_symbols(config)
            self._symbol_to_id = {s: i for i, s in enumerate(self.symbols)}
        elif lm_tokenizer:
            raise NotImplementedError()
        else:
            Exception()
        return 
    
    def __getitem__(self, index):
        
        text = open(self.texts[index], "r").read().strip()
        if self.config_path:
            seq  = [self._symbol_to_id[symb] for symb in text]
            x    = torch.LongTensor(seq)
        else:
            raise NotImplementedError()
        
        # ===> bottleneck:
        # HUBERT_SR = 16_000
        # wav       = Path(self.dataset[index]).with_suffix(".wav").as_posix()
        # audio, sr = librosa.load(wav, sr=self.srate, mono=True)
        # audio     = torch.from_numpy(audio).float()
        # audio     = Resample(sr, HUBERT_SR).to(audio.device)(audio)
        # if audio.ndim == 1: audio = audio.unsqueeze(0)
        # with torch.no_grad():
        #     contents = self.hubert(audio)
        #     contents = contents.last_hidden_state

        y = []
        contents = torch.load(self.contents[index], weights_only=True)
        contents = contents["content"].squeeze(0).numpy()
        contents = contents.astype(np.float32)
        for pseudo_ph in contents:
            pseudo_ph = pseudo_ph.reshape(1, -1)
            pred_ph = self.pseudo_phonem_clusters.predict_cluster_center(pseudo_ph)
            y.append(pred_ph[0])

        return {"tokens": x, "pseudo_ph": torch.LongTensor(y), "text": text}
    
    def __len__(self):
        return len(self.texts)
    
    def __init_symbols(self, config):
        if isinstance(config, str):
            with open(config, 'r') as f: #e.g. "YourTTS-RU-RUSLAN-April-30-2023_03+48PM-0000000/config.json"
                data = json.load(f)
            data = data['characters']
            self.symbols = [data["pad"]] + list(data["punctuations"]) + list(data["characters"]) + list(data["phonemes"]) + ["<BLNK>"]
        else:
            self.symbols = [config["_pad"]] + list(config["_punctuation"]) + list(config["_letters"]) + list(config["_letters_ipa"])
        #TODO: examples
        # _pad         = '_'
        # _punctuation = ' ' #';:,.!?¡¿—…"«»“” '
        # _letters     = 'абвгдежзийклмнопрстуфхцчшщъыьэюяё' #"ЁАБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдежзийклмнопрстуфхцчшщъыьэюяё"
        #_letters     = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
        # _letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"
        # self.symbols = [_pad] + list(_punctuation) + list(_letters) # + list(_letters_ipa)
        return 
    