from typing import * 
from pathlib import Path
import json

import librosa
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torchaudio.transforms import Resample
from transformers import HubertModel
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.models.vits import Vits, VitsArgs, VitsAudioConfig
from TTS.tts.utils.text.tokenizer import TTSTokenizer

from cutils import get_dataset_from_dir
from clustering import PseudoPhonemes


symbols_dict = {
    "_pad" : '_',
    "_punctuation" : ' !+,-.:;?«»—',
    "_letters" : 'абвгдежзийклмнопрстуфхцчшщъыьэюяё',
    "_letters_ipa" : "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ",
}


class CoquiTTSTokenizer:
    
    def __init__(self, config_path="ckpts/yourrtts_config.json") -> None:
        config = VitsConfig()
        config.load_json(file_name=config_path)
        config["add_blank"] = False
        tokenizer = TTSTokenizer.init_from_config(config)
        self.tokenizer = tokenizer

    @property
    def get_tokenizer(self):  
        return self.tokenizer[0]


class GraphemeTTSTokenizer:
    '''
    Тут еще много доделывать, попробуем воспользоваться Tokenizer из coqui/TTS (CoquiTTSTokenizer, но там слишком много зависимостей)
    '''
    
    def __init__(self, characters: str=None, punctuations: str=None, pad: str=None, 
                 eos: str=None, bos: str=None, blank: str=None) -> None:
        self.characters   = characters
        self.punctuations = punctuations
        self.pad          = pad
        self.eos          = eos
        self.bos          = bos
        self.blank        = blank
        self._build_vocab()
    
    def _build_vocab(self):
        
        vocab = self.characters #set(self.characters)
        vocab = sorted(list(vocab))
        
        vocab = [self.blank] + vocab if self.blank is not None and len(self.blank) > 0 else vocab
        vocab = [self.bos]   + vocab if self.bos   is not None and len(self.bos)   > 0 else vocab
        vocab = [self.eos]   + vocab if self.eos   is not None and len(self.eos)   > 0 else vocab
        vocab = [self.pad]   + vocab if self.pad   is not None and len(self.pad)   > 0 else vocab
        
        self.vocab = vocab + list(self.punctuations)
        self._char2id = {char: idx for idx, char in enumerate(self.vocab)}
        self._id2char = {idx: char for idx, char in enumerate(self.vocab)}

        # if not (len(self.vocab) == len(self._char2id) == len(self._id2char)):
        #     duplicates = {x for x in self.vocab if self.vocab.count(x) > 1}
        #     print("Exception! ", duplicates)
        
        return 

    @property
    def pad_id(self):
        return self.char2id(self.pad)

    def char2id(self, char: str) -> int:
        try:
            return self._char2id[char]
        except KeyError as e:
            print("Check your vocab !")
    
    def id2char(self, idx: int) -> str:
        return self._id2char[idx]
    
    def size(self):
        return len(self.vocab)

    def encode(self, text: str):
        
        #TODO: add text cleaned

        token_ids = []
        for char in text:
            try:
                idx = self.char2id(char)
                token_ids.append(idx)
            except KeyError:
                print("Check your vocab!")
        
        #TODO: add blank?
        #TODO: use eos/bos ?

        return token_ids

    def decode(self, token_ids: list):
        text = ""
        for token_id in token_ids:
            text += self.id2char(token_id)
        return text


class Text2PseudoPhonemes(Dataset):
    '''
        Тут есть пару идей:
            1) Можно токенизировать в формате YourTTS и использовать TextEncoder для генерации последовательности псевдо-фонем
            2) Можно просто токенизировать с помощью T5 и генерировать псевдо-фонемы (более унифицирваонный)
    '''

    def __init__(self, texts_path: str, contents_path: str, clusters_path: str, 
                 pretrain_path: Optional[str]=None, lm_tokenizer: Optional[str]=None,
                 config: Optional[Union[str, dict]]=None, dsrate: int=16_000, coquitokenizer: bool=True) -> None:
        super().__init__()
        
        self.texts    = sorted(get_dataset_from_dir(texts_path, "*.txt"))
        self.contents = [Path(file).relative_to(texts_path) for file in self.texts]
        self.contents = [file.with_suffix(".wav.content.pt").as_posix() for file in self.contents]
        self.contents = [".".join(file.split("/")) for file in self.contents]
        self.contents = [(Path(contents_path) / file).as_posix() for file in self.contents]
            
        
        self.pseudo_phonem_clusters = PseudoPhonemes(clusters_path)
        self.pseudo_phonem_clusters.build_clusters()

        if pretrain_path:
            self.hubert = HubertModel.from_pretrained(pretrain_path)
        self.srate = dsrate
        
        self.config_path = config
        self.lm_tokenizer = lm_tokenizer
        self.use_coquitokenizer = coquitokenizer
        if config:
            self.__init_ttstokenizer(config)
            # self._symbol_to_id = {s: i for i, s in enumerate(self.symbols)}
        elif lm_tokenizer:
            raise NotImplementedError()
        else:
            Exception()

        gen_bos = self.pseudo_phonem_clusters.kmeans.cluster_centers_.shape[0]
        self.gen_bos = gen_bos
        self.gen_eos = gen_bos + 1
        self.gen_pad = gen_bos + 2

        return 
    
    def __getitem__(self, index):
        
        text = open(self.texts[index], "r").read().strip()
        if self.config_path:
            # seq  = [self._symbol_to_id[symb] for symb in text]
            seq = self.encode(text)
            x    = torch.LongTensor(seq)
        else:
            raise NotImplementedError()
        
        # ===> BOTTLENECK:
        # HUBERT_SR = 16_000
        # wav       = Path(self.dataset[index]).with_suffix(".wav").as_posix()
        # audio, sr = librosa.load(wav, sr=self.srate, mono=True)
        # audio     = torch.from_numpy(audio).float()
        # audio     = Resample(sr, HUBERT_SR).to(audio.device)(audio)
        # if audio.ndim == 1: audio = audio.unsqueeze(0)
        # with torch.no_grad():
        #     contents = self.hubert(audio)
        #     contents = contents.last_hidden_state

        # Была идея испольлзвать в качестве эмбеддингов для декодинга pseudo_ph_embeds (эмбеддинги центры кластеров)
        # Но возникли трудности 
        y = [self.gen_bos]
        # pseudo_ph_embeds = []
        contents = torch.load(self.contents[index], weights_only=True)
        contents = contents["content"].squeeze(0).numpy()
        contents = contents.astype(np.float32)
        for pseudo_ph in contents:
            pseudo_ph = pseudo_ph.reshape(1, -1)
            pred_ph = self.pseudo_phonem_clusters.predict_cluster_center(pseudo_ph)
            y.append(pred_ph[0])
            # pseudo_ph_embed = self.pseudo_phonem_clusters.get_cluster_center(pseudo_ph)
            # pseudo_ph_embeds.append(torch.from_numpy(pseudo_ph_embed[0]))
        y.append(self.gen_eos)
        
        # pseudo_ph_embed = torch.stack(pseudo_ph_embeds)
        # print(pseudo_ph_embed.shape)

        return {
            "tokens": x, 
            "pseudo_ph": torch.LongTensor(y), 
            "text": text, 
            "decoded_tokens": self.decode(seq),
            # "pseudo_ph_embeds": pseudo_ph_embed,
            }
    
    def __len__(self):
        return len(self.texts)

    def collate_fn(self, batch):
        # https://github.com/coqui-ai/TTS/blob/dev/recipes/vctk/yourtts/train_yourtts.py
        # https://github.com/coqui-ai/TTS/blob/dev/TTS/tts/models/vits.py#L302
        # https://github.com/coqui-ai/TTS/blob/dev/TTS/tts/utils/text/tokenizer.py#L10
        # https://pytorch.org/tutorials/beginner/translation_transformer.html
        
        B = len(batch)
        batch = {k: [dic[k] for dic in batch] for k in batch[0]}

        #TODO: if use eos/bos ?
        max_token_len = max([len(x) for x in batch['tokens']])
        tokens_padded = torch.LongTensor(B, max_token_len)
        tokens_padded = tokens_padded.zero_() + self.pad_id
        for i in range(len(batch["tokens"])):
            token_ids = batch["tokens"][i]
            token_len = len(token_ids)
            tokens_padded[i,:token_len] = torch.LongTensor(token_ids)
        
        #TODO: if use eos/bos ?
        max_label_len = max([len(x) for x in batch['pseudo_ph']])
        lables_padded = torch.LongTensor(B, max_label_len)
        lables_padded = lables_padded.zero_() + self.gen_pad
        for i in range(len(batch["pseudo_ph"])):
            label     = batch["pseudo_ph"][i]
            label_len = len(label)
            lables_padded[i,:label_len] = torch.LongTensor(label)
        
        text_lens = torch.LongTensor(B)
        for i in range(len(batch["text"])):
            text         = batch["text"][i]
            text_len     = len(text)
            text_lens[i] = torch.LongTensor([text_len])
        
        # print(batch['pseudo_ph_embeds'])
        # ph_embeds = []
        # for i in range(len(batch['pseudo_ph_embeds'])):
        #     ph_embed = batch["pseudo_ph_embeds"][i]
        #     ph_embeds.append(ph_embed)
        
        # ph_embeds = pad_sequence(batch['pseudo_ph_embeds'], batch_first=True, padding_value=0) #tODO
        
        return {
            "tokens_padded": tokens_padded, 
            "lables": lables_padded, 
            "text_lens": text_lens,
            # "ph_embeds": ph_embeds,
            }

    def encode(self, text, use_phonemes=False):
        if not use_phonemes:
            token_ids = self.tokenizer.text_to_ids(text) #encode(text)
            return token_ids # np.array(token_ids, dtype=np.int32)
        else:
            raise NotImplementedError()
    
    def decode(self, token_ids):
        return self.tokenizer.ids_to_text(token_ids) #decode(token_ids)
    
    def __init_ttstokenizer(self, config):
        '''
        E.g.:
            _pad         = '_'
            _punctuation = ' ' #';:,.!?¡¿—…"«»“” '
            _letters     = 'абвгдежзийклмнопрстуфхцчшщъыьэюяё' #"ЁАБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдежзийклмнопрстуфхцчшщъыьэюяё"
            _letters     = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
            _letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"
            self.symbols = [_pad] + list(_punctuation) + list(_letters) # + list(_letters_ipa)
        '''

        if not self.use_coquitokenizer:

            if isinstance(config, str):
                with open(config, 'r') as f: #e.g. "YourTTS-RU-RUSLAN-April-30-2023_03+48PM-0000000/config.json"
                    data = json.load(f)
                data = data['characters']
                # self.symbols = [data["pad"]] + list(data["punctuations"]) + list(data["characters"]) + list(data["phonemes"]) + ["<BLNK>"]
                config = data

            # self.symbols = [config["_pad"]] + list(config["_punctuation"]) + list(config["_letters"]) + list(config["_letters_ipa"])
            self.tokenizer = GraphemeTTSTokenizer(
                characters=config["characters"], punctuations=config["punctuations"], 
                pad=config["pad"], eos=config["eos"], bos=config["bos"], blank=config["blank"]
                )
        
        else:
            self.tokenizer = CoquiTTSTokenizer(config).get_tokenizer

        self.pad_id = self.tokenizer.pad_id
        return 
    