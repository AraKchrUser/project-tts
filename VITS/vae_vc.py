import logging

logging.getLogger('numba').setLevel(logging.WARNING)

import os
import sys
import json
import math
import copy
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import soundfile

sys.path.append('../vits_cloning')
import commons
import utils
from data_utils import TextAudioLoader, TextAudioCollate, TextAudioSpeakerLoader, TextAudioSpeakerCollate
from models import SynthesizerTrn
from text.symbols import symbols
from text import text_to_sequence

from scipy.io.wavfile import write
from mel_processing import spectrogram_torch
from utils import load_wav_to_torch


ROOT_PATH = '../vits_cloning'
HPS_PATH = os.path.join(ROOT_PATH, "./configs/vctk_base.json")
MODEL_PATH = os.path.join(ROOT_PATH, "pretrained_models/pretrained_vctk.pth")
SRC_WAV_PATH = os.path.join(ROOT_PATH, './path/to/Test/test3_F.wav')
OUT_DIR = './out'


class VITS_Voice_Conversion:
    @staticmethod
    def get_text(text, hps):
        text_norm = text_to_sequence(text, hps.data.text_cleaners)
        if hps.data.add_blank:
            text_norm = commons.intersperse(text_norm, 0)
        text_norm = torch.LongTensor(text_norm)
        return text_norm

    def __init__(self, hps_path, pretrained_path):
        self.y = None
        self.hps = utils.get_hparams_from_file(hps_path)
        self.net = SynthesizerTrn(
            len(symbols),
            self.hps.data.filter_length // 2 + 1,
            self.hps.train.segment_size // self.hps.data.hop_length,
            n_speakers=self.hps.data.n_speakers,
            **self.hps.model)
        self.net.eval()
        utils.load_checkpoint(pretrained_path, self.net, None)

    def infer(self, text, sid, dir_path, **kwargs):
        sr = self.hps.data.sampling_rate
        processed_text = VITS_Voice_Conversion.get_text(text, self.hps)
        net = self.net.cuda()

        with torch.no_grad():
            sid = torch.LongTensor([sid]).cuda()
            x = processed_text.cuda().unsqueeze(0)
            x_lengths = torch.LongTensor([processed_text.size(0)]).cuda()
            audio = net.infer(x,
                              x_lengths,
                              sid=sid,
                              noise_scale=kwargs['noise_scale'],
                              noise_scale_w=kwargs['noise_scale_w'],
                              length_scale=kwargs['length_scale'])
            audio = audio[0][0, 0].data.cpu().float().numpy()

        soundfile.write(os.path.join(dir_path, f'infer.wav'), audio, sr)

        return

    def load_source_wav(self, path):
        audio, sampling_rate = load_wav_to_torch(path)
        y = audio / self.hps.data.max_wav_value
        self.y = y.unsqueeze(0)
        return self.y

    @property
    def spectrogram(self):
        spec = spectrogram_torch(
            self.y,
            self.hps.data.filter_length,
            self.hps.data.sampling_rate,
            self.hps.data.hop_length,
            self.hps.data.win_length,
            center=False)
        spec_lengths = torch.LongTensor([spec.size(-1)])
        return spec, spec_lengths

    def conversion(self, dir_path, sid_src, *sid_targets):
        spec, spec_lengths = self.spectrogram
        sr = self.hps.data.sampling_rate
        sid_src = torch.LongTensor([sid_src])

        with torch.no_grad():
            for sid in sid_targets:
                sid_tgt = torch.LongTensor([sid])
                audio = self.net.cpu().voice_conversion(spec, spec_lengths, sid_src=sid_src, sid_tgt=sid_tgt)
                audio = audio[0][0, 0].data.float().numpy()
                soundfile.write(os.path.join(dir_path, f'{sid}.wav'), audio, sr)

        return


if __name__ == '__main__':
    vc = VITS_Voice_Conversion(HPS_PATH, MODEL_PATH)
    vc.load_source_wav(SRC_WAV_PATH)
    args = dict(noise_scale=.667, noise_scale_w=0.8, length_scale=1)
    text = ("VITS is Awesome! We propose VITS, "
            "Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech.")
    vc.infer(text, 44, OUT_DIR, **args)
    vc.conversion(OUT_DIR, 17, *[77, 14, 1, 17])
