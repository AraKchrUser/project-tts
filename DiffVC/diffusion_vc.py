import argparse
import json
import os
import numpy as np
import soundfile
from tqdm import tqdm
from scipy.io.wavfile import write
import torch

use_gpu = torch.cuda.is_available()
import librosa
from librosa.core import load
from librosa.filters import mel as librosa_mel_fn

mel_basis = librosa_mel_fn(sr=22050, n_fft=1024, n_mels=80, fmin=0, fmax=8000)

import sys

sys.path.append('../gradtts_cloning/DiffVC')
import params
from model import DiffVC

sys.path.append('../gradtts_cloning/DiffVC/hifi-gan/')
from env import AttrDict
from models import Generator as HiFiGAN

sys.path.append('../gradtts_cloning/DiffVC/speaker_encoder/')
from encoder import inference as spk_encoder
from pathlib import Path

USE_GPU = use_gpu
ROOT_PATH = '../gradtts_cloning/DiffVC/'
VC_PATH = os.path.join(ROOT_PATH, 'checkpts/vc/vc_libritts_wodyn.pt')
HFG_PATH = os.path.join(ROOT_PATH, 'checkpts/vocoder/')
SPK_ENCODER_PATH = os.path.join(ROOT_PATH, 'checkpts/spk_encoder/pretrained.pt')
ROOT_PATH = '../vits_cloning'
SRC_WAV_PATH = os.path.join(ROOT_PATH, './path/to/Test/test3_F.wav')
TGT_WAV_PATH = os.path.join(ROOT_PATH, './path/to/Test/test1_M.wav')


class Diffusion_Voice_Conversion:

    @staticmethod
    def get_mel(wav_path):
        wav, _ = load(wav_path, sr=22050)
        wav = wav[:(wav.shape[0] // 256) * 256]
        wav = np.pad(wav, 384, mode='reflect')
        stft = librosa.core.stft(wav, n_fft=1024, hop_length=256, win_length=1024, window='hann', center=False)
        stftm = np.sqrt(np.real(stft) ** 2 + np.imag(stft) ** 2 + 1e-9)
        mel_spectrogram = np.matmul(mel_basis, stftm)
        log_mel_spectrogram = np.log(np.clip(mel_spectrogram, a_min=1e-5, a_max=None))
        return log_mel_spectrogram

    @staticmethod
    def get_embed(spk_encoder, wav_path):
        wav_preprocessed = spk_encoder.preprocess_wav(wav_path)
        embed = spk_encoder.embed_utterance(wav_preprocessed)
        return embed

    @staticmethod
    def noise_median_smoothing(x, w=5):
        y = np.copy(x)
        x = np.pad(x, w, "edge")
        for i in range(y.shape[0]):
            med = np.median(x[i:i + 2 * w + 1])
            y[i] = min(x[i + w + 1], med)
        return y

    def mel_spectral_subtraction(mel_synth, mel_source, spectral_floor=0.02, silence_window=5, smoothing_window=5):
        '''Алгоритм спектрального вычитания для уменьшения шума в синтезированной мел-спектрограмме.
        Спектр шума определяется из исходного мел-спектра по фрагментам, которые являются тишиной'''
        mel_len = mel_source.shape[-1]
        energy_min = 100000.0
        i_min = 0
        for i in range(mel_len - silence_window):
            energy_cur = np.sum(np.exp(2.0 * mel_source[:, i:i + silence_window]))
            if energy_cur < energy_min:
                i_min = i
                energy_min = energy_cur
        estimated_noise_energy = np.min(np.exp(2.0 * mel_synth[:, i_min:i_min + silence_window]), axis=-1)
        if smoothing_window is not None:
            estimated_noise_energy = Diffusion_Voice_Conversion.noise_median_smoothing(estimated_noise_energy,
                                                                                       smoothing_window)
        mel_denoised = np.copy(mel_synth)
        for i in range(mel_len):
            signal_subtract_noise = np.exp(2.0 * mel_synth[:, i]) - estimated_noise_energy
            estimated_signal_energy = np.maximum(signal_subtract_noise, spectral_floor * estimated_noise_energy)
            mel_denoised[:, i] = np.log(np.sqrt(estimated_signal_energy))
        return mel_denoised

    def __init__(self, vc_path, hfg_path, spk_encoder_path, use_gpu):

        self.use_gpu = use_gpu
        self.__spk_encoder = spk_encoder
        self.__generator = None
        self.__hifigan = None
        self._source_wav = None
        self._target_wav = None

        self.__generator = DiffVC(params.n_mels, params.channels, params.filters, params.heads,
                                  params.layers, params.kernel, params.dropout, params.window_size,
                                  params.enc_dim, params.spk_dim, params.use_ref_t, params.dec_dim,
                                  params.beta_min, params.beta_max)
        if self.use_gpu:
            self.__generator = self.__generator.cuda()
            self.__generator.load_state_dict(torch.load(vc_path))
        else:
            self.__generator.load_state_dict(torch.load(vc_path, map_location='cpu'))
        self.__generator.eval()

        with open(hfg_path + 'config.json') as f:
            h = AttrDict(json.load(f))
        if self.use_gpu:
            self.__hifigan = HiFiGAN(h).cuda()
            self.__hifigan.load_state_dict(torch.load(hfg_path + 'generator')['generator'])
        else:
            self.__hifigan = HiFiGAN(h)
            self.__hifigan.load_state_dict(torch.load(hfg_path + 'generator', map_location='cpu')['generator'])
        _ = self.__hifigan.eval()
        self.__hifigan.remove_weight_norm()

        enc_model_fpath = Path(spk_encoder_path)
        if self.use_gpu:
            self.__spk_encoder.load_model(enc_model_fpath, device="cuda")
        else:
            self.__spk_encoder.load_model(enc_model_fpath, device="cpu")

    @property
    def source_wav(self):
        return self._source_wav

    @property
    def target_wav(self):
        return self._target_wav

    @source_wav.setter
    def source_wav(self, path):
        self._source_wav = path
        return

    @target_wav.setter
    def target_wav(self, path):
        self._target_wav = path
        return

    @property
    def mel_source(self):
        mel_source = torch.from_numpy(Diffusion_Voice_Conversion.get_mel(self.source_wav)).float().unsqueeze(0)
        if self.use_gpu:
            mel_source = mel_source.cuda()
        mel_source_lengths = torch.LongTensor([mel_source.shape[-1]])
        if self.use_gpu:
            mel_source_lengths = mel_source_lengths.cuda()
        return mel_source, mel_source_lengths

    @property
    def mel_target(self):
        mel_target = torch.from_numpy(Diffusion_Voice_Conversion.get_mel(self.target_wav)).float().unsqueeze(0)
        if self.use_gpu:
            mel_target = mel_target.cuda()
        mel_target_lengths = torch.LongTensor([mel_target.shape[-1]])
        if self.use_gpu:
            mel_target_lengths = mel_target_lengths.cuda()
        return mel_target, mel_target_lengths

    @property
    def embed_target(self):
        embed_target = Diffusion_Voice_Conversion.get_embed(self.__spk_encoder, self.target_wav)
        embed_target = torch.from_numpy(embed_target).float().unsqueeze(0)
        if self.use_gpu:
            embed_target = embed_target.cuda()
        return embed_target

    def conversion(self, out_file, sr=22050):

        mel_source, mel_source_lengths = self.mel_source
        mel_target, mel_target_lengths = self.mel_target
        embed_target = self.embed_target

        _, mel_ = self.__generator.forward(mel_source, mel_source_lengths,
                                           mel_target, mel_target_lengths,
                                           embed_target, n_timesteps=30, mode='ml')

        mel_synth_np = mel_.cpu().detach().squeeze().numpy()
        mel_source_np = mel_.cpu().detach().squeeze().numpy()
        mel = Diffusion_Voice_Conversion.mel_spectral_subtraction(mel_synth_np, mel_source_np, smoothing_window=1)
        mel = torch.from_numpy(mel).float().unsqueeze(0)

        if self.use_gpu:
            mel = mel.cuda()
        with torch.no_grad():
            audio = self.__hifigan.forward(mel).cpu().squeeze().clamp(-1, 1)

        soundfile.write(out_file, audio, sr)
        return


if __name__ == "__main__":
    vc = Diffusion_Voice_Conversion(VC_PATH, HFG_PATH, SPK_ENCODER_PATH, USE_GPU)
    vc.source_wav = SRC_WAV_PATH
    vc.target_wav = TGT_WAV_PATH
    vc.conversion(out_file='out/infer.wav')
