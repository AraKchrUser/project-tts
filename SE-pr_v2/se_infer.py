from typing import *
from pathlib import Path
from copy import deepcopy

from cm_time import timer
from tqdm import tqdm
import librosa
import soundfile
import numpy as np
import torch
import gc
import so_vits_svc_fork
from so_vits_svc_fork.inference.core import Svc, split_silence
from so_vits_svc_fork.utils import repeat_expand_2d, get_content

VERSION = 1


class SvcSpeechEdit(Svc):

    def infer_silence(
            self, audio, *,
            # replaced data info
            src_idxs, tgt_contents,
            # svc config
            speaker, transpose, auto_predict_f0=False,
            cluster_infer_ratio=0, noise_scale=0.4, f0_method: Literal[
                            "crepe", "crepe-tiny", 
                            "parselmouth", "dio", "harvest",
                        ] = "dio",
            # slice config
            db_thresh=-40, pad_seconds=0.5, chunk_seconds=0.5,
            absolute_thresh=False, max_chunk_seconds=40,
            ): # -> np.ndarray
        sr = self.target_sample
        out = np.array([], dtype=np.float32)
        chunk_length_min = chunk_length_min = (int(min(
            sr / so_vits_svc_fork.f0.f0_min * 20 + 1, chunk_seconds * sr,
            )) // 2)
        splited_silence = split_silence(
            audio, top_db=-db_thresh, frame_length=chunk_length_min * 2, 
            hop_length=chunk_length_min, ref=1 if absolute_thresh else np.max, 
            max_chunk_length=int(max_chunk_seconds * sr),
            )
        for chunk in splited_silence:
            if not chunk.is_speech:
                audio_chunk_infer = np.zeros_like(chunk.audio)
            else:
                pad_len = int(sr * pad_seconds)
                audio_chunk_pad = np.concatenate([
                    np.zeros([pad_len], dtype=np.float32), 
                    chunk.audio, 
                    np.zeros([pad_len], dtype=np.float32),
                    ])
                audio_chunk_padded = self.infer(
                    speaker, transpose, audio_chunk_pad, 
                    cluster_infer_ratio=cluster_infer_ratio,
                    auto_predict_f0=auto_predict_f0,
                    noise_scale=noise_scale, f0_method=f0_method,
                    replaced_contents_info=(src_idxs, tgt_contents),
                    )[0].cpu().numpy()
                pad_len = int(self.target_sample * pad_seconds)
                cut_len_2 = (len(audio_chunk_padded) - len(chunk.audio)) // 2
                audio_chunk_infer = audio_chunk_padded[cut_len_2 : cut_len_2 + len(chunk.audio)]
                torch.cuda.empty_cache()
            
            out = np.concatenate([out, audio_chunk_infer])
        
        return out[: audio.shape[0]]

    
    def infer(self, speaker, transpose, audio, 
              cluster_infer_ratio, auto_predict_f0, noise_scale, 
              f0_method, replaced_contents_info,
              ): # -> tuple[torch.Tensor, int]:
        audio = audio.astype(np.float32)
        if isinstance(speaker, int):
            if len(self.spk2id.__dict__) >= speaker:
                speaker_id = speaker
            else:
                raise ValueError()
        else:
            if speaker in self.spk2id.__dict__:
                speaker_id = self.spk2id.__dict__[speaker]
            else:
                speaker_id = 0
        speaker_candidates = list(
            filter(lambda x: x[1] == speaker_id, self.spk2id.__dict__.items())
        )
        if len(speaker_candidates) > 1:
            raise ValueError()
        elif len(speaker_candidates) == 0:
            raise ValueError()
        speaker = speaker_candidates[0][0]
        sid = torch.LongTensor([int(speaker_id)])
        sid = sid.to(self.device).unsqueeze(0)

        idxs, contents = replaced_contents_info
        
        if VERSION == 1:
            # (499, 256)
            c, f0, uv = self.get_unit_f0(
                None, audio, transpose, cluster_infer_ratio, speaker, f0_method
            ) #TODO: Добавить предсказание F0 для замененных кластеров 
            # contents = torch.from_numpy(np.stack(contents, axis=0)) # [N, 256] -> 
            # contents = contents.transpose(1, 0) # [256, N], where [h, t]
            # # if c.shape[-1] != contents.shape[-1]
            # contents = torch.nn.functional.interpolate(
            #     contents.unsqueeze(0), size=contents.shape[-1]*2, mode="nearest"
            #         ).squeeze(0)
            n = 5
            # 
            # for i in range(len(idxs)):
            #     for _ in range(n):
            #         expended_idx.append(idxs[i])
            # _contents = deepcopy(contents)
            # contents = {}
            # for i, idx in enumerate(idxs):
            #     contents[idx] = _contents[i]
            # expended_idx = []
            # for i in range(idxs[1], idxs[0] + len(idxs)*n):
            #     expended_idx.append(i)
           
            # _contents = deepcopy(contents)
            # contents = []
            # for i in range(len(idxs)):
            #     for _ in range(n):
            #         contents.append(_contents[i])
            # print(f"{len(idxs) * n=} {c.shape=} {idxs}")
            # for i in range(len(idxs) * n):
            #     idx = i
            #     c[0, :, idx] = torch.from_numpy(contents[idx])
            

            # Теперь используем кластеризатор 
            from clustering import PseudoPhonemes
            clusters_path = "../../NIR/RuDevices_content_clusters/clusters_10000.pt"
            # TODO check clusters count 
            semantic_codes_clusters = PseudoPhonemes(clusters_path)
            semantic_codes_clusters.build_clusters()
            semantic_codes_clusters.on_labling_mode()

            new_c = c[0, :, 12:120].squeeze(0).transpose(1, 0) # [256, N] -> [N, 256]
            new_c = new_c.cpu().numpy().astype(np.float32)
            print(f"{new_c.shape=}")
            for i, semantic in enumerate(new_c):
                semantic = np.array(semantic)
                assert len(semantic) == 256
                semantic = semantic.reshape(1, -1)
                semantic = semantic_codes_clusters.predict_cluster_center(semantic)[0]
                c[0, :, 192+i] = semantic #0

            # c[0, :, 192:300] = c[0, :, 12:120]
            f0[192:300] = f0[12:120]
            
            print(f"{f0.shape=} {c.shape=}")
            # print(f"{c.shape=}, ") #{contents.shape=}, {idxs=} [1, 256, 801]
        else:
            print(f"{contents.shape=}, ")
            c = torch.from_numpy(contents).transpose(1, 0).unsqueeze(0).to("cuda") #.transpose(0, 1)
            c, f0, uv = self.get_unit_f0(
                c, audio, transpose, cluster_infer_ratio, speaker, f0_method
            ) #TODO: Добавить предсказание F0 для замененных кластеров 
            print(f"{c.shape=}, ") # [1, 256, 801]

        with torch.no_grad():
            with timer() as t:
                audio = self.net_g.infer(
                    c, f0=f0, g=sid, uv=uv,
                    predict_f0=auto_predict_f0,
                    noice_scale=noise_scale,
                )[0, 0].data.float()
            audio_duration = audio.shape[-1] / self.target_sample
            print(f"Inference time: {t.elapsed:.2f}s, RTF: {t.elapsed / audio_duration:.2f}")
        torch.cuda.empty_cache()
        return audio, audio.shape[-1]
    

    def get_unit_f0(
        self,
        c: Any, 
        audio: np.array,
        tran: int,
        cluster_infer_ratio: float,
        speaker: int | str,
        f0_method: Literal[
            "crepe", "crepe-tiny", "parselmouth", "dio", "harvest"
        ] = "dio",
    ):
        f0 = so_vits_svc_fork.f0.compute_f0(
            audio,
            sampling_rate=self.target_sample,
            hop_length=self.hop_size,
            method=f0_method,
        )
        f0, uv = so_vits_svc_fork.f0.interpolate_f0(f0)
        f0 = torch.as_tensor(f0, dtype=self.dtype, device=self.device)
        uv = torch.as_tensor(uv, dtype=self.dtype, device=self.device)
        f0 = f0 * 2 ** (tran / 12)
        f0 = f0.unsqueeze(0)
        uv = uv.unsqueeze(0)

        if VERSION == 1:
            c = get_content(
                    self.hubert_model,
                    audio,
                    self.device,
                    self.target_sample,
                    self.contentvec_final_proj,
                ).to(self.dtype)
        c = repeat_expand_2d(c.squeeze(0), f0.shape[1]) #

        # if cluster_infer_ratio != 0:
        #     cluster_c = cluster.get_cluster_center_result(
        #         self.cluster_model, c.cpu().numpy().T, speaker
        #     ).T
        #     cluster_c = torch.FloatTensor(cluster_c).to(self.device)
        #     c = cluster_infer_ratio * cluster_c + (1 - cluster_infer_ratio) * c

        c = c.unsqueeze(0)
        return c, f0, uv



class SVCInfer:

    def __init__(self, model_path: Union[Path, str], conf_path: Union[Path, str],
                 auto_predict_f0: bool, noise_scale: float=.4, device: str="cuda",
                 f0_method: Literal["crepe", "crepe-tiny", "parselmouth", "dio", "harvest"]="dio",
                 ):
        
        self.auto_predict_f0 = auto_predict_f0
        self.f0_method = f0_method
        self.noise_scale = noise_scale
        self.device = device

        # Используется для нахождения trade-off между схожестью 
        # спикеров и понятностью речи (тут не используется)
        self.speaker_cluster_path = None
        self.transpose = 0
        self.cluster_infer_ratio = 0
        # Slice confs
        self.db_thresh = -40
        self.pad_seconds = .5
        self.chunk_seconds = .5
        self.absolute_thresh = False
        self.max_chunk_seconds = 40

        model_path = Path(model_path)
        conf_path = Path(conf_path)

        self.svc_model = SvcSpeechEdit(
            net_g_path=model_path.as_posix(), config_path=conf_path.as_posix(), 
            cluster_model_path=None, device=device,
            )


    @staticmethod
    def prepare_data(input_paths: List[Union[Path, str]], output_dir: Union[Path, str]):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        input_paths = [Path(p) for p in input_paths]
        output_paths = [output_dir / p.name for p in input_paths]
        print(f"{input_paths=}, {output_paths=}")
        return input_paths, output_paths


    def inference(self, input_paths: List[Union[Path, str]], 
                  output_dir: Union[Path, str], speaker: Union[int, str],
                  src_idxs=Optional[List[int]], tgt_contents=Optional[np.array],
                  ):

        input_paths, output_paths = SVCInfer.prepare_data(input_paths, output_dir)
        
        pbar = tqdm(list(zip(input_paths, output_paths)), disable=len(input_paths) == 1)
        for input_path, output_path in pbar:
            audio, _ = librosa.load(str(input_path), sr=self.svc_model.target_sample)

            audio = self.svc_model.infer_silence(
                    audio.astype(np.float32), speaker=speaker, 
                    transpose=self.transpose, auto_predict_f0=self.auto_predict_f0, 
                    cluster_infer_ratio=self.cluster_infer_ratio, noise_scale=self.noise_scale, 
                    f0_method=self.f0_method, db_thresh=self.db_thresh, pad_seconds=self.pad_seconds, 
                    chunk_seconds=self.chunk_seconds, absolute_thresh=self.absolute_thresh, 
                    max_chunk_seconds=self.max_chunk_seconds, 
                    src_idxs=src_idxs, tgt_contents=tgt_contents,
                )
            soundfile.write(str(output_path), audio, self.svc_model.target_sample)
    
    def __del__(self):
        del self.svc_model
        gc.collect()
        torch.cuda.empty_cache()