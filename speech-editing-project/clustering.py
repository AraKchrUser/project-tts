
from pathlib import Path
from typing import Union, Any, Tuple

from joblib import Parallel, delayed, cpu_count
from tqdm_joblib import tqdm_joblib
from cm_time import timer

import numpy as np
import torch 
from sklearn.cluster import MiniBatchKMeans, KMeans


class CharClusters:
    '''https://github.com/voicepaw/so-vits-svc-fork/blob/main/src/so_vits_svc_fork/cluster/__init__.py'''
    
    def __init__(self, checkpoint_path: Union[str, Path]):
        self.checkpoint_path = Path(checkpoint_path)

    def build_chars_clusters(self):
        with self.checkpoint_path.open("rb") as f:
            checkpoint = torch.load(f, map_location="cpu")
        kmeans = dict()
        for char, ckpt in checkpoint.items():
            _kmns                              = KMeans(ckpt["n_features"])
            _kmns.__dict__["n_features_in_"]   = ckpt["n_features"]
            _kmns.__dict__["_n_threads"]       = ckpt["n_threads"]
            _kmns.__dict__["cluster_centers_"] = ckpt["cluster_centers"].astype(np.float32)
            kmeans[char]                       = _kmns
        self.kmeans = kmeans
        
    def get_cluster_center(self, char: str, item: Any):
        model   = self.kmeans[char]
        predict = model.predict(item)
        return model.cluster_centers_[predict]

    def predict_cluster_center(self, char: str, item: Any):
        model   = self.kmeans[char]
        predict = model.predict(item)
        return predict

def _build_and_fit_minibatch_kmeans(input_path: Path, data_pattern: str, 
                     n_clusters: int, batch_size: int = 4096) -> dict:
    # input_path = Path(input_path)
    feats = list()
    for data_path in input_path.rglob(data_pattern):
        with data_path.open("rb") as file:
            content = (
                torch.load(file, weights_only=True)["content"]
                ).squeeze(0).numpy()# .T # TODO: load form disk
            feats.append(content)
    if not feats:
        raise Exception()
    feats = np.concatenate(feats, axis=0).astype(np.float32)
    print(f"hubert contents shape: {feats.shape}, {feats.nbytes / 1024 / 1024:.2f}")
    with timer() as time:
        kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=batch_size, 
                                 n_init="auto", max_iter=100, verbose=False)
        kmeans = kmeans.fit(feats)
    print(f"Clustering time {time.elapsed:.2f} seconds")
    return {
        "n_features": kmeans.n_features_in_,
        "n_threads": kmeans._n_threads,
        "cluster_centers": kmeans.cluster_centers_,
    }

def _kmeans_fitting(input_path: Path, **kwargs: Any) -> Tuple[str, dict]:
    return input_path.stem, _build_and_fit_minibatch_kmeans(input_path, **kwargs)

def cluster_training_on_data(input_path_dir: Union[Path, str], out_path_file: Union[Path, str], 
                             n_clusters: int = 10_000, batch_size: int = 4096, 
                             data_pattern: str = "*.content.pt") -> None:
    input_path_dir = Path(input_path_dir)
    out_path_file  = Path(out_path_file)
    with tqdm_joblib(desc="Training clusters", total=len(list(input_path_dir.iterdir()))): # TODO: tqdm
        resault = Parallel(n_jobs=3)(delayed(_kmeans_fitting)(
            char, data_pattern=data_pattern, 
            n_clusters=n_clusters, batch_size=batch_size
        ) for char in input_path_dir.iterdir())
    out_path_file.parent.mkdir(exist_ok=True, parents=True)
    with out_path_file.open("wb") as f:
        torch.save(dict(resault), f)