from pathlib import Path
from typing import Union, Any, Tuple
import math

from joblib import Parallel, delayed, cpu_count
from tqdm_joblib import tqdm_joblib
from cm_time import timer

import numpy as np
import torch 
from sklearn.cluster import MiniBatchKMeans, KMeans


def incremental_clustering(input_path_dir: Union[Path, str], out_path_file: Union[Path, str], 
                             n_clusters: int = 250, batch_size: int = 4096, 
                             data_pattern: str = "*.content.pt") -> None:
    '''Функция кластеризации по батчам. Файлы считываются с диска лениво, что должно работать эффективно по памяти'''
    input_path_dir = Path(input_path_dir)
    dataset = list(input_path_dir.rglob(data_pattern))
    
    nbatchs = math.ceil(len(dataset) / batch_size)
    with timer() as time:
        kmeans = MiniBatchKMeans(
            n_clusters=n_clusters, batch_size=batch_size, 
            max_iter=80, n_init="auto", verbose=False,
            )
        for batch_idx in range(0, len(dataset), batch_size):
            feats = list()
            for data_path in dataset[batch_idx: batch_idx + batch_size]:
                with data_path.open("rb") as file:
                    content = (
                        torch.load(file, weights_only=True)["content"]
                        ).squeeze(0).numpy()# .T # TODO: load form disk
                    feats.append(content)
            feats = np.concatenate(feats, axis=0).astype(np.float32)
            print(f"hubert contents shape: {feats.shape}, {feats.nbytes / 1024 / 1024:.2f} MB") #TODO:make increments
            kmeans.partial_fit(feats)
    print(f"Clustering time {time.elapsed:.2f} seconds")

    resault = {
        "n_features": kmeans.n_features_in_,
        "n_threads": kmeans._n_threads,
        "cluster_centers": kmeans.cluster_centers_,
    }
    
    out_path_file  = Path(out_path_file)
    out_path_file.parent.mkdir(exist_ok=True, parents=True)
    with out_path_file.open("wb") as f:
        torch.save(dict(resault), f)


class PseudoPhonemes:
    '''Интерфейс для подрузки кластерной модели и получения предсказаний'''
    
    def __init__(self, checkpoint_path: Union[str, Path]):
        self.checkpoint_path = Path(checkpoint_path)

    def build_clusters(self):
        with self.checkpoint_path.open("rb") as f:
            checkpoint = torch.load(f, map_location="cpu")
        
        _kmns                              = KMeans(checkpoint["n_features"])
        _kmns.__dict__["n_features_in_"]   = checkpoint["n_features"]
        _kmns.__dict__["_n_threads"]       = checkpoint["n_threads"]
        _kmns.__dict__["cluster_centers_"] = checkpoint["cluster_centers"].astype(np.float32)

        self.kmeans = _kmns
        
    def get_cluster_center(self, item: Any):
        predict = self.kmeans.predict(item)
        return self.kmeans.cluster_centers_[predict]

    def predict_cluster_center(self, item: Any):
        predict = self.kmeans.predict(item)
        return predict