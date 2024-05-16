from inferencers import *
from utils import *
from datasets import *

from tqdm import tqdm

import torch


def _one_item_whisper_infer(file, whisper):
    res = (
        file, postprocess_whisper_output(
            whisper_inference_for_file(file, whisper)
        )
    )
    torch.cuda.empty_cache()
    return res

def _batch_whisper_infer(files, pbar):
    res = []
    device = "cuda"
    whisper = WhisperXInference("float16", device, "ru")
    for file in tqdm(files, position=pbar):
        res.append(_one_item_whisper_infer(file, whisper))
    wip_memory(whisper)
    return res


def postprocess_whisper_output(output):
    segments = output.get("segments", [])
    if segments:
        if not isinstance(segments[0], dict):
            return []
        return segments[0].get("chars", [])
    return []