import whisperx

import torch
from torch.nn import Module


class HuBERTInference(Module):
    def __init__(self, hubert):
        super().__init__()
        self.hubert = hubert

    @torch.inference_mode()
    def forward(self, x):
        x = self.hubert(x)
        x = x.last_hidden_state
        return x


class WhisperXInference(Module): # переписать 
    
    def __init__(self, compute_type, device, language):
        super().__init__()
        self.device = device
        self.language = language
        self.whisper = whisperx.load_model("large-v2", self.device, compute_type=compute_type)
        align_model, metadata = whisperx.load_align_model(language_code=self.language, device=self.device)
        self.align_model = align_model
        self.metadata = metadata
        self.ali = whisperx.align

    @torch.inference_mode()
    def forward(self, x):
        output = self.whisper.transcribe(x)
        x = self.ali(
            output["segments"], self.align_model, self.metadata, 
            x, self.device, return_char_alignments=True,
        )
        return x