# YourTTS train script

Training the YourTTS-model (parallel end-to-end TTS method that generates more natural sounding audio than current two-stage models) from the article [paper](https://arxiv.org/abs/2106.06103)
Beginning of work:
```shell
git clone https://github.com/coqui-ai/TTS.git
pip install TTS
pip install Trainer==0.0.20
apt-get install espeak
import TTS # in python code
```
Inference example:
```shell
CUDA_VISIBLE_DEVICES="0" tts --text "Привет, думаешь, качество синтеза после файн-тюнинга приемлимое?" --model_path "./YourTTS-RU-RUSLAN-April-30-2023_03+48PM-0000000/best_model.pth" --config_path "./YourTTS-RU-RUSLAN-April-30-2023_03+48PM-0000000/config.json" --out_path "rusl_testing.wav" --encoder_config_path "
/home/stc/.local/share/tts/tts_models--multilingual--multi-dataset--your_tts/config_se.json" --encoder_path "/home/stc/.local/share/tts/tts_models--multilingual--multi-dataset--your_tt
s/model_se.pth" --speakers_file_path "/mnt/storage/kocharyan/NIR/../datasets_ruslan/ruslan_ds/speakers.pth" --speaker_idx 0 --speaker_wav  "../datasets_ruslan/ruslan_ds/RUSLAN/016117_R
USLAN.wav"
```

Train example:
```shell
CUDA_VISIBLE_DEVICES="0" python train-your-tts.py
```
If troubles during resample - use code bellow 
```python
import librosa
import os
import soundfile as sf

dir_ = '../datasets_ruslan/ruslan_ds/RUSLAN/'
for i, f in enumerate(os.listdir(dir_)):
    file = os.path.join('../datasets_ruslan/ruslan_ds/RUSLAN/', f)
    audio, sr = librosa.load(path=file, sr=None)
    audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    sf.write(file, audio, samplerate=16000)
    if i % 1000 == 0: print(i)
```

#### TODO: Use argparse / Refactoring code / Build venv