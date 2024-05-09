from string import punctuation

from models import WhisperX


class SpeechEditor:
    def __init__(self) -> None:
        self.whisperx_model = WhisperX()
        return
    
    def editing(self, src_text, tgt_text, audio_path):
        audio = WhisperX.load_audio(audio_file=audio_path)
        out = self.whisperx_model(audio)
        alignment = WhisperX.postprocess_out(out, by='words')
        timesteps = WhisperX.formed_timesteps(alignment)
        if src_text is None:
            src_text = out['segments'][0]['text']
        wer_info = SpeechEditor.levenshtein(src_text, tgt_text)
        # ...

    def deleting(self):
        pass

    def inserting(self):
        pass

    def sub(self):
        pass

    @staticmethod
    def levenshtein(reference_text: str, recognized_text: str):
        
        remove_punctuation = lambda string: ''.join(filter(lambda sym: sym not in punctuation, string.lower().strip())).split()
        reference_words = remove_punctuation(reference_text)
        recognized_words = remove_punctuation(recognized_text)

        distance_matrix = [[0] * (len(recognized_words) + 1) for _ in range(len(reference_words) + 1)]
        for i in range(len(reference_words) + 1):
            distance_matrix[i][0] = i
        for j in range(len(recognized_words) + 1):
            distance_matrix[0][j] = j
        for i in range(1, len(reference_words) + 1):
            for j in range(1, len(recognized_words) + 1):
                if reference_words[i - 1] == recognized_words[j - 1]:
                    distance_matrix[i][j] = distance_matrix[i - 1][j - 1]
                else:
                    insert = distance_matrix[i][j - 1] + 1
                    delete = distance_matrix[i - 1][j] + 1
                    substitute = distance_matrix[i - 1][j - 1] + 1
                    distance_matrix[i][j] = min(insert, delete, substitute)
        wer = distance_matrix[-1][-1] / len(reference_words) * 100
        
        ali = [[] for _ in range(3)]
        correct = 0
        insertion = 0
        substitution = 0
        deletion = 0
        i, j = len(reference_words), len(recognized_words)
        while True:
            if i == 0 and j == 0:
                break
            elif (i >= 1 and j >= 1
                and distance_matrix[i][j] == distance_matrix[i - 1][j - 1] 
                and reference_words[i - 1] == recognized_words[j - 1]):
                ali[0].append(reference_words[i - 1])
                ali[1].append(recognized_words[j - 1])
                ali[2].append('C')
                correct += 1
                i -= 1
                j -= 1
            elif j >= 1 and distance_matrix[i][j] == distance_matrix[i][j - 1] + 1:
                ali[0].append("***")
                ali[1].append(recognized_words[j - 1])
                ali[2].append('I')
                insertion += 1
                j -= 1
            elif i >= 1 and j >= 1 and distance_matrix[i][j] == distance_matrix[i - 1][j - 1] + 1:
                ali[0].append(reference_words[i - 1])
                ali[1].append(recognized_words[j - 1])
                ali[2].append('S')
                substitution += 1
                i -= 1
                j -= 1
            else:
                ali[0].append(reference_words[i - 1])
                ali[1].append("***")
                ali[2].append('D')
                deletion += 1
                i -= 1
        ali[0] = ali[0][::-1]
        ali[1] = ali[1][::-1]
        ali[2] = ali[2][::-1]
        assert len(ali[0]) == len(ali[1]) == len(ali[2]), f"wrong ali {ali}"
        
        return {"wer" : wer,
                "cor": correct, 
                "del": deletion,
                "ins": insertion,
                "sub": substitution,
                "ali": ali,
                "reference_words": reference_words,
                "recognized_words": recognized_words,
                }