
from collections import defaultdict
from typing import Callable


def get_all_alignments(files_to_whispout: dict, calc_hubert_ali_for_one_char: Callable) -> dict:
    all_alignments = defaultdict(lambda: defaultdict(list))
    for file, whisper_output in files_to_whispout.items(): #TODO: tqdm
        for alignment in whisper_output:
            try:
                char, content_ids = calc_hubert_ali_for_one_char(alignment)
                all_alignments[file][char].append(content_ids)
            except:
                continue
    return all_alignments