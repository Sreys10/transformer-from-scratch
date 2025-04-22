import torch
import torch.nn as nn
from torch.utils.data import Dataset

class BilingualDataset(Dataset):
    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len):
        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang


    self.sos_token= torch.tensor([tokenizer_src.token_to_id("[SOS]")])
    self.eos_token= torch.tensor([tokenizer_src.token_to_id("[EOS]")])