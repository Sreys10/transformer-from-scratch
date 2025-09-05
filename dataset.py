from typing import Any
import torch
from torch.utils.data import Dataset


class BilingualDataset(Dataset):
    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len):
        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.seq_len = seq_len

        # store token ids as ints (not tensors) to simplify building tensors later
        self.sos_token_src = tokenizer_src.token_to_id("[SOS]")
        self.eos_token_src = tokenizer_src.token_to_id("[EOS]")
        self.pad_token_src = tokenizer_src.token_to_id("[PAD]")

        self.sos_token_tgt = tokenizer_tgt.token_to_id("[SOS]")
        self.eos_token_tgt = tokenizer_tgt.token_to_id("[EOS]")
        self.pad_token_tgt = tokenizer_tgt.token_to_id("[PAD]")

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index: Any) -> Any:
        src_target_pair = self.ds[index]
        src_text = src_target_pair["translation"][self.src_lang]
        tgt_text = src_target_pair["translation"][self.tgt_lang]

        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        # pad/truncate to fit seq_len
        enc_input_tokens = enc_input_tokens[: self.seq_len - 2]  # reserve for SOS/EOS
        dec_input_tokens = dec_input_tokens[: self.seq_len - 1]  # reserve for SOS or EOS depending usage

        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2  # SOS + EOS
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1  # only SOS in decoder_input

        # Encoder input = SOS + tokens + EOS + PAD
        encoder_list = [self.sos_token_src] + enc_input_tokens + [self.eos_token_src] + [self.pad_token_src] * enc_num_padding_tokens
        encoder_input = torch.tensor(encoder_list, dtype=torch.long)

        # Decoder input = SOS + tokens + PAD
        decoder_list = [self.sos_token_tgt] + dec_input_tokens + [self.pad_token_tgt] * dec_num_padding_tokens
        decoder_input = torch.tensor(decoder_list, dtype=torch.long)

        # Label = tokens + EOS + PAD (what model should predict)
        label_list = dec_input_tokens + [self.eos_token_tgt] + [self.pad_token_tgt] * dec_num_padding_tokens
        label = torch.tensor(label_list, dtype=torch.long)

        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        # Masks (boolean)
        # encoder_mask shape per sample: (1, 1, seq_len) -> stacked to (batch,1,1,seq_len)
        encoder_mask = (encoder_input != self.pad_token_src).unsqueeze(0).unsqueeze(0)

        # decoder pad mask shape per sample: (1,1,seq_len)
        dec_pad_mask = (decoder_input != self.pad_token_tgt).unsqueeze(0).unsqueeze(0)

        # causal mask per sample: (1, seq_len, seq_len)
        causal = self.causal_mask(self.seq_len)

        # combine: dec_pad_mask (1,1,seq_len) will broadcast against causal (1,seq_len,seq_len)
        # result shape -> (1, seq_len, seq_len) (then stacked across batch)
        decoder_mask = dec_pad_mask & causal

        return {
            "encoder_input": encoder_input,     # (seq_len,)
            "decoder_input": decoder_input,     # (seq_len,)
            "encoder_mask": encoder_mask,       # (1,1,seq_len)  (bool)
            "decoder_mask": decoder_mask,       # (1,seq_len,seq_len) (bool)
            "label": label,                     # (seq_len,)
            "src_text": src_text,
            "tgt_text": tgt_text,
        }

    @staticmethod
    def causal_mask(size):
        """Creates a causal mask for decoder self-attention (1, seq_len, seq_len)"""
        mask = torch.triu(torch.ones((1, size, size), dtype=torch.bool), diagonal=1)
        return ~mask  # lower triangle True (allowed), upper triangle False
