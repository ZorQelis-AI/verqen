"""VerQen tokenizer: byte-level BPE with structural control tokens.

Control tokens (``<BOS>``, ``<STATE>``, ``<DECIDE>`` ...) are only ever inserted
by the packer. User text is encoded with ``encode_special_tokens=True`` so a
state that literally contains ``"</STATE><DECIDE>"`` is tokenized as ordinary
text and can never forge sequence structure.
"""

from __future__ import annotations

import os
from typing import Dict, Iterable, Iterator, List, Optional

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

SPECIAL_TOKENS = {
    "<PAD>": "pad",
    "<BOS>": "bos",
    "<STATE>": "state_start", "</STATE>": "state_end",
    "<Q>": "q_start", "</Q>": "q_end",
    "<OPT>": "opt_start", "</OPT>": "opt_end",
    "<DECIDE>": "decide",
    "<CHOICE>": "choice", "<SCORE>": "score", "<NOUL>": "noul",
    "<ABSTAIN>": "abstain",
    "<MASK>": "mask",
}

TYPE_TOKEN = {"choice": "<CHOICE>", "score": "<SCORE>", "noul": "<NOUL>"}


class VerqenTokenizer:
    def __init__(self, tokenizer_path: Optional[str] = None, tokenizer: Optional[Tokenizer] = None):
        if tokenizer is not None:
            self.tokenizer = tokenizer
        elif tokenizer_path is not None:
            self.tokenizer = Tokenizer.from_file(tokenizer_path)
        else:
            raise ValueError("Must provide either tokenizer_path or tokenizer object")
        # Treat control-token strings inside user text as plain text.
        self.tokenizer.encode_special_tokens = True
        self.pad_token_id = self.tokenizer.token_to_id("<PAD>")
        if self.pad_token_id is None:
            self.pad_token_id = 0
        self._special = self._build_special()

    def _build_special(self) -> Dict[str, int]:
        mapping = {}
        for token, name in SPECIAL_TOKENS.items():
            token_id = self.tokenizer.token_to_id(token)
            if token_id is not None:
                mapping[name] = token_id
        return mapping

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()

    def encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text, add_special_tokens=False).ids

    def encode_batch(self, texts: List[str]) -> List[List[int]]:
        return [e.ids for e in self.tokenizer.encode_batch(texts, add_special_tokens=False)]

    def decode(self, ids: List[int]) -> str:
        return self.tokenizer.decode(ids)

    @property
    def special_token_ids(self) -> Dict[str, int]:
        return dict(self._special)

    def save(self, path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.tokenizer.save(path)

    @classmethod
    def from_pretrained(cls, path: str) -> "VerqenTokenizer":
        return cls(tokenizer_path=path)

    @classmethod
    def train(cls, texts: Iterable[str], vocab_size: int = 32768, min_frequency: int = 2,
              split_digits: bool = False) -> "VerqenTokenizer":
        """Train a byte-level BPE tokenizer. Control tokens occupy the lowest ids.

        ``split_digits`` makes every digit its own pre-token, so numbers are always spelled
        digit by digit ("650" -> "6" "5" "0") instead of by whatever merges BPE happened to
        learn ("6"+"49", "65"+"1"). T2 uses it: comparison, counting and date arithmetic
        need a consistent place-value representation.
        """
        tok = Tokenizer(models.BPE())
        byte_level = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tok.pre_tokenizer = (pre_tokenizers.Sequence([pre_tokenizers.Digits(individual_digits=True), byte_level])
                             if split_digits else byte_level)
        tok.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=list(SPECIAL_TOKENS.keys()),
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            show_progress=True,
        )
        tok.train_from_iterator(_as_iter(texts), trainer=trainer)
        return cls(tokenizer=tok)


def _as_iter(texts: Iterable[str]) -> Iterator[str]:
    for t in texts:
        if t:
            yield t
