"""
 Copyright 2024 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

from pathlib import Path
from typing import Dict

import sentencepiece as spm
import yaml

from ...utils.misc import PathLike
from .hyp_tokenizer import HypTokenizer


class SPTokenizer(HypTokenizer):
    """Sentence Piece Tokenizer"""

    def __init__(
        self, sp_model: spm.SentencePieceProcessor, uppercase_text: bool = True
    ):
        super().__init__()
        self.sp_model = sp_model
        self.uppercase_text = uppercase_text
        self.blank_id = self.sp_model.piece_to_id("<blk>")
        self.vocab_size = self.sp_model.get_piece_size()
        self._token2id = None

    @property
    def token2id(self):
        if self._token2id is not None:
            return self._token2id

        token2id: Dict[str, int] = {
            self.sp_model.id_to_piece(i): i for i in range(self.sp_model.vocab_size())
        }
        self._token2id = token2id
        return token2id

    def normalize(self, text):
        if self.uppercase_text:
            text = text.upper()
        return text

    def encode(self, text):
        return self.sp_model.encode(text, out_type=int)

    def decode(self, tokens):
        return self.sp_model.decoder(tokens)

    def save(self, file_path: PathLike, sp_model_prefix: str = "tokenizer"):
        file_path = Path(file_path)
        if file_path.suffix != ".yaml":
            output_dir = file_path
            file_path = output_dir / (sp_model_prefix + ".yaml")
        else:
            output_dir = file_path.parent

        output_dir.mkdir(parents=True, exist_ok=True)
        sp_model_file = sp_model_prefix + ".model"
        sp_tokens_file = sp_model_prefix + ".tokens"
        cfg = {
            "class_name": self.__class__.__name__,
            "sp_model": sp_model_file,
            "sp_tokens": sp_tokens_file,
            "uppercase_text": self.uppercase_text,
        }
        with open(file_path, "w") as f:
            yaml.dump(cfg, f)

        with open(output_dir / sp_tokens_file, "w", encoding="utf-8") as f:
            for sym, i in self.token2id.items():
                f.write(f"{sym} {i}\n")

    @classmethod
    def load(cls, file_path: PathLike):
        file_path = Path(file_path)
        if file_path.suffix == ".model":
            sp_model = spm.SentencePieceProcessor()
            sp_model.load(str(file_path))
            return cls(sp_model)

        with open(file_path, "r") as f:
            cfg = yaml.safe_load(f)

        sp_model_file = Path(cfg["sp_model"])
        if not sp_model_file.is_file():
            sp_model_file = file_path.parent / sp_model_file
            assert sp_model_file.is_file(), f"{sp_model_file} not found"

        sp_model = spm.SentencePieceProcessor()
        sp_model.load(str(sp_model_file))
        return cls(sp_model)
