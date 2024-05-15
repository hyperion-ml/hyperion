#!/usr/bin/env python
"""
 Copyright 2024 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
import logging
import os
from pathlib import Path
from typing import Dict, List

import sentencepiece as spm
from jsonargparse import (
    ActionConfigFile,
    ActionParser,
    ActionYesNo,
    ArgumentParser,
    namespace_to_dict,
)

from hyperion.hyp_defs import config_logger
from hyperion.utils import PathLike, SegmentSet

tokenizer_list = ["sentencepiece"]


def add_common_args(parser):
    parser.add_argument(
        "--segments-file",
        required=True,
        help="input segments file with sentence transcriptions",
    )
    parser.add_argument(
        "--text-column", default="text", help="text column in segments file"
    )
    parser.add_argument("--tokenizer-path", required=True, help="tokenizer model dir")
    parser.add_argument(
        "-v",
        "--verbose",
        dest="verbose",
        default=1,
        choices=[0, 1, 2, 3],
        type=int,
    )


def train_sentencepiece(
    segments_file: PathLike,
    text_column: str,
    vocab_size: int,
    model_type: str,
    char_coverage: str,
    sentence_size: int,
    user_defined_symbols: List[str],
    unk_id: int,
    sos_id: int,
    eos_id: int,
    pad_id: int,
    unk_piece: str,
    sos_piece: str,
    eos_piece: str,
    pad_piece: str,
    uppercase_text: bool,
    tokenizer_path: PathLike,
):
    from hyperion.torch.tokenizers import SPTokenizer

    tokenizer_path = Path(tokenizer_path)
    tokenizer_path.mkdir(exist_ok=True, parents=True)

    text_file = tokenizer_path / "text"
    if not text_file.is_file():
        segments = SegmentSet.load(segments_file)
        with open(text_file, "w", encoding="utf-8") as f_text:
            for text in segments[text_column]:
                if uppercase_text:
                    text = text.upper()
                f_text.write(f"{text}\n")

    model_prefix = tokenizer_path / "tokenizer"
    model_file = model_prefix.with_suffix(".model")
    if not model_file.is_file():
        spm.SentencePieceTrainer.train(
            input=text_file,
            vocab_size=vocab_size,
            model_type=model_type,
            model_prefix=str(model_prefix),
            input_sentence_size=sentence_size,
            character_coverage=char_coverage,
            user_defined_symbols=user_defined_symbols,
            unk_id=unk_id,
            bos_id=sos_id,
            eos_id=eos_id,
            pad_id=pad_id,
            unk_piece=unk_piece,
            bos_piece=sos_piece,
            eos_piece=eos_piece,
            pad_piece=pad_piece,
        )

    tokenizer = SPTokenizer.load(model_file)
    tokenizer.save(model_file.with_suffix(".yaml"))

    # generate_sentencepiece_tokens(model_file, tokenizer_path)


def generate_sentencepiece_tokens(model_file: PathLike, tokenizer_path: PathLike):
    sp = spm.SentencePieceProcessor()
    sp.load(str(model_file))
    token2id: Dict[str, int] = {sp.id_to_piece(i): i for i in range(sp.vocab_size())}
    with open(tokenizer_path / "tokens.txt", "w", encoding="utf-8") as f:
        for sym, i in token2id.items():
            f.write(f"{sym} {i}\n")


def make_sentencepiece_parser():
    parser = ArgumentParser()
    parser.add_argument("--cfg", action=ActionConfigFile)
    parser.add_argument(
        "--vocab-size", default=1000, type=int, help="output vocabulary size"
    )
    parser.add_argument(
        "--model-type", default="unigram", choices=["unigram", "bpe", "char", "word"]
    )
    parser.add_argument("--char-coverage", default=1.0, type=float)
    parser.add_argument("--sentence-size", default=100000000, type=int)
    parser.add_argument(
        "--user-defined-symbols",
        default=["<blk>", "<sos/eos>"],
        nargs="+",
        help="user defined symbols",
    )
    parser.add_argument("--unk-id", default=2, type=int)
    parser.add_argument("--sos-id", default=-1, type=int)
    parser.add_argument("--eos-id", default=-1, type=int)
    parser.add_argument("--pad-id", default=-1, type=int)
    parser.add_argument("--unk-piece", default="<unk>")
    parser.add_argument("--sos-piece", default="<s>")
    parser.add_argument("--eos-piece", default="</s>")
    parser.add_argument("--pad-piece", default="<pad>")
    parser.add_argument("--uppercase-text", default=True, action=ActionYesNo)

    add_common_args(parser)
    return parser


def main():
    parser = ArgumentParser(description="Train sentence piece tokenizer")
    parser.add_argument("--cfg", action=ActionConfigFile)

    subcommands = parser.add_subcommands()
    for subcommand in tokenizer_list:
        parser_func = f"make_{subcommand}_parser"
        subparser = globals()[parser_func]()
        subcommands.add_subcommand(subcommand, subparser)

    args = parser.parse_args()
    try:
        gpu_id = int(os.environ["LOCAL_RANK"])
    except:
        gpu_id = 0

    subcommand = f"train_{args.subcommand}"
    kwargs = namespace_to_dict(args)[args.subcommand]
    if gpu_id == 0:
        try:
            config_file = Path(kwargs["tokenizer_path"]) / "config.yaml"
            parser.save(args, str(config_file), format="yaml", overwrite=True)
        except Exception as err:
            logging.warning(f"failed saving {args} err={err}")

    config_logger(kwargs["verbose"])
    del kwargs["verbose"]
    del kwargs["cfg"]
    globals()[subcommand](**kwargs)


if __name__ == "__main__":
    main()
