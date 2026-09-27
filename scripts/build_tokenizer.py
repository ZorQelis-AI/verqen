"""Train the VerQen 32K byte-level BPE tokenizer on FineWeb-Edu + real decision text.

    python scripts/build_tokenizer.py --docs 150000 --out data/tokenizer/verqen_32k.json
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from verqen.tokenizer import VerqenTokenizer  # noqa: E402


def fineweb_texts(pattern: str, max_docs: int):
    n = 0
    for path in sorted(glob.glob(pattern)):
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=2048, columns=["text"]):
            for t in batch.column(0).to_pylist():
                yield t
                n += 1
                if n >= max_docs:
                    return


def decision_texts():
    from verqen.data.real.sources import iter_all_train_text
    yield from iter_all_train_text()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default="data/pretrain/raw/sample/10BT/*.parquet")
    ap.add_argument("--docs", type=int, default=150000)
    ap.add_argument("--vocab-size", type=int, default=32768)
    ap.add_argument("--out", default="data/tokenizer/verqen_32k.json")
    args = ap.parse_args()

    def corpus():
        yield from fineweb_texts(args.parquet, args.docs)
        try:
            yield from decision_texts()
        except Exception as e:  # real-data module optional at tokenizer time
            print(f"[tokenizer] skipping decision text: {e}")

    tok = VerqenTokenizer.train(corpus(), vocab_size=args.vocab_size)
    tok.save(args.out)
    print(f"[tokenizer] saved {args.out} vocab={tok.vocab_size} specials={tok.special_token_ids}")


if __name__ == "__main__":
    main()
