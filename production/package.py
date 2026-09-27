"""Package the code (+ the inspected tokenizer) and publish it for the box to PULL.

    set HF_TOKEN=...           (a write token)
    python production/package.py --repo <user>/verqen-t2-code

LESSONS #34: the vast.ai SSH proxy moves 0.01 MB/s inbound; the box's own CDN download is ~800x
faster. So the package goes to a private HF dataset repo from here, and this script resolves a
*signed* CDN URL locally (huggingface.co may be blocked on the box, but us.aws.cdn.hf.co is not).
It prints the exact `export` lines to paste on the box, including the md5 the box verifies.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)

INCLUDE = ["verqen", "scripts", "configs", "tests", "production", "evals", "docs", "verqen_benchmark",
           "data/tokenizer/verqen_32k.json", "data/pretrain/v3/tokenizer_v3_32k.json", "pyproject.toml", "README.md"]
EXCLUDE_DIRS = {"__pycache__", ".pytest_cache", "_legacy_v1", "legacy_v1", "results"}


def _keep(ti):
    parts = Path(ti.name).parts
    if any(part in EXCLUDE_DIRS for part in parts):
        return None
    # a box config is the preflight stage's marker: shipping one written on THIS machine would make
    # the box skip its NCCL check and train with local-GPU throughput settings
    if ti.name.endswith(".box.yaml"):
        return None
    return ti


def build(out: str) -> str:
    leftovers = list(Path("configs").rglob("*.box.yaml"))
    if leftovers:
        print(f"[package] excluding local box configs from the package: {[str(p) for p in leftovers]}")
    with tarfile.open(out, "w:gz") as tar:
        for item in INCLUDE:
            p = Path(item)
            if not p.exists():
                print(f"[package] (missing, skipped) {item}")
                continue
            tar.add(p, arcname=item, filter=_keep)
    return out


def md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def signed_url(repo: str, filename: str, token: str) -> str:
    """Follow the resolve redirect locally and return the signed CDN location (not the hub URL)."""
    url = f"https://huggingface.co/datasets/{repo}/resolve/main/{filename}"

    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *a, **k):
            return None

    req = urllib.request.Request(url, method="HEAD", headers={"Authorization": f"Bearer {token}", "User-Agent": "curl/8"})
    try:
        urllib.request.build_opener(NoRedirect).open(req, timeout=30)
    except urllib.error.HTTPError as e:
        if e.code in (301, 302, 303, 307, 308):
            return e.headers["Location"]
        raise
    return url


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="private HF dataset repo, e.g. you/verqen-t2-code")
    ap.add_argument("--out", default="dist/verqen_t2_code.tar.gz")
    a = ap.parse_args()
    tok = Path("data/pretrain/v3/tokenizer_v3_32k.json")
    if not tok.exists():
        sys.exit("ship the inspected tokenizer: copy it to data/pretrain/v3/tokenizer_v3_32k.json first")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    build(a.out)
    digest = md5(a.out)
    size = os.path.getsize(a.out) / 1e6
    print(f"[package] {a.out}: {size:.1f} MB md5 {digest}")
    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("[package] set HF_TOKEN (write) to upload")
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    api.create_repo(a.repo, repo_type="dataset", private=True, exist_ok=True)
    for f in (a.out, "production/setup.sh"):
        api.upload_file(path_or_fileobj=f, path_in_repo=os.path.basename(f), repo_id=a.repo, repo_type="dataset")
    code_url = signed_url(a.repo, os.path.basename(a.out), token)
    setup_url = signed_url(a.repo, "setup.sh", token)
    print("\n# paste on the box (signed URLs expire; re-run package.py if they do):")
    print(f"export CODE_URL='{code_url}'\nexport CODE_MD5='{digest}'\nexport SETUP_URL='{setup_url}'")
    print("export HF_TOKEN='<your token>'  VERQEN_HF_REPO='<you>/verqen-t2-results'")
    print("curl -fsSL \"$SETUP_URL\" -o setup.sh && bash setup.sh")


if __name__ == "__main__":
    main()
