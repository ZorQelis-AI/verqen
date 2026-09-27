"""python -m verqen.inference --checkpoint runs/verqen-t1-v2/release --input request.json"""

import argparse
import json

from verqen.inference.api import Verqen

ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint", required=True)
ap.add_argument("--input", required=True)
ap.add_argument("--device", default="auto")
args = ap.parse_args()
with open(args.input, encoding="utf-8") as f:
    req = json.load(f)
print(json.dumps(Verqen.from_pretrained(args.checkpoint, args.device).decide_request(req), indent=2))
