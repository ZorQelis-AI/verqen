"""VerQen-T1 inference engine.

* ``decide_many`` batches many independent requests into packed rows (throughput path).
* ``encode_state`` + ``decide_on_state`` cache the state's key/value tensors once and
  answer any number of later questions against it (agent loops, many labels per document).
  This is exact, not an approximation: the state never attends to questions, and questions
  use branch-local positions, so the cached state is identical to a full forward pass.
* Outputs are calibrated (temperature), carry ``p_correct`` from the selective head,
  a conformal ``prediction_set``, and ``should_escalate`` from the risk-controlled threshold.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import torch

from verqen.calibration.fit import TYPES, apply_calibration
from verqen.data.collator import collate_rows, pack_rows, to_device
from verqen.data.packing import SequencePacker
from verqen.model.attention_mask import SEG_BOS, SEG_STATE, build_attention_mask
from verqen.model.verqen import VerqenModel
from verqen.tokenizer import VerqenTokenizer
from verqen.utils.checkpoint import load_calibration, load_model, resolve_checkpoint


@dataclass
class StateHandle:
    kvs: list
    n_ctx: int
    seg: torch.Tensor
    base_pos: int


class VerqenInferenceEngine:
    def __init__(self, model: VerqenModel, tokenizer: VerqenTokenizer, device: torch.device | str = "cpu",
                 max_seq_len: int = 512, calibration: dict | None = None, risk: str | None = None,
                 dtype: torch.dtype | None = None, cast_weights: bool = False, fast: bool = False,
                 risk_mode: str = "marginal", risk_domain: str | None = None):
        """risk_mode: "marginal" (default) bounds the error rate of the pooled calibration mixture;
        "worst_group" uses calibration["risk_control_worst_group"], which bounds it for every
        calibration source separately (docs/audit/ROOT_CAUSES.md RC-4). It escalates far more,
        and it is the setting for high-stakes use. With ``risk_domain`` naming a calibration
        source, that source's own certified threshold is used (group-conditional control); if the
        source could not be certified at the chosen risk, every decision escalates."""
        self.device = torch.device(device) if isinstance(device, str) else device
        self.model = model.to(self.device).eval()
        self.dtype = dtype or (torch.bfloat16 if self.device.type == "cuda" else torch.float32)
        self.fast = fast and self.device.type == "cuda"
        if (cast_weights or self.fast) and self.device.type == "cuda":
            # bf16 backbone; the decision head stays fp32 (it is tiny and calibration-sensitive)
            self.model.transformer.to(self.dtype)
            self.model.transformer.rope.float()
        if self.fast:
            import torch._inductor.config as inductor_config
            inductor_config.triton.cudagraph_trees = False  # plain per-shape graphs: no tree bookkeeping on bucket switches
            import torch._dynamo.config as dynamo_config
            # Every shape bucket is a specialization of the same code object; the default limit
            # of 8 silently drops later buckets back to eager (measured: 36 ms instead of 5 ms).
            dynamo_config.recompile_limit = 256
            dynamo_config.accumulated_recompile_limit = 4096
            # torch.compile + CUDA graphs over fixed shape buckets: removes per-kernel launch
            # overhead, which dominates small requests (31.7 ms eager -> ~0.9 ms backbone).
            self._graphs = {}
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.packer = SequencePacker(tokenizer, max_seq_len)
        self.calibration = calibration
        self.risk = risk or (calibration or {}).get("default_risk", "0.1")
        if risk_mode not in ("marginal", "worst_group"):
            raise ValueError(f"risk_mode must be 'marginal' or 'worst_group', got {risk_mode!r}")
        if risk_mode == "worst_group" and "risk_control_worst_group" not in (calibration or {}):
            raise ValueError("risk_mode='worst_group' needs a calibration with risk_control_worst_group "
                             "(scripts/recalibrate_worst_group.py)")
        self.risk_mode = risk_mode
        if risk_domain is not None:
            known = (calibration or {}).get("risk_control_worst_group", {}).get("groups", {})
            if risk_mode != "worst_group" or risk_domain not in known:
                raise ValueError(f"risk_domain needs risk_mode='worst_group' and one of {sorted(known)}")
        self.risk_domain = risk_domain

    @classmethod
    def from_checkpoint(cls, path: str, device: str = "auto", risk: str | None = None,
                        risk_mode: str = "marginal", fast: bool = False,
                        risk_domain: str | None = None) -> "VerqenInferenceEngine":
        dev = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device))
        path = resolve_checkpoint(path)  # a local directory, or a Hub id such as "zorqelis-ai/VerQen"
        model = load_model(path, dev)
        tok_path = os.path.join(path, "tokenizer.json")
        tok = VerqenTokenizer.from_pretrained(tok_path)
        return cls(model, tok, dev, model.config.max_seq_len, load_calibration(path), risk, fast=fast, risk_mode=risk_mode,
                   risk_domain=risk_domain)

    # ── helpers ───────────────────────────────────────────────────────────
    @property
    def threshold(self) -> float:
        if not self.calibration:
            return 0.0
        key = "risk_control_worst_group" if self.risk_mode == "worst_group" else "risk_control"
        rc = self.calibration.get(key) or self.calibration.get("worst_group" if key == "risk_control_worst_group" else "marginal_risk")
        if not rc or not isinstance(rc, dict):
            return 0.0
        by_risk = rc.get("by_risk") or rc
        if not isinstance(by_risk, dict):
            return 0.0
        entry = by_risk.get(str(self.risk), {"threshold": 0.0})
        if not isinstance(entry, dict):
            return 0.0
        if self.risk_domain is not None and "per_group_threshold" in entry:
            # groups below min_group were never certified: escalate everything for them
            return float(entry["per_group_threshold"].get(self.risk_domain, 1.01))
        return float(entry.get("threshold", 0.0))

    @property
    def escalation_score(self) -> str:
        if self.risk_mode == "worst_group" and self.calibration:
            return self.calibration["risk_control_worst_group"].get("score", "p_correct")
        return "p_correct"

    def _results(self, out, b, questions_per_example, raw_questions) -> list[list[dict[str, Any]]]:
        probs, p_correct = apply_calibration(out["logp"].float(), out["opt_mask"], b["qtype"], out["selective_logit"].float(),
                                             self.calibration)
        qhat = (self.calibration or {}).get("conformal", {}).get("qhat")
        probs, p_correct = probs.cpu(), p_correct.cpu()
        results, qi = [], 0
        for n_q, qs in zip(questions_per_example, raw_questions):
            ex_res = []
            for q in qs[:n_q]:
                qtype = q["type"]
                opts = q["options"]
                p = probs[qi, : len(opts)].tolist()
                best = max(range(len(opts)), key=p.__getitem__)
                res: dict[str, Any] = {
                    "type": qtype,
                    "selected": opts[best],
                    "probabilities": {o: round(v, 6) for o, v in zip(opts, p)},
                    "p_correct": round(float(p_correct[qi]), 4),
                    "margin": round(sorted(p)[-1] - (sorted(p)[-2] if len(p) > 1 else 0.0), 4),
                }
                if qtype == "noul":
                    res["value"] = p[1] >= 0.5
                    res["p_true"] = round(p[1], 6)
                if qtype == "score":
                    res["expected_score"] = round(sum(i * v for i, v in enumerate(p)), 4)
                if qhat is not None:
                    res["prediction_set"] = [o for o, v in zip(opts, p) if v >= 1 - qhat]
                score = max(p) if self.escalation_score == "max_prob" else float(p_correct[qi])
                res["should_escalate"] = bool(score < self.threshold)
                res["confidence"] = res["p_correct"]
                cost = q.get("cost")
                if cost:  # Fix D: cost-aware escalation (opt-in per question; RC-5)
                    # act iff the expected loss of acting does not exceed the cost of escalating
                    exp_loss = (1.0 - float(p_correct[qi])) * float(cost["error_cost"])
                    res["expected_loss_if_acting"] = round(exp_loss, 4)
                    if exp_loss > float(cost["escalation_cost"]) and not res["should_escalate"]:
                        res["should_escalate"] = True
                        res["escalation_reason"] = "expected_loss"
                    elif res["should_escalate"]:
                        res["escalation_reason"] = "risk_threshold"
                ex_res.append(res)
                qi += 1
            results.append(ex_res)
        return results

    @staticmethod
    def _normalize_questions(questions):
        out = []
        for q in questions:
            qtype = str(q.get("type", q.get("question_type", "choice"))).lower()
            opts = ["no", "yes"] if qtype == "noul" else list(q.get("options", []))
            nq = {"type": qtype, "text": q.get("text", q.get("question", "")), "options": opts,
                  "target": [1.0 / max(1, len(opts))] * len(opts)}
            if q.get("cost"):
                nq["cost"] = {"error_cost": float(q["cost"]["error_cost"]), "escalation_cost": float(q["cost"]["escalation_cost"])}
            out.append(nq)
        return out

    # ── throughput path ──────────────────────────────────────────────────
    # Requests per forward pass, scaled so rows x length stays bounded: 2048 requests packed into one
    # pass at 1024 tokens asked for a 1 GiB-per-layer attention buffer and OOM'd a 32 GB GPU (25 Sep 2026).
    MAX_REQUESTS_PER_PASS_AT_512 = 512

    @torch.no_grad()
    def decide_many(self, requests: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
        """requests: [{"state": str, "questions": [{"type", "text", "options"}]}] -> results per request.
        Large calls are split into bounded chunks; results come back in request order."""
        chunk = max(16, self.MAX_REQUESTS_PER_PASS_AT_512 * 512 * 512 // max(512, self.max_seq_len) ** 2)
        if len(requests) > chunk:
            out: list[list[dict[str, Any]]] = []
            for i in range(0, len(requests), chunk):
                out += self._decide_many(requests[i:i + chunk])
            return out
        return self._decide_many(requests)

    @torch.no_grad()
    def _decide_many(self, requests: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
        norm = [self._normalize_questions(r["questions"]) for r in requests]
        # Bundle a request's questions into one sequence (state encoded once) unless that would
        # truncate the state; then every question gets its own sequence with the full state, so
        # an answer never depends on which other questions were asked alongside it.
        jobs = []  # (request index, question indices)
        for i, q in enumerate(norm):
            jobs.append((i, list(range(len(q)))))
        exs = self.packer.pack_many([{"state": requests[i].get("state", ""), "questions": [norm[i][k] for k in ks]} for i, ks in jobs])
        split_jobs, split_exs = [], []
        for (i, ks), e in zip(jobs, exs):
            if len(ks) > 1:
                n_state = sum(1 for s in e.seg if s == SEG_STATE)
                full = len(self.tokenizer.encode(requests[i].get("state", ""))) + 2
                if n_state < min(full, self.max_seq_len):
                    split_jobs += [(i, [k]) for k in ks]
                    continue
            split_jobs.append((i, ks))
            split_exs.append(e)
        if len(split_jobs) != len(split_exs):
            jobs = split_jobs
            exs = self.packer.pack_many([{"state": requests[i].get("state", ""), "questions": [norm[i][k] for k in ks]} for i, ks in jobs])
        for j, e in enumerate(exs):
            e.example_id = str(j)
        packed, _ = pack_rows(exs, self.max_seq_len, None)
        order = [e for row in packed for e in row]
        longest = max(sum(len(e) for e in row) for row in packed)
        if self.fast:
            # buckets up to the model's context (T2: 2048), not a fixed 512 that crashed longer inputs
            row_len = next(b for b in (64, 128, 256, 512, 1024, 2048, 4096) if b >= min(longest, self.max_seq_len))
            n_rows = 1 << (len(packed) - 1).bit_length()
            b = to_device(collate_rows(packed, row_len, self.tokenizer.pad_token_id, n_rows), self.device)
        else:
            # Pad rows only to the longest row (multiple of 16), not to max_seq_len.
            row_len = min(self.max_seq_len, -(-longest // 16) * 16)
            b = to_device(collate_rows(packed, row_len, self.tokenizer.pad_token_id), self.device)
        with torch.autocast(self.device.type, dtype=self.dtype, enabled=self.device.type == "cuda"):
            if self.fast:
                h = self._encoder(tuple(b["input_ids"].shape))(b["input_ids"], b["seg"], b["pos"], b["doc"], b["qid"], b["oid"])
                out = self.model.decision_readout(h.clone(), b, head=self._fast_head)
            else:
                out = self.model.forward_decisions(b)
        res_in_order = self._results(out, b, [len(e.questions) for e in order],
                                     [[norm[jobs[int(e.example_id)][0]][k] for k in jobs[int(e.example_id)][1]] for e in order])
        results = [[None] * len(q) for q in norm]
        for e, r in zip(order, res_in_order):
            i, ks = jobs[int(e.example_id)]
            for k, res in zip(ks, r):
                results[i][k] = res
        return [[x for x in r if x is not None] for r in results]

    def _fast_head(self, d, opts, opt_mask, qtype):
        """Run the decision head through CUDA graphs by padding (questions, options) to buckets."""
        N, O = opt_mask.shape
        Np = 1 << (N - 1).bit_length()
        # wide option sets (Banking77: 77 labels) need buckets past 64; the set-attention head has no limit
        Op = next((x for x in (4, 8, 16, 64, 128, 256) if x >= O), O)
        # float32 inputs always: the head runs in fp32, and a dtype change would fail the compile guard
        dp = d.new_zeros(Np, d.size(-1), dtype=torch.float32); dp[:N] = d
        op = opts.new_zeros(Np, Op, opts.size(-1), dtype=torch.float32); op[:N, :O] = opts
        mp = torch.zeros(Np, Op, dtype=torch.bool, device=d.device); mp[:N, :O] = opt_mask
        mp[N:, :2] = True  # padded questions get two dummy options (keeps softmax finite)
        qp = torch.zeros(Np, dtype=torch.long, device=d.device); qp[:N] = qtype
        key = ("head", Np, Op)
        fn = self._graphs.get(key)
        if fn is None:
            head = self.model.decision_head
            fn = torch.compile(lambda a, b_, c, e: {k: v.clone() for k, v in head(a, b_, c, e).items()},
                               mode="reduce-overhead", dynamic=False)
            self._graphs[key] = fn
        # Always call with autocast off (the head is fp32): the compiled graph guards on
        # autocast state, so a caller-dependent state would force a recompile per call.
        with torch.autocast(device_type=self.device.type, enabled=False):
            out = fn(dp, op, mp, qp)
        return {k: (v[:N, :O] if v.dim() == 2 else v[:N]) for k, v in out.items()}

    def _encoder(self, shape):
        fn = self._graphs.get(shape)
        if fn is None:
            m = self.model

            def enc(ids, seg, pos, doc, qid, oid):
                return m.transformer(ids, seg, pos, build_attention_mask(seg, doc, qid, oid))
            fn = torch.compile(enc, mode="reduce-overhead" if shape[0] <= 16 else "default", dynamic=False)
            self._graphs[shape] = fn
        return fn

    def warmup(self, rows=(1, 2, 4, 8), lengths=(64, 128, 256, 512)) -> None:
        """Compile the interactive (small-batch) shape buckets up front. Large batches compile
        lazily on first use; pre-capturing graphs for every big shape bloats GPU memory pools
        and made *every* call slower (41 ms vs 5 ms) in testing."""
        for L in lengths:
            for R in rows:
                z = torch.zeros(R, L, dtype=torch.long, device=self.device)
                with torch.no_grad(), torch.autocast(self.device.type, dtype=self.dtype):
                    for _ in range(3):
                        self._encoder((R, L))(z, z + 2, z, z, z - 1, z - 1)
        D = self.model.config.hidden_size
        for Np in (1, 2, 4, 8):
            for Op in (4, 8, 16):
                d = torch.zeros(Np, D, device=self.device)
                m = torch.zeros(Np, Op, dtype=torch.bool, device=self.device)
                m[:, :2] = True
                with torch.no_grad():
                    for _ in range(3):
                        self._fast_head(d, torch.zeros(Np, Op, D, device=self.device), m,
                                        torch.zeros(Np, dtype=torch.long, device=self.device))

    def decide_multi(self, state: str, questions: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return self.decide_many([{"state": state, "questions": questions}])[0]

    def decide(self, state: str, question: str, options: list[str] | None = None, question_type: str = "choice",
               cost: dict[str, float] | None = None) -> dict:
        """cost (optional): {"error_cost": c_e, "escalation_cost": c_x}. When given, the decision is
        also escalated whenever (1 - p_correct) * c_e > c_x."""
        q = {"type": question_type, "text": question, "options": options or []}
        if cost:
            q["cost"] = cost
        return self.decide_multi(state, [q])[0]

    def decide_request(self, request: dict[str, Any]) -> dict[str, Any]:
        decisions = self.decide_multi(request.get("state", request.get("state_text", "")), request.get("questions", []))
        return {"id": request.get("id", "req-001"), "decisions": decisions,
                "has_escalation": any(d["should_escalate"] for d in decisions)}

    # ── cached-state path ────────────────────────────────────────────────
    @torch.no_grad()
    def encode_state(self, state: str) -> StateHandle:
        ex = self.packer.pack({"state": state, "questions": [{"type": "noul", "text": "", "target": [0.5, 0.5]}]})
        n_ctx = next(i for i, s in enumerate(ex.seg) if s not in (SEG_BOS, SEG_STATE))
        ids = torch.tensor([ex.ids[:n_ctx]], device=self.device)
        seg = torch.tensor([ex.seg[:n_ctx]], device=self.device)
        pos = torch.tensor([ex.pos[:n_ctx]], device=self.device)
        with torch.autocast(self.device.type, dtype=self.dtype, enabled=self.device.type == "cuda"):
            _, kvs = self.model.transformer(ids, seg, pos, None, return_kvs=True)
        return StateHandle(kvs, n_ctx, seg, ex.pos[n_ctx])

    @torch.no_grad()
    def decide_on_state(self, handle: StateHandle, questions: list[dict[str, Any]]) -> list[dict[str, Any]]:
        norm = self._normalize_questions(questions)
        ex = self.packer.pack({"state": "", "questions": norm})
        # Re-base branch positions/indices onto the cached context.
        first = next(i for i, s in enumerate(ex.seg) if s not in (SEG_BOS, SEG_STATE))
        shift_pos = handle.base_pos - ex.pos[first]
        ids = torch.tensor([ex.ids[first:]], device=self.device)
        seg = torch.tensor([ex.seg[first:]], device=self.device)
        pos = torch.tensor([p + shift_pos for p in ex.pos[first:]], device=self.device)[None]
        qid = torch.tensor([ex.qid[first:]], device=self.device)
        oid = torch.tensor([ex.oid[first:]], device=self.device)
        L = ids.size(1)
        doc = torch.zeros_like(ids)
        k_seg = torch.cat([handle.seg, seg], 1)
        k_doc = torch.zeros_like(k_seg)
        k_qid = torch.cat([torch.full_like(handle.seg, -1), qid], 1)
        k_oid = torch.cat([torch.full_like(handle.seg, -1), oid], 1)
        mask = build_attention_mask(seg, doc, qid, oid, k_seg, k_doc, k_qid, k_oid)
        with torch.autocast(self.device.type, dtype=self.dtype, enabled=self.device.type == "cuda"):
            h = self.model.transformer(ids, seg, pos, mask, past_kvs=handle.kvs)
            b = collate_rows([[ex]], len(ex), self.tokenizer.pad_token_id)
            b = to_device(b, self.device)
            # hidden for the full layout: pad the context part (never read by the readout)
            full = torch.zeros(1, len(ex), h.size(-1), device=self.device, dtype=h.dtype)
            full[:, first:] = h
            out = self.model.decision_readout(full, b)
        return self._results(out, b, [len(norm)], [norm])[0]
