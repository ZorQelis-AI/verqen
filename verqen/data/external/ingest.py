"""External benchmark dataset ingestion and normalization for VerQen-T1.

Converts public foundation model benchmarks:
- AI2 ARC (Easy & Challenge)
- CAIS MMLU (All 57 subjects)
- TIGER-Lab MMLU-Pro (Advanced 10-choice college reasoning)
- AllenAI OpenBookQA (Science & common sense)
- AllenAI SciQ (Science questions with explanation support)
- Tau CommonsenseQA (5-choice heuristic reasoning)
- DeepMind AQUA-RAT (Algebraic word problems)
- OpenLifeScienceAI MedMCQA (Clinical / medical MCQs)

into VerQen's validated machine-native decision schema:
- state: Domain/context prefix + question / support context
- question: Exact question prompt
- options: Standardized candidate options (2 to 64)
- target: Calibrated probabilities (sum=1.0) and ground truth label
"""

import random
from typing import Any, Dict, List, Optional
from datasets import load_dataset


LABEL_TO_IDX = {
    "A": 0, "B": 1, "C": 2, "D": 3, "E": 4,
    "F": 5, "G": 6, "H": 7, "I": 8, "J": 9,
    "1": 0, "2": 1, "3": 2, "4": 3, "5": 4,
    "0": 0
}

# Domain mapping from MMLU 57 subjects into VerQen's 24 Canonical Domains
MMLU_SUBJECT_MAP = {
    "abstract_algebra": "03_mathematics_quantitative",
    "college_mathematics": "03_mathematics_quantitative",
    "elementary_mathematics": "03_mathematics_quantitative",
    "high_school_mathematics": "03_mathematics_quantitative",
    "high_school_statistics": "16_data_analytics",
    "formal_logic": "04_formal_logic",
    "logical_fallacies": "04_formal_logic",
    "college_computer_science": "14_programming_algorithms",
    "computer_security": "19_cybersecurity_safety",
    "high_school_computer_science": "14_programming_algorithms",
    "machine_learning": "14_programming_algorithms",
    "astronomy": "06_physics_world",
    "college_physics": "06_physics_world",
    "conceptual_physics": "06_physics_world",
    "high_school_physics": "06_physics_world",
    "college_chemistry": "07_chemistry_materials",
    "high_school_chemistry": "07_chemistry_materials",
    "anatomy": "08_biology_life_sciences",
    "college_biology": "08_biology_life_sciences",
    "high_school_biology": "08_biology_life_sciences",
    "clinical_knowledge": "09_health_wellbeing",
    "college_medicine": "09_health_wellbeing",
    "human_aging": "09_health_wellbeing",
    "medical_genetics": "09_health_wellbeing",
    "nutrition": "09_health_wellbeing",
    "professional_medicine": "09_health_wellbeing",
    "virology": "09_health_wellbeing",
    "econometrics": "10_economics_finance",
    "high_school_macroeconomics": "10_economics_finance",
    "high_school_microeconomics": "10_economics_finance",
    "business_ethics": "21_planning_strategy",
    "management": "21_planning_strategy",
    "marketing": "21_planning_strategy",
    "international_law": "11_law_rules_compliance",
    "jurisprudence": "11_law_rules_compliance",
    "professional_law": "11_law_rules_compliance",
    "moral_disputes": "11_law_rules_compliance",
    "moral_scenarios": "05_common_sense",
    "high_school_geography": "13_geography_navigation",
    "high_school_government_and_politics": "11_law_rules_compliance",
    "high_school_psychology": "20_social_interpersonal",
    "human_sexuality": "20_social_interpersonal",
    "professional_psychology": "20_social_interpersonal",
    "public_relations": "20_social_interpersonal",
    "security_studies": "19_cybersecurity_safety",
    "sociology": "20_social_interpersonal",
    "high_school_european_history": "02_general_knowledge",
    "high_school_us_history": "02_general_knowledge",
    "high_school_world_history": "02_general_knowledge",
    "philosophy": "04_formal_logic",
    "prehistory": "02_general_knowledge",
    "world_religions": "02_general_knowledge",
    "global_facts": "02_general_knowledge",
    "miscellaneous": "02_general_knowledge",
}

# Domain mapping from MMLU-Pro categories
MMLU_PRO_CATEGORY_MAP = {
    "math": "03_mathematics_quantitative",
    "physics": "06_physics_world",
    "chemistry": "07_chemistry_materials",
    "biology": "08_biology_life_sciences",
    "computer science": "14_programming_algorithms",
    "health": "09_health_wellbeing",
    "economics": "10_economics_finance",
    "law": "11_law_rules_compliance",
    "business": "21_planning_strategy",
    "psychology": "20_social_interpersonal",
    "philosophy": "04_formal_logic",
    "history": "02_general_knowledge",
    "other": "02_general_knowledge",
}


def normalize_record(
    example_id: str,
    domain: str,
    task_family: str,
    state: str,
    question: str,
    options: List[str],
    target_label: str,
    source: str,
    difficulty: str = "d3_multi_hop",
    split: str = "train",
) -> Optional[Dict[str, Any]]:
    """Convert raw external fields into a valid VerQen DecisionExample dictionary."""
    if not question or not options or not target_label:
        return None
    # Strip options
    clean_opts = [str(opt).strip() for opt in options if str(opt).strip()]
    if len(clean_opts) < 2 or len(clean_opts) > 64:
        return None
    # Deduplicate while preserving order; if duplicates exist, reject to avoid ambiguity
    if len(clean_opts) != len(set(clean_opts)):
        return None
    if target_label not in clean_opts:
        return None

    # Construct target distribution (near one-hot probability for ground truth)
    p_true = 0.94
    distractors = [opt for opt in clean_opts if opt != target_label]
    p_dist = round((1.0 - p_true) / len(distractors), 5)
    dist = {target_label: p_true}
    for opt in distractors:
        dist[opt] = p_dist
    # Normalize to ensure sum is strictly 1.0
    total_p = sum(dist.values())
    dist = {k: round(v / total_p, 5) for k, v in dist.items()}

    return {
        "id": example_id,
        "domain": domain,
        "task_family": task_family,
        "experience_type": "hybrid_external",
        "difficulty": difficulty,
        "question_type": "CHOICE",
        "state": state.strip(),
        "question": question.strip(),
        "options": clean_opts,
        "target": {
            "probabilities": dist,
            "label": target_label,
        },
        "ground_truth_method": "expert_ground_truth",
        "verification_status": "passed",
        "evidence_strength": 3,
        "is_adversarial": False,
        "adversarial_type": None,
        "counterfactual_meta": None,
        "trajectory": None,
        "sub_questions": None,
        "provenance": {
            "generator_version": "1.0.0",
            "schema_version": "1.0.0",
            "seed": 42,
            "worker_seed": None,
            "template_id": source,
            "verifier_type": "benchmark_ground_truth",
            "verifier_version": "1.0.0",
            "teacher_metadata": None,
            "timestamp": None,
        },
        "metadata": {
            "source_benchmark": source,
            "split": split,
        },
    }


def ingest_arc(subset: str = "ARC-Easy", split: str = "train", limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Ingest AI2 ARC dataset (ARC-Easy or ARC-Challenge)."""
    print(f"[Ingest] Loading allenai/ai2_arc ({subset}, split={split})...")
    ds = load_dataset("allenai/ai2_arc", subset, split=split)
    records = []
    domain = "06_physics_world"

    for i, ex in enumerate(ds):
        if limit and i >= limit:
            break
        q = ex.get("question", "")
        choices = ex.get("choices", {}).get("text", [])
        labels = ex.get("choices", {}).get("label", [])
        ans_key = ex.get("answerKey", "")

        if not q or not choices or not ans_key:
            continue

        if ans_key in labels:
            idx = labels.index(ans_key)
        elif ans_key in LABEL_TO_IDX:
            idx = LABEL_TO_IDX[ans_key]
        else:
            continue

        if idx >= len(choices):
            continue

        target = choices[idx].strip()
        state = f"Scientific foundation reasoning ({subset}): {q}"
        rec = normalize_record(
            example_id=f"ext_{subset.lower().replace('-', '_')}_{split}_{i:06d}",
            domain=domain,
            task_family="science_reasoning",
            state=state,
            question=q,
            options=choices,
            target_label=target,
            source=f"ai2_arc_{subset}",
            difficulty="d3_multi_hop",
            split=split,
        )
        if rec:
            records.append(rec)

    print(f"[Ingest] Ingested {len(records)} valid examples from {subset} ({split}).")
    return records


def ingest_openbookqa(split: str = "train", limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Ingest AllenAI OpenBookQA."""
    print(f"[Ingest] Loading allenai/openbookqa (split={split})...")
    ds = load_dataset("allenai/openbookqa", "main", split=split)
    records = []
    domain = "05_common_sense"

    for i, ex in enumerate(ds):
        if limit and i >= limit:
            break
        q = ex.get("question_stem", "")
        choices = ex.get("choices", {}).get("text", [])
        labels = ex.get("choices", {}).get("label", [])
        ans_key = ex.get("answerKey", "")

        if not q or not choices or not ans_key:
            continue

        if ans_key in labels:
            idx = labels.index(ans_key)
        elif ans_key in LABEL_TO_IDX:
            idx = LABEL_TO_IDX[ans_key]
        else:
            continue

        if idx >= len(choices):
            continue

        target = choices[idx].strip()
        state = f"Open-book factual reasoning: {q}"
        rec = normalize_record(
            example_id=f"ext_openbookqa_{split}_{i:06d}",
            domain=domain,
            task_family="openbookqa",
            state=state,
            question=q,
            options=choices,
            target_label=target,
            source="allenai_openbookqa",
            difficulty="d3_multi_hop",
            split=split,
        )
        if rec:
            records.append(rec)

    print(f"[Ingest] Ingested {len(records)} valid examples from OpenBookQA ({split}).")
    return records


def ingest_mmlu_all(splits: Optional[List[str]] = None, limit_per_split: Optional[int] = None) -> List[Dict[str, Any]]:
    """Ingest all 57 subjects of CAIS MMLU using the 'all' configuration."""
    if splits is None:
        splits = ["test", "validation", "dev"]
    records = []

    for split in splits:
        try:
            print(f"[Ingest] Loading cais/mmlu (all, split={split})...")
            ds = load_dataset("cais/mmlu", "all", split=split)
            count = 0
            for i, ex in enumerate(ds):
                if limit_per_split and count >= limit_per_split:
                    break
                q = ex.get("question", "")
                subject = ex.get("subject", "miscellaneous")
                choices = [str(c).strip() for c in ex.get("choices", [])]
                ans_idx = ex.get("answer")

                if ans_idx is None or not (0 <= ans_idx < len(choices)):
                    continue

                target = choices[ans_idx]
                domain = MMLU_SUBJECT_MAP.get(subject, "02_general_knowledge")
                state = f"Academic examination ({subject.replace('_', ' ')}): {q}"
                rec = normalize_record(
                    example_id=f"ext_mmlu_{split}_{i:06d}",
                    domain=domain,
                    task_family=f"mmlu_{subject}",
                    state=state,
                    question=q,
                    options=choices,
                    target_label=target,
                    source=f"cais_mmlu_{subject}",
                    difficulty="d4_constraint",
                    split=split,
                )
                if rec:
                    records.append(rec)
                    count += 1
            print(f"[Ingest] Ingested {count} examples from MMLU ({split}).")
        except Exception as e:
            print(f"[Ingest Warning] MMLU ({split}) skipped: {e}")

    print(f"[Ingest] Total MMLU Ingested: {len(records)}")
    return records


# Backward compatibility alias
ingest_mmlu = ingest_mmlu_all


def ingest_mmlu_pro(splits: Optional[List[str]] = None, limit_per_split: Optional[int] = None) -> List[Dict[str, Any]]:
    """Ingest TIGER-Lab/MMLU-Pro (Advanced 10-option multi-choice college reasoning)."""
    if splits is None:
        splits = ["test", "validation"]
    records = []

    for split in splits:
        try:
            print(f"[Ingest] Loading TIGER-Lab/MMLU-Pro (split={split})...")
            ds = load_dataset("TIGER-Lab/MMLU-Pro", split=split)
            count = 0
            for i, ex in enumerate(ds):
                if limit_per_split and count >= limit_per_split:
                    break
                q = ex.get("question", "")
                raw_opts = ex.get("options", [])
                ans_idx = ex.get("answer_index")
                category = ex.get("category", "other")

                choices = [str(c).strip() for c in raw_opts if str(c).strip()]
                if ans_idx is None or not (0 <= ans_idx < len(choices)):
                    continue

                target = choices[ans_idx]
                domain = MMLU_PRO_CATEGORY_MAP.get(category.lower(), "02_general_knowledge")
                state = f"Advanced college reasoning ({category}): {q}"
                rec = normalize_record(
                    example_id=f"ext_mmlupro_{split}_{i:06d}",
                    domain=domain,
                    task_family=f"mmlu_pro_{category}",
                    state=state,
                    question=q,
                    options=choices,
                    target_label=target,
                    source=f"mmlu_pro_{category}",
                    difficulty="d5_counterfactual",
                    split=split,
                )
                if rec:
                    records.append(rec)
                    count += 1
            print(f"[Ingest] Ingested {count} examples from MMLU-Pro ({split}).")
        except Exception as e:
            print(f"[Ingest Warning] MMLU-Pro ({split}) skipped: {e}")

    print(f"[Ingest] Total MMLU-Pro Ingested: {len(records)}")
    return records


def ingest_sciq(splits: Optional[List[str]] = None, limit_per_split: Optional[int] = None) -> List[Dict[str, Any]]:
    """Ingest AllenAI SciQ (Science questions with explanation paragraphs)."""
    if splits is None:
        splits = ["train", "validation", "test"]
    records = []

    for split in splits:
        try:
            print(f"[Ingest] Loading allenai/sciq (split={split})...")
            ds = load_dataset("allenai/sciq", split=split)
            count = 0
            for i, ex in enumerate(ds):
                if limit_per_split and count >= limit_per_split:
                    break
                q = ex.get("question", "")
                corr = ex.get("correct_answer", "").strip()
                d1 = ex.get("distractor1", "").strip()
                d2 = ex.get("distractor2", "").strip()
                d3 = ex.get("distractor3", "").strip()
                support = ex.get("support", "").strip()

                if not q or not corr or not d1 or not d2 or not d3:
                    continue

                choices = [corr, d1, d2, d3]
                random.seed(i)
                random.shuffle(choices)

                # Use support paragraph as state if present
                state = f"Scientific context: {support}\nQuestion: {q}" if support else f"Scientific inquiry: {q}"
                rec = normalize_record(
                    example_id=f"ext_sciq_{split}_{i:06d}",
                    domain="06_physics_world",
                    task_family="sciq_examination",
                    state=state,
                    question=q,
                    options=choices,
                    target_label=corr,
                    source="allenai_sciq",
                    difficulty="d3_multi_hop",
                    split=split,
                )
                if rec:
                    records.append(rec)
                    count += 1
            print(f"[Ingest] Ingested {count} examples from SciQ ({split}).")
        except Exception as e:
            print(f"[Ingest Warning] SciQ ({split}) skipped: {e}")

    print(f"[Ingest] Total SciQ Ingested: {len(records)}")
    return records


def ingest_commonsense_qa(splits: Optional[List[str]] = None, limit_per_split: Optional[int] = None) -> List[Dict[str, Any]]:
    """Ingest Tau CommonsenseQA (5-choice heuristic questions)."""
    if splits is None:
        splits = ["train", "validation"]
    records = []

    for split in splits:
        try:
            print(f"[Ingest] Loading tau/commonsense_qa (split={split})...")
            ds = load_dataset("tau/commonsense_qa", split=split)
            count = 0
            for i, ex in enumerate(ds):
                if limit_per_split and count >= limit_per_split:
                    break
                q = ex.get("question", "")
                choices_raw = ex.get("choices", {})
                labels = choices_raw.get("label", [])
                texts = [str(t).strip() for t in choices_raw.get("text", [])]
                ans_key = ex.get("answerKey", "")

                if not q or not texts or not ans_key or ans_key not in labels:
                    continue

                idx = labels.index(ans_key)
                if idx >= len(texts):
                    continue

                target = texts[idx]
                state = f"Common-sense heuristic reasoning: {q}"
                rec = normalize_record(
                    example_id=f"ext_csqa_{split}_{i:06d}",
                    domain="05_common_sense",
                    task_family="commonsense_qa",
                    state=state,
                    question=q,
                    options=texts,
                    target_label=target,
                    source="tau_commonsense_qa",
                    difficulty="d3_multi_hop",
                    split=split,
                )
                if rec:
                    records.append(rec)
                    count += 1
            print(f"[Ingest] Ingested {count} examples from CommonsenseQA ({split}).")
        except Exception as e:
            print(f"[Ingest Warning] CommonsenseQA ({split}) skipped: {e}")

    print(f"[Ingest] Total CommonsenseQA Ingested: {len(records)}")
    return records


def ingest_aqua_rat(split: str = "train", limit: int = 15000) -> List[Dict[str, Any]]:
    """Ingest DeepMind AQUA-RAT (Algebraic word problems with rationales)."""
    print(f"[Ingest] Loading deepmind/aqua_rat (raw, split={split}, limit={limit})...")
    records = []
    try:
        ds = load_dataset("deepmind/aqua_rat", "raw", split=split)
        count = 0
        for i, ex in enumerate(ds):
            if limit and count >= limit:
                break
            q = ex.get("question", "")
            raw_opts = ex.get("options", [])
            corr_letter = ex.get("correct", "").strip().upper()
            rationale = ex.get("rationale", "")

            if not q or not raw_opts or not corr_letter:
                continue

            parsed_opts = []
            for o in raw_opts:
                o_str = str(o).strip()
                if len(o_str) > 2 and o_str[1] in [")", ":", "."]:
                    parsed_opts.append(o_str[2:].strip())
                else:
                    parsed_opts.append(o_str)

            if corr_letter not in LABEL_TO_IDX:
                continue
            corr_idx = LABEL_TO_IDX[corr_letter]
            if corr_idx >= len(parsed_opts):
                continue

            target = parsed_opts[corr_idx]
            state = f"Quantitative problem solving: {q}"
            rec = normalize_record(
                example_id=f"ext_aqua_{split}_{i:06d}",
                domain="03_mathematics_quantitative",
                task_family="aqua_algebra",
                state=state,
                question=q,
                options=parsed_opts,
                target_label=target,
                source="deepmind_aqua_rat",
                difficulty="d4_constraint",
                split=split,
            )
            if rec:
                records.append(rec)
                count += 1
        print(f"[Ingest] Ingested {count} examples from AQUA-RAT ({split}).")
    except Exception as e:
        print(f"[Ingest Warning] AQUA-RAT skipped: {e}")

    return records


def ingest_medmcqa(split: str = "train", limit: int = 20000) -> List[Dict[str, Any]]:
    """Ingest OpenLifeScienceAI MedMCQA (Clinical / medical multiple choice)."""
    print(f"[Ingest] Loading openlifescienceai/medmcqa (split={split}, limit={limit})...")
    records = []
    try:
        ds = load_dataset("openlifescienceai/medmcqa", split=split)
        count = 0
        for i, ex in enumerate(ds):
            if limit and count >= limit:
                break
            q = ex.get("question", "")
            opa = str(ex.get("opa", "")).strip()
            opb = str(ex.get("opb", "")).strip()
            opc = str(ex.get("opc", "")).strip()
            opd = str(ex.get("opd", "")).strip()
            cop = ex.get("cop")
            subj = ex.get("subject_name", "Clinical Medicine")

            if not q or not opa or not opb or not opc or not opd or cop is None:
                continue

            try:
                cop_idx = int(cop) - 1
            except (ValueError, TypeError):
                continue

            choices = [opa, opb, opc, opd]
            if not (0 <= cop_idx < len(choices)):
                continue

            target = choices[cop_idx]
            domain = "09_health_wellbeing" if "medicine" in subj.lower() or "surgery" in subj.lower() or "clinical" in subj.lower() else "08_biology_life_sciences"
            state = f"Clinical case & medical reasoning ({subj}): {q}"
            rec = normalize_record(
                example_id=f"ext_medmcqa_{split}_{i:06d}",
                domain=domain,
                task_family="medmcqa_clinical",
                state=state,
                question=q,
                options=choices,
                target_label=target,
                source="medmcqa",
                difficulty="d4_constraint",
                split=split,
            )
            if rec:
                records.append(rec)
                count += 1
        print(f"[Ingest] Ingested {count} examples from MedMCQA ({split}).")
    except Exception as e:
        print(f"[Ingest Warning] MedMCQA skipped: {e}")

    return records
