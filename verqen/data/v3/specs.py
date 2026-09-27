"""The 24-domain spec library for the v3 scenario engine (docs/production/T2_PLAN.md section 5.2).

Each spec is one subdomain: variables, an ordered policy, actions. The engine does the rest
(rendering, counterfactual families, info-sufficiency, bait, frame hold-out, verification).
Option ids are chosen to be distinct from Artifact-G and the audit probes; the v3 build
hard-fails on any overlap, so a clash here is caught, not shipped.
"""

from __future__ import annotations

from .engine import CatVar as C
from .engine import Clause, Cond, Spec
from .engine import CountVar as Ct
from .engine import DateVar as Dt
from .engine import FlagVar as F
from .engine import NumVar as N
from .engine import ProductVar as P


def flag(name, phrase, t, f, ct, cf):
    """Flag with 4 fact phrasings per polarity (frame 0 held out) and its rule phrases."""
    return F(name, phrase, true_text=t, false_text=f, cond_true=ct, cond_false=cf)


def A(**kw):
    return {k: tuple(v) for k, v in kw.items()}


SPECS: list[Spec] = []


def spec(*a, **kw):
    SPECS.append(Spec(*a, **kw))


# 1 ── customer support ───────────────────────────────────────────────────────────────
spec("customer_support", "return_window",
     [Dt("age", "days since delivery", lo=0, hi=120, event="the delivery"),
      N("value", "order value", lo=5, hi=2500, kind="money"),
      flag("opened", "seal", ("The seal is broken.", "The customer opened the product.", "Packaging: opened.", "The item has been unsealed."),
           ("The seal is intact.", "The product is still sealed.", "Packaging: unopened.", "The item has never been opened."),
           "the product has been opened", "the product is still sealed")],
     [Clause([Cond("opened", "is", [True]), Cond("value", "<", [50, 100])], "store_credit"),
      Clause([Cond("age", ">", [14, 30, 45, 60])], "decline"),
      Clause([Cond("value", ">", [800, 1000, 1500])], "supervisor")],
     "accept",
     A(accept=("accept_return", "accept the return"), decline=("decline_return", "decline the return"),
       store_credit=("offer_store_credit", "offer store credit instead of cash"),
       supervisor=("route_to_returns_supervisor", "route the case to the returns supervisor")),
     ("ask_for_delivery_proof", "ask the customer for proof of delivery details"))
spec("customer_support", "sla_breach",
     [N("wait", "time the ticket has been open", lo=1, hi=600, unit="minutes"),
      C("tier", "support tier", values=["basic", "business", "enterprise"]),
      Ct("reopens", "reopen events", lo=0, hi=6, match_items=["ticket reopened by requester", "customer reopened the ticket"],
         other_items=["agent added an internal note", "priority field viewed", "attachment uploaded ({n} KB)"])],
     [Clause([Cond("tier", "in", [("enterprise",)]), Cond("wait", ">", [30, 60, 120])], "page_duty"),
      Clause([Cond("reopens", ">=", [2, 3])], "senior"),
      Clause([Cond("wait", ">", [240, 360, 480])], "apology")],
     "queue",
     A(queue=("keep_in_standard_queue", "keep it in the standard queue"), page_duty=("page_duty_lead", "page the duty lead"),
       senior=("assign_senior_agent", "assign a senior agent"), apology=("send_delay_apology", "send a delay apology")),
     ("confirm_ticket_history", "confirm the ticket history first"))
spec("customer_support", "account_recovery",
     [Ct("fails", "failed verification attempts", lo=0, hi=7,
         match_items=["verification failed from {n}.20.4.1", "wrong security answer", "one-time code rejected"],
         other_items=["password reset email opened", "profile page viewed", "login success on known device"]),
      flag("id_doc", "identity document", ("A valid ID was uploaded.", "Government ID verified.", "ID check: passed.", "The customer provided a valid ID."),
           ("No ID was uploaded.", "ID check: not done.", "The customer has not provided an ID.", "Identity document: none."),
           "a valid ID has been verified", "no ID has been verified"),
      Dt("since", "days since the last successful login", lo=0, hi=400, event="the last successful login")],
     [Clause([Cond("fails", ">=", [3, 4, 5])], "lock"),
      Clause([Cond("id_doc", "is", [False]), Cond("since", ">", [90, 180])], "manual"),
      Clause([Cond("id_doc", "is", [True])], "restore")],
     "code",
     A(lock=("lock_account_temporarily", "lock the account temporarily"), manual=("send_to_manual_review", "send it to manual review"),
       restore=("restore_access", "restore access"), code=("resend_login_code", "resend a login code")),
     ("request_identity_proof", "request proof of identity"))

# 2 ── finance & accounting ───────────────────────────────────────────────────────────
spec("finance", "invoice_approval",
     [P("amount", "invoice total", qty=(1, 40), each=(10, 400), item="licences", kind="money"),
      C("vendor", "vendor status", values=["approved", "new", "on hold"]),
      Dt("due", "days until the due date", lo=0, hi=60, event="the invoice date")],
     [Clause([Cond("vendor", "in", [("on hold",)])], "reject"),
      Clause([Cond("amount", ">", [2000, 5000, 8000])], "cfo"),
      Clause([Cond("vendor", "in", [("new",)])], "verify_vendor")],
     "pay",
     A(pay=("schedule_payment", "schedule the payment"), reject=("reject_invoice", "reject the invoice"),
       cfo=("send_for_cfo_signoff", "send it for CFO sign-off"), verify_vendor=("run_vendor_verification", "run vendor verification")),
     ("request_itemised_invoice", "request an itemised invoice"))
spec("finance", "expense_claim",
     [N("claim", "claimed amount", lo=5, hi=900, kind="money"),
      C("category", "expense category", values=["meals", "travel", "software", "gifts"]),
      Dt("age", "days since the expense", lo=0, hi=150, event="the expense date"),
      flag("receipt", "receipt", ("An itemised receipt is attached.", "Receipt: attached.", "The claim includes a receipt.", "A receipt was provided."),
           ("No receipt is attached.", "Receipt: missing.", "The claim has no receipt.", "No receipt was provided."),
           "a receipt is attached", "no receipt is attached")],
     [Clause([Cond("age", ">", [60, 90])], "late"),
      Clause([Cond("receipt", "is", [False]), Cond("claim", ">", [25, 75])], "need_receipt"),
      Clause([Cond("category", "in", [("gifts",)]), Cond("claim", ">", [50, 100])], "cap")],
     "reimburse",
     A(reimburse=("reimburse_in_full", "reimburse in full"), late=("refuse_late_claim", "refuse it as a late claim"),
       need_receipt=("hold_until_receipt", "hold it until a receipt arrives"), cap=("reimburse_to_cap", "reimburse only up to the gift cap")),
     ("ask_claimant_to_clarify", "ask the claimant to clarify the claim"))
spec("finance", "payment_run",
     [N("balance", "account balance", lo=0, hi=50000, kind="money", step=100),
      P("batch", "payment batch total", qty=(2, 30), each=(100, 2000), item="payments", kind="money"),
      Ct("flags", "fraud-screen flags", lo=0, hi=4, match_items=["screen flag: mismatched payee name", "screen flag: new bank account"],
         other_items=["payee address updated", "remittance advice generated", "batch exported"])],
     [Clause([Cond("flags", ">=", [1, 2])], "screen"),
      Clause([Cond("batch", ">", [20000, 30000])], "split")],
     "release",
     A(release=("release_payment_batch", "release the batch"), screen=("hold_for_fraud_screening", "hold it for fraud screening"),
       split=("split_into_two_runs", "split it into two runs")),
     ("confirm_account_balance", "confirm the account balance"))

# 3 ── banking & lending ──────────────────────────────────────────────────────────────
spec("banking", "loan_decision",
     [N("score", "credit score", lo=450, hi=850),
      N("dti", "debt-to-income ratio", lo=5, hi=70, kind="pct"),
      N("amount", "requested amount", lo=1000, hi=90000, kind="money", step=500),
      N("years", "years at current employer", lo=0, hi=25, unit="years")],
     [Clause([Cond("score", "<", [560, 600, 620])], "decline"),
      Clause([Cond("dti", ">", [40, 45, 50])], "counter"),
      Clause([Cond("amount", ">", [40000, 50000]), Cond("years", "<", [2, 3])], "underwriter")],
     "approve",
     A(approve=("approve_loan_application", "approve the application"), decline=("decline_loan_application", "decline the application"),
       counter=("offer_smaller_loan", "offer a smaller loan"), underwriter=("send_to_credit_underwriter", "send it to a credit underwriter")),
     ("request_income_documents", "request income documents"))
spec("banking", "wire_screening",
     [N("amount", "transfer amount", lo=100, hi=120000, kind="money", step=50),
      C("country", "destination risk rating", values=["low", "medium", "high"]),
      Dt("acct_age", "account age", lo=0, hi=900, event="the account opening"),
      Ct("prior", "transfers to this payee in the history", lo=0, hi=6,
         match_items=["outgoing wire to this payee", "earlier transfer to the same beneficiary"],
         other_items=["card purchase at a grocery store", "salary deposit", "ATM withdrawal of ${n}"])],
     [Clause([Cond("country", "in", [("high",)]), Cond("amount", ">", [5000, 10000])], "block"),
      Clause([Cond("acct_age", "<", [30, 60]), Cond("amount", ">", [2000, 3000])], "callback"),
      Clause([Cond("prior", "<", [1, 2]), Cond("amount", ">", [15000, 25000])], "callback")],
     "send",
     A(send=("send_the_wire", "send the wire"), block=("block_and_file_report", "block it and file a report"),
       callback=("call_customer_to_confirm", "call the customer to confirm")),
     ("ask_for_payee_details", "ask for full payee details"))
spec("banking", "overdraft",
     [N("overdraft", "overdrawn amount", lo=1, hi=3000, kind="money"),
      Dt("days", "days overdrawn", lo=0, hi=90, event="the account going negative"),
      C("segment", "customer segment", values=["student", "standard", "premier"])],
     [Clause([Cond("segment", "in", [("premier",)]), Cond("overdraft", "<=", [500, 1000])], "waive"),
      Clause([Cond("days", ">", [30, 45, 60])], "collections"),
      Clause([Cond("overdraft", ">", [250, 500])], "fee")],
     "remind",
     A(remind=("send_balance_reminder", "send a balance reminder"), waive=("waive_overdraft_fee", "waive the overdraft fee"),
       collections=("move_to_collections", "move the account to collections"), fee=("charge_overdraft_fee", "charge the overdraft fee")),
     ("verify_account_status", "verify the account status"))

# 4 ── insurance ──────────────────────────────────────────────────────────────────────
spec("insurance", "claim_eligibility",
     [Dt("tenure", "days since the policy started", lo=0, hi=400, event="the policy start"),
      N("claim", "claim amount", lo=100, hi=60000, kind="money", step=50),
      N("limit", "coverage limit", lo=1000, hi=50000, kind="money", step=1000),
      flag("disclosed", "prior damage disclosure", ("Prior damage was disclosed at signup.", "Disclosure: complete.", "The insured declared earlier damage.", "Earlier damage is on file."),
           ("Prior damage was not disclosed.", "Disclosure: missing.", "The insured did not declare earlier damage.", "No disclosure is on file."),
           "prior damage was disclosed", "prior damage was not disclosed")],
     [Clause([Cond("tenure", "<", [30, 60, 90])], "waiting"),
      Clause([Cond("disclosed", "is", [False]), Cond("claim", ">", [5000, 10000])], "investigate"),
      Clause([Cond("claim", ">", [20000, 30000])], "adjuster")],
     "pay",
     A(pay=("pay_the_claim", "pay the claim"), waiting=("deny_within_waiting_period", "deny it as inside the waiting period"),
       investigate=("open_fraud_investigation", "open an investigation"), adjuster=("assign_field_adjuster", "assign a field adjuster")),
     ("request_damage_photos", "request photos of the damage"))
spec("insurance", "deductible",
     [N("loss", "assessed loss", lo=50, hi=12000, kind="money"),
      N("deductible", "policy deductible", lo=100, hi=2500, kind="money", step=50),
      Ct("claims", "claims filed this year", lo=0, hi=5, match_items=["claim filed: water damage", "claim filed: theft", "claim filed: hail"],
         other_items=["policy document downloaded", "address change request", "premium paid"])],
     [Clause([Cond("loss", "<=", [250, 500, 1000])], "below"),
      Clause([Cond("claims", ">=", [3, 4])], "review")],
     "settle",
     A(settle=("settle_minus_deductible", "settle the loss minus the deductible"), below=("close_below_deductible", "close it as below the deductible"),
       review=("send_to_claims_review", "send it to claims review")),
     ("request_repair_estimate", "request a repair estimate"))

# 5 ── healthcare operations (administrative rules only; no diagnosis) ─────────────────
spec("healthcare_ops", "triage_protocol",
     [N("spo2", "oxygen saturation", lo=80, hi=100, kind="pct"),
      N("hr", "heart rate", lo=40, hi=170, unit="bpm"),
      N("pain", "reported pain score", lo=0, hi=10),
      N("age", "patient age", lo=1, hi=95, unit="years")],
     [Clause([Cond("spo2", "<", [90, 92])], "resus"),
      Clause([Cond("hr", ">", [120, 130])], "urgent"),
      Clause([Cond("pain", ">=", [7, 8]), Cond("age", ">", [65, 70])], "urgent")],
     "standard",
     A(standard=("standard_queue", "place the patient in the standard queue"), resus=("resuscitation_bay", "send the patient to the resuscitation bay"),
       urgent=("urgent_assessment", "arrange urgent assessment")),
     ("repeat_vital_signs", "repeat the vital signs"))
spec("healthcare_ops", "appointment_booking",
     [Dt("since", "days since the referral", lo=0, hi=120, event="the referral"),
      C("urgency", "referral urgency", values=["routine", "soon", "urgent"]),
      Ct("missed", "missed appointments", lo=0, hi=5, match_items=["did not attend: clinic slot {n}", "no-show recorded", "missed booked slot"],
         other_items=["appointment attended", "phone consultation completed", "letter sent to GP"])],
     [Clause([Cond("urgency", "in", [("urgent",)])], "two_week"),
      Clause([Cond("missed", ">=", [2, 3])], "discharge"),
      Clause([Cond("since", ">", [60, 90])], "escalate")],
     "book",
     A(book=("book_next_routine_slot", "book the next routine slot"), two_week=("book_within_two_weeks", "book within two weeks"),
       discharge=("return_to_referrer", "return the referral to the referrer"), escalate=("flag_to_clinic_manager", "flag it to the clinic manager")),
     ("contact_referrer_for_details", "contact the referrer for details"))

# 6 ── legal & compliance ─────────────────────────────────────────────────────────────
spec("legal", "contract_signoff",
     [N("value", "contract value", lo=1000, hi=900000, kind="money", step=1000),
      N("term", "contract term", lo=1, hi=72, unit="months"),
      flag("std", "template", ("The contract uses the standard template.", "Template: standard.", "No clauses were changed from the template.", "It is the unmodified standard form."),
           ("The contract has non-standard clauses.", "Template: modified.", "Clauses were changed from the template.", "It deviates from the standard form."),
           "the standard template is used", "the contract deviates from the standard template")],
     [Clause([Cond("std", "is", [False])], "counsel"),
      Clause([Cond("value", ">", [250000, 500000])], "board"),
      Clause([Cond("term", ">", [24, 36])], "director")],
     "sign",
     A(sign=("manager_may_sign", "let the manager sign"), counsel=("send_to_legal_counsel", "send it to legal counsel"),
       board=("needs_board_approval", "obtain board approval"), director=("needs_director_signature", "obtain a director's signature")),
     ("request_final_contract_copy", "request the final contract copy"))
spec("legal", "data_retention",
     [Dt("age", "record age", lo=0, hi=3650, event="the record creation"),
      C("kind", "record type", values=["invoice", "chat log", "employee file", "marketing list"]),
      flag("hold", "litigation hold", ("A litigation hold applies.", "Legal hold: active.", "The record is under legal hold.", "Hold status: on."),
           ("No litigation hold applies.", "Legal hold: none.", "The record is not under legal hold.", "Hold status: off."),
           "a litigation hold applies", "no litigation hold applies")],
     [Clause([Cond("hold", "is", [True])], "keep"),
      Clause([Cond("kind", "in", [("invoice",), ("invoice", "employee file")]), Cond("age", "<", [2555, 2190])], "keep"),
      Clause([Cond("age", ">", [365, 730])], "delete")],
     "archive",
     A(keep=("retain_record", "retain the record"), delete=("securely_delete_record", "securely delete the record"),
       archive=("move_to_cold_archive", "move it to the cold archive")),
     ("ask_records_officer", "ask the records officer"))

# 7 ── cybersecurity ──────────────────────────────────────────────────────────────────
spec("security", "login_anomaly",
     [Ct("fails", "failed logins in the last hour", lo=0, hi=12,
         match_items=["auth failure for user admin from 10.{n}.0.7", "invalid password attempt", "login denied: bad credentials"],
         other_items=["successful login from office VPN", "session refreshed", "MFA prompt approved"]),
      C("geo", "login location", values=["usual city", "new city", "new country"]),
      flag("mfa", "MFA", ("MFA is enabled on the account.", "Second factor: on.", "The account enforces MFA.", "MFA status: active."),
           ("MFA is disabled on the account.", "Second factor: off.", "The account has no MFA.", "MFA status: inactive."),
           "MFA is enabled", "MFA is disabled")],
     [Clause([Cond("fails", ">=", [5, 8])], "lock"),
      Clause([Cond("geo", "in", [("new country",)]), Cond("mfa", "is", [False])], "force_reset"),
      Clause([Cond("geo", "in", [("new country",), ("new city", "new country")])], "notify")],
     "allow",
     A(allow=("allow_session", "allow the session"), lock=("lock_the_account", "lock the account"),
       force_reset=("force_password_reset", "force a password reset"), notify=("notify_account_owner", "notify the account owner")),
     ("pull_full_auth_logs", "pull the full authentication logs"))
spec("security", "vuln_priority",
     [N("cvss", "CVSS score", lo=0.1, hi=10, kind="dec", step=0.1),
      flag("exposed", "internet exposure", ("The service is internet-facing.", "Exposure: public.", "The host is reachable from the internet.", "Public exposure: yes."),
           ("The service is internal only.", "Exposure: internal.", "The host is not reachable from the internet.", "Public exposure: no."),
           "the service is internet-facing", "the service is internal only"),
      flag("exploit", "known exploit", ("A public exploit exists.", "Exploit status: public.", "Exploit code is circulating.", "Known exploited: yes."),
           ("No public exploit is known.", "Exploit status: none.", "No exploit code is known.", "Known exploited: no."),
           "a public exploit exists", "no public exploit is known")],
     [Clause([Cond("exploit", "is", [True]), Cond("exposed", "is", [True])], "now"),
      Clause([Cond("cvss", ">=", [7.0, 9.0])], "week"),
      Clause([Cond("cvss", ">=", [4.0])], "month")],
     "backlog",
     A(now=("patch_within_24_hours", "patch within 24 hours"), week=("patch_within_7_days", "patch within 7 days"),
       month=("patch_within_30_days", "patch within 30 days"), backlog=("add_to_patch_backlog", "add it to the patch backlog")),
     ("confirm_asset_exposure", "confirm the asset exposure"))

# 8 ── IT & DevOps ────────────────────────────────────────────────────────────────────
spec("devops", "incident_severity",
     [N("users", "affected users", lo=0, hi=200000, step=10),
      N("err", "error rate", lo=0, hi=60, kind="pct", step=0.5),
      flag("data", "data loss", ("Data loss has been confirmed.", "Data loss: yes.", "Some records were lost.", "Integrity check failed with lost rows."),
           ("There is no data loss.", "Data loss: no.", "No records were lost.", "Integrity check passed."),
           "data loss is confirmed", "there is no data loss")],
     [Clause([Cond("data", "is", [True])], "sev1"),
      Clause([Cond("users", ">", [10000, 50000])], "sev1"),
      Clause([Cond("err", ">", [5, 10])], "sev2"),
      Clause([Cond("users", ">", [100, 500])], "sev3")],
     "sev4",
     A(sev1=("declare_sev1", "declare SEV1"), sev2=("declare_sev2", "declare SEV2"), sev3=("declare_sev3", "declare SEV3"),
       sev4=("log_as_sev4", "log it as SEV4")),
     ("gather_impact_metrics", "gather impact metrics first"))
spec("devops", "deploy_gate",
     [Ct("failing", "failing test suites", lo=0, hi=5, match_items=["suite {n}: FAILED", "integration tests: FAILED", "e2e checkout: FAILED"],
         other_items=["unit tests: PASSED", "lint: PASSED", "build: PASSED", "suite {n}: PASSED"]),
      N("coverage", "test coverage", lo=40, hi=99, kind="pct"),
      C("window", "deploy window", values=["open", "freeze", "weekend"])],
     [Clause([Cond("window", "in", [("freeze",)])], "block"),
      Clause([Cond("failing", ">=", [1])], "block"),
      Clause([Cond("coverage", "<", [70, 80])], "waiver"),
      Clause([Cond("window", "in", [("weekend",)])], "canary")],
     "deploy",
     A(deploy=("deploy_to_production", "deploy to production"), block=("block_the_release", "block the release"),
       waiver=("request_coverage_waiver", "request a coverage waiver"), canary=("canary_deploy_only", "run a canary deploy only")),
     ("rerun_ci_pipeline", "rerun the CI pipeline"))
spec("devops", "capacity",
     [N("cpu", "CPU utilisation", lo=5, hi=99, kind="pct"),
      N("disk", "free disk space", lo=1, hi=90, kind="pct"),
      Ct("oom", "out-of-memory kills", lo=0, hi=5, match_items=["kernel: OOM killer terminated process {n}", "container OOMKilled"],
         other_items=["cron job finished", "cache warmed", "health check OK"])],
     [Clause([Cond("disk", "<", [5, 10])], "purge"),
      Clause([Cond("oom", ">=", [1, 2])], "mem"),
      Clause([Cond("cpu", ">", [80, 90])], "scale")],
     "nothing",
     A(nothing=("no_action_needed", "take no action"), purge=("purge_old_logs", "purge old logs"),
       mem=("raise_memory_limit", "raise the memory limit"), scale=("add_two_instances", "add two instances")),
     ("collect_more_metrics", "collect more metrics"))

# 9 ── software engineering ───────────────────────────────────────────────────────────
spec("software", "code_review",
     [N("lines", "lines changed", lo=1, hi=3000),
      Ct("approvals", "approvals", lo=0, hi=4, match_items=["review by {n}: APPROVED", "approved by a code owner"],
         other_items=["comment: nit", "review requested", "CI passed"]),
      flag("security", "security-sensitive paths", ("The change touches authentication code.", "Security-sensitive files changed.", "The diff edits crypto helpers.", "Auth module modified."),
           ("No security-sensitive files changed.", "The diff avoids auth and crypto code.", "Only UI files changed.", "Security paths untouched."),
           "security-sensitive files changed", "no security-sensitive files changed")],
     [Clause([Cond("security", "is", [True]), Cond("approvals", "<", [2])], "sec_review"),
      Clause([Cond("lines", ">", [800, 1500])], "split"),
      Clause([Cond("approvals", "<", [1, 2])], "wait")],
     "merge",
     A(merge=("merge_the_change", "merge the change"), sec_review=("request_security_review", "request a security review"),
       split=("ask_author_to_split", "ask the author to split the change"), wait=("wait_for_approvals", "wait for approvals")),
     ("ask_for_change_description", "ask for a change description"))
spec("software", "dependency_policy",
     [Dt("age", "days since the dependency's last release", lo=0, hi=1500, event="the dependency's last release"),
      C("license", "licence", values=["MIT", "Apache-2.0", "GPL-3.0", "unknown"]),
      N("cves", "open CVEs", lo=0, hi=6)],
     [Clause([Cond("license", "in", [("GPL-3.0", "unknown")])], "reject"),
      Clause([Cond("cves", ">=", [1, 2])], "pin_fix"),
      Clause([Cond("age", ">", [540, 730])], "alt")],
     "add",
     A(add=("add_dependency", "add the dependency"), reject=("reject_dependency", "reject the dependency"),
       pin_fix=("pin_patched_version", "pin a patched version"), alt=("look_for_maintained_alternative", "look for a maintained alternative")),
     ("check_licence_file", "check the licence file"))

# 10 ── data labelling policy ─────────────────────────────────────────────────────────
spec("labeling", "annotation_qa",
     [N("agree", "annotator agreement", lo=20, hi=100, kind="pct"),
      Ct("flags", "annotator flags", lo=0, hi=4, match_items=["flag: ambiguous text", "flag: unclear guideline"],
         other_items=["label submitted", "label edited", "time on task {n} s"]),
      N("conf", "model pre-label confidence", lo=10, hi=99, kind="pct")],
     [Clause([Cond("flags", ">=", [2])], "adjudicate"),
      Clause([Cond("agree", "<", [60, 70])], "relabel"),
      Clause([Cond("conf", ">=", [90, 95]), Cond("agree", ">=", [90])], "auto")],
     "accept",
     A(accept=("accept_majority_label", "accept the majority label"), adjudicate=("send_to_adjudicator", "send it to an adjudicator"),
       relabel=("collect_two_more_labels", "collect two more labels"), auto=("auto_accept_label", "auto-accept the label")),
     ("review_guideline_version", "review which guideline version was used"))

# 11 ── web & browser agents ──────────────────────────────────────────────────────────
spec("browser", "checkout_step",
     [C("page", "current page", values=["cart", "shipping form", "payment form", "confirmation"]),
      Ct("empty", "empty required fields", lo=0, hi=4, match_items=["field 'postcode' (required): empty", "field 'phone' (required): empty", "field 'city' (required): empty"],
         other_items=["field 'company' (optional): empty", "field 'name' (required): filled", "field 'notes' (optional): filled"]),
      N("total", "cart total", lo=5, hi=900, kind="money")],
     [Clause([Cond("page", "in", [("confirmation",)])], "done"),
      Clause([Cond("empty", ">=", [1])], "fill"),
      Clause([Cond("page", "in", [("payment form",)]), Cond("total", ">", [300, 500])], "user")],
     "next",
     A(next=("click_continue", "click Continue"), done=("report_task_complete", "report the task complete"),
       fill=("fill_required_fields", "fill the required fields"), user=("ask_user_to_confirm_payment", "ask the user to confirm the payment")),
     ("read_the_page_again", "read the page again"))

# 12 ── tool use & API routing ─────────────────────────────────────────────────────────
spec("tools", "destructive_call",
     [C("tool", "requested tool", values=["read_file", "delete_records", "send_email", "transfer_funds"]),
      N("rows", "rows affected", lo=0, hi=50000, step=10),
      flag("approved", "user approval", ("The user explicitly approved this call.", "Approval: granted.", "The user said yes to this action.", "User consent is recorded."),
           ("The user has not approved this call.", "Approval: none.", "No user consent is recorded.", "The user was not asked."),
           "the user approved the call", "the user has not approved the call")],
     [Clause([Cond("tool", "in", [("transfer_funds",)]), Cond("approved", "is", [False])], "confirm"),
      Clause([Cond("tool", "in", [("delete_records",)]), Cond("rows", ">", [100, 1000])], "confirm"),
      Clause([Cond("tool", "in", [("delete_records", "send_email")]), Cond("approved", "is", [False])], "dry")],
     "run",
     A(run=("execute_tool_call", "execute the call"), confirm=("ask_user_confirmation", "ask the user for confirmation"),
       dry=("run_as_dry_run", "run it as a dry run first")),
     ("request_missing_arguments", "request the missing arguments"))

# 13 ── games & planning: handled by procedural v2 families (tictactoe, nim, tetris, grid, graph)

# 14 ── robotics & physical rules ─────────────────────────────────────────────────────
spec("robotics", "pick_place",
     [N("mass", "object mass", lo=0.1, hi=40, unit="kg", kind="dec", step=0.1),
      N("payload", "gripper payload limit", lo=2, hi=25, unit="kg", kind="dec", step=0.5),
      C("zone", "target zone", values=["free", "occupied", "restricted"]),
      N("dist", "distance to the nearest person", lo=0.2, hi=10, unit="m", kind="dec", step=0.1)],
     [Clause([Cond("dist", "<", [0.5, 1.0])], "stop"),
      Clause([Cond("zone", "in", [("restricted",)])], "stop"),
      Clause([Cond("zone", "in", [("occupied",)])], "wait"),
      Clause([Cond("mass", ">", [8.0, 12.0])], "two_arm")],
     "pick",
     A(pick=("pick_and_place", "pick and place the object"), stop=("halt_motion", "halt all motion"),
       wait=("wait_for_zone_clear", "wait until the zone is clear"), two_arm=("use_two_arm_grasp", "use a two-arm grasp")),
     ("rescan_workspace", "rescan the workspace"))

# 15 ── logistics & supply chain ──────────────────────────────────────────────────────
spec("logistics", "carrier_choice",
     [N("weight", "parcel weight", lo=0.1, hi=90, unit="kg", kind="dec", step=0.1),
      Dt("deadline", "days until the delivery deadline", lo=0, hi=14, event="the order date"),
      flag("hazmat", "hazardous contents", ("The parcel contains lithium batteries.", "Hazmat: yes.", "Contents are classed as dangerous goods.", "Hazard label required."),
           ("The parcel has no hazardous contents.", "Hazmat: no.", "Contents are not dangerous goods.", "No hazard label needed."),
           "the parcel is hazardous", "the parcel is not hazardous")],
     [Clause([Cond("hazmat", "is", [True])], "specialist"),
      Clause([Cond("weight", ">", [30.0, 40.0])], "freight"),
      Clause([Cond("deadline", "<=", [2, 3])], "express")],
     "ground",
     A(ground=("book_ground_shipping", "book ground shipping"), specialist=("book_dangerous_goods_carrier", "book a dangerous-goods carrier"),
       freight=("book_freight_pallet", "book freight on a pallet"), express=("book_express_courier", "book an express courier")),
     ("weigh_and_measure_parcel", "weigh and measure the parcel"))
spec("logistics", "reorder_point",
     [N("stock", "units in stock", lo=0, hi=5000, step=5),
      N("daily", "average daily sales", lo=1, hi=400),
      N("lead", "supplier lead time", lo=1, hi=30, unit="days")],
     [Clause([Cond("stock", "<=", [0])], "expedite"),
      Clause([Cond("stock", "<", [200, 500, 1000])], "reorder")],
     "hold",
     A(hold=("hold_current_stock", "hold current stock"), expedite=("place_expedited_order", "place an expedited order"),
       reorder=("place_standard_reorder", "place a standard reorder")),
     ("run_stock_recount", "run a stock recount"))

# 16 ── e-commerce & retail ───────────────────────────────────────────────────────────
spec("retail", "promotion_rules",
     [P("basket", "basket value", qty=(1, 12), each=(3, 120), item="items", kind="money"),
      C("tier", "loyalty tier", values=["none", "silver", "gold"]),
      flag("coupon", "coupon", ("A coupon code was entered.", "Coupon: applied.", "The shopper used a promo code.", "Promo code present."),
           ("No coupon code was entered.", "Coupon: none.", "The shopper used no promo code.", "Promo code absent."),
           "a coupon was entered", "no coupon was entered")],
     [Clause([Cond("coupon", "is", [True]), Cond("tier", "in", [("gold",)])], "best_only"),
      Clause([Cond("basket", ">=", [100, 150, 200])], "free_ship"),
      Clause([Cond("tier", "in", [("silver", "gold")])], "loyalty")],
     "none",
     A(none=("no_discount", "apply no discount"), best_only=("apply_best_single_discount", "apply only the best single discount"),
       free_ship=("apply_free_shipping", "apply free shipping"), loyalty=("apply_loyalty_discount", "apply the loyalty discount")),
     ("recalculate_basket", "recalculate the basket"))

# 17 ── HR & workplace policy (job-related criteria only) ───────────────────────────────
spec("hr", "leave_request",
     [N("balance", "remaining leave balance", lo=0, hi=30, unit="days"),
      N("days", "requested days", lo=1, hi=25, unit="days"),
      Dt("notice", "notice given", lo=0, hi=60, event="the request submission")],
     [Clause([Cond("days", ">", [10, 15])], "director"),
      Clause([Cond("notice", "<", [7, 14])], "short_notice")],
     "approve",
     A(approve=("approve_leave", "approve the leave"), director=("needs_director_approval", "get director approval"),
       short_notice=("discuss_with_line_manager", "discuss it with the line manager")),
     ("confirm_leave_balance", "confirm the leave balance"))
spec("hr", "overtime",
     [N("hours", "overtime hours this week", lo=0, hi=30, unit="hours", kind="dec", step=0.5),
      Ct("weeks", "consecutive weeks with overtime", lo=0, hi=8, match_items=["week {n}: overtime logged"],
         other_items=["week {n}: standard hours", "training day logged"]),
      C("role", "contract type", values=["hourly", "salaried", "contractor"])],
     [Clause([Cond("role", "in", [("contractor",)])], "vendor"),
      Clause([Cond("hours", ">", [12.0, 16.0])], "cap"),
      Clause([Cond("weeks", ">=", [3, 4])], "wellbeing")],
     "pay",
     A(pay=("pay_overtime_rate", "pay the overtime rate"), vendor=("bill_through_agency", "bill it through the agency"),
       cap=("cap_and_review_rota", "cap the hours and review the rota"), wellbeing=("schedule_wellbeing_check", "schedule a wellbeing check")),
     ("verify_timesheet", "verify the timesheet"))

# 18 ── manufacturing & industrial ────────────────────────────────────────────────────
spec("manufacturing", "qc_tolerance",
     [N("dev", "measured deviation", lo=0, hi=2.0, unit="mm", kind="dec", step=0.01),
      N("tol", "tolerance", lo=0.05, hi=1.0, unit="mm", kind="dec", step=0.05),
      Ct("defects", "defects in the sample", lo=0, hi=6, match_items=["unit {n}: scratch", "unit {n}: burr", "unit {n}: porosity"],
         other_items=["unit {n}: OK", "unit {n}: within spec"])],
     [Clause([Cond("defects", ">=", [2, 3])], "quarantine"),
      Clause([Cond("dev", ">", [0.5, 0.8])], "scrap"),
      Clause([Cond("dev", ">", [0.2, 0.3])], "rework")],
     "ship",
     A(ship=("release_lot", "release the lot"), quarantine=("quarantine_lot", "quarantine the lot"),
       scrap=("scrap_part", "scrap the part"), rework=("send_to_rework", "send the part to rework")),
     ("remeasure_part", "remeasure the part"))
spec("manufacturing", "sensor_voting",
     [N("s1", "sensor A reading", lo=20, hi=180, unit="C"),
      N("s2", "sensor B reading", lo=20, hi=180, unit="C"),
      N("s3", "sensor C reading", lo=20, hi=180, unit="C")],
     [Clause([Cond("s1", ">", [120]), Cond("s2", ">", [120])], "trip"),
      Clause([Cond("s1", ">", [120]), Cond("s3", ">", [120])], "trip"),
      Clause([Cond("s2", ">", [120]), Cond("s3", ">", [120])], "trip"),
      Clause([Cond("s1", ">", [120])], "check"),
      Clause([Cond("s2", ">", [120])], "check"),
      Clause([Cond("s3", ">", [120])], "check")],
     "run",
     A(run=("continue_line_operation", "keep the line running"), trip=("trip_the_line", "trip the line"),
       check=("inspect_single_sensor", "inspect the single high sensor")),
     ("recalibrate_sensors", "recalibrate the sensors"))

# 19 ── energy & utilities ────────────────────────────────────────────────────────────
spec("energy", "load_shedding",
     [N("load", "grid load", lo=40, hi=110, kind="pct"),
      N("reserve", "spinning reserve", lo=0, hi=2000, unit="MW", step=10),
      C("period", "time period", values=["off-peak", "shoulder", "peak"])],
     [Clause([Cond("load", ">", [100])], "shed"),
      Clause([Cond("reserve", "<", [200, 300])], "start_peaker"),
      Clause([Cond("period", "in", [("peak",)]), Cond("load", ">", [90, 95])], "demand_response")],
     "normal",
     A(normal=("normal_operation", "continue normal operation"), shed=("shed_non_critical_load", "shed non-critical load"),
       start_peaker=("start_peaker_plant", "start the peaker plant"), demand_response=("trigger_demand_response", "trigger demand response")),
     ("refresh_scada_readings", "refresh the SCADA readings"))

# 20 ── transportation & travel ───────────────────────────────────────────────────────
spec("travel", "missed_connection",
     [N("delay", "arrival delay", lo=0, hi=400, unit="minutes"),
      N("gap", "scheduled connection time", lo=25, hi=240, unit="minutes"),
      C("fare", "fare class", values=["basic", "standard", "flexible"])],
     [Clause([Cond("delay", ">", [180, 240])], "hotel"),
      Clause([Cond("fare", "in", [("flexible",)])], "any_flight"),
      Clause([Cond("delay", ">", [30, 45, 60])], "rebook")],
     "keep",
     A(keep=("keep_original_itinerary", "keep the original itinerary"), hotel=("rebook_and_offer_hotel", "rebook and offer a hotel"),
       any_flight=("offer_any_same_day_flight", "offer any same-day flight"), rebook=("rebook_next_connection", "rebook the next connection")),
     ("check_live_flight_status", "check the live flight status"))

# 21 ── education ─────────────────────────────────────────────────────────────────────
spec("education", "rubric_grading",
     [N("score", "rubric points", lo=0, hi=100),
      Dt("late", "days late", lo=0, hi=10, event="the submission deadline"),
      Ct("sources", "cited sources", lo=0, hi=8, match_items=["reference: journal article", "reference: textbook chapter", "reference: government report"],
         other_items=["figure caption", "appendix table", "acknowledgement"])],
     [Clause([Cond("late", ">", [5, 7])], "zero"),
      Clause([Cond("sources", "<", [2, 3])], "cap"),
      Clause([Cond("score", ">=", [85, 90])], "distinction"),
      Clause([Cond("score", ">=", [50])], "pass")],
     "fail",
     A(fail=("record_fail", "record a fail"), zero=("record_zero_for_lateness", "record zero for lateness"),
       cap=("cap_at_pass_mark", "cap the grade at the pass mark"), distinction=("record_distinction", "record a distinction"),
       **{"pass": ("record_pass", "record a pass")}),
     ("ask_marker_to_recheck", "ask the marker to recheck"))

# 22 ── math & quantitative decisions ─────────────────────────────────────────────────
spec("quantitative", "budget_check",
     [P("cost", "total cost", qty=(2, 60), each=(4, 90), item="seats", kind="money"),
      N("budget", "remaining budget", lo=100, hi=3000, kind="money", step=10)],
     [Clause([Cond("cost", ">", [1500, 2000])], "cfo")],
     "book",
     A(book=("book_within_budget", "book it"), cfo=("seek_budget_increase", "seek a budget increase")),
     ("get_exact_quote", "get an exact quote"))
spec("quantitative", "unit_conversion_limit",
     [N("load_kg", "cargo mass", lo=100, hi=6000, unit="kg", step=10),
      N("axles", "number of axles", lo=2, hi=6)],
     [Clause([Cond("load_kg", ">", [3500, 4500])], "heavy"),
      Clause([Cond("load_kg", ">", [2000, 2500]), Cond("axles", "<", [3])], "reroute")],
     "go",
     A(go=("dispatch_vehicle", "dispatch the vehicle"), heavy=("need_heavy_goods_permit", "obtain a heavy goods permit"),
       reroute=("use_truck_route", "use the truck route")),
     ("weigh_at_depot", "weigh it at the depot"))

# 23 ── reading & NLI: real datasets (RACE, BoolQ, MNLI, ...) cover this domain

# 24 ── safety & adversarial robustness ───────────────────────────────────────────────
spec("safety", "escalation_policy",
     [C("request", "request type", values=["account question", "refund request", "threat of self-harm", "legal threat"]),
      N("sentiment", "sentiment score", lo=-100, hi=100),
      Ct("caps", "all-caps messages", lo=0, hi=6, match_items=["message: I WANT THIS FIXED NOW", "message: WHERE IS MY ORDER"],
         other_items=["message: thanks for your help", "message: could you check this?", "message: ok"])],
     [Clause([Cond("request", "in", [("threat of self-harm",)])], "crisis"),
      Clause([Cond("request", "in", [("legal threat",)])], "legal"),
      Clause([Cond("sentiment", "<", [-60, -40]), Cond("caps", ">=", [2, 3])], "deescalate")],
     "normal",
     A(normal=("answer_normally", "answer normally"), crisis=("follow_crisis_protocol", "follow the crisis protocol"),
       legal=("hand_to_legal_team", "hand it to the legal team"), deescalate=("use_deescalation_script", "use the de-escalation script")),
     ("ask_clarifying_question", "ask a clarifying question"))

from .specs_more import register as _register_more  # noqa: E402

_register_more(spec, flag, A)

DOMAINS = sorted({s.domain for s in SPECS})
