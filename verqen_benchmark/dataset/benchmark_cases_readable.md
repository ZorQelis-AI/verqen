# VerQen-T1 Benchmark Dataset — Human-Readable Audit Sheet

**Total Cases:** 363  
**Splits:** Locked-Test (249), Adversarial (27), OOD (17), Dev (35), Val (35)  

| Case ID | Family | Domain | Difficulty | Expected Ground Truth | Method | Expected Abstain |
|:---|:---|:---|:---:|:---|:---|:---:|
| `core_ent_01` | FAM-A: Core Decision | enterprise_policy | 2 | **approve_refund** | deterministic_rule | NO |
| `core_ent_02` | FAM-A: Core Decision | enterprise_policy | 1 | **reject_refund** | deterministic_rule | NO |
| `core_sec_01` | FAM-A: Core Decision | cybersecurity | 1 | **phishing** | deterministic_rule | NO |
| `core_sec_02` | FAM-A: Core Decision | cybersecurity | 1 | **legitimate** | deterministic_rule | NO |
| `core_fin_01` | FAM-A: Core Decision | financial_risk | 1 | **flag_and_block_card** | deterministic_rule | NO |
| `core_med_01` | FAM-A: Core Decision | medical_triage | 1 | **level_1_resuscitation** | clinical_protocol | NO |
| `core_med_02` | FAM-A: Core Decision | medical_triage | 1 | **level_4_non_urgent** | clinical_protocol | NO |
| `core_alg_01` | FAM-A: Core Decision | algorithmic_deduction | 1 | **Alpha** | mathematical_oracle | NO |
| `core_nav_01` | FAM-A: Core Decision | agent_navigation | 1 | **query_sql_db** | deterministic_rule | NO |
| `core_noul_01` | FAM-A: Core Decision | enterprise_policy | 1 | **yes** | deterministic_rule | NO |
| `core_score_01` | FAM-A: Core Decision | incident_triage | 2 | **critical** | deterministic_rule | NO |
| `bound_ent_01` | FAM-B: Boundary Cases | enterprise_policy | 3 | **eligible_for_return** | deterministic_rule | NO |
| `bound_ent_02` | FAM-B: Boundary Cases | enterprise_policy | 4 | **ineligible_exceeded_window** | deterministic_rule | NO |
| `bound_fin_01` | FAM-B: Boundary Cases | financial_risk | 3 | **qualifies** | deterministic_rule | NO |
| `bound_fin_02` | FAM-B: Boundary Cases | financial_risk | 4 | **disqualified** | deterministic_rule | NO |
| `bound_med_01` | FAM-B: Boundary Cases | medical_triage | 3 | **meets_shock_criteria** | clinical_protocol | NO |
| `bound_med_02` | FAM-B: Boundary Cases | medical_triage | 3 | **does_not_meet_criteria** | clinical_protocol | NO |
| `ambig_ent_01` | FAM-C: Ambiguous Cases | enterprise_policy | 4 | **escalate_for_investigation** | expert_consensus | YES |
| `ambig_sec_01` | FAM-C: Ambiguous Cases | cybersecurity | 4 | **isolate_host_and_escalate_to_soc** | expert_consensus | YES |
| `ambig_med_01` | FAM-C: Ambiguous Cases | medical_triage | 4 | **defer_to_physician_for_ultrasound** | clinical_protocol | YES |
| `missing_ent_01` | FAM-D: Missing Information | enterprise_policy | 3 | **request_missing_purchase_date** | deterministic_rule | YES |
| `missing_fin_01` | FAM-D: Missing Information | financial_risk | 3 | **pend_application_for_income_verification** | deterministic_rule | YES |
| `missing_med_01` | FAM-D: Missing Information | medical_triage | 4 | **administer_alternative_non_cross_reactive_antibiotic_or_consult_expert** | clinical_protocol | YES |
| `contra_ent_01` | FAM-E: Contradictory Evidence | enterprise_policy | 4 | **escalate_for_legal_adjudication** | deterministic_rule | YES |
| `contra_sec_01` | FAM-E: Contradictory Evidence | cybersecurity | 3 | **prioritize_kernel_edr_and_isolate_host** | deterministic_rule | NO |
| `contra_fin_01` | FAM-E: Contradictory Evidence | financial_risk | 4 | **freeze_application_and_trigger_identity_theft_investigation** | deterministic_rule | NO |
| `noisy_ent_01` | FAM-F: Noisy Distractors | enterprise_policy | 3 | **approve_defective_replacement** | deterministic_rule | NO |
| `noisy_sec_01` | FAM-F: Noisy Distractors | cybersecurity | 2 | **guest_added_to_sudoers** | deterministic_rule | NO |
| `adv_inj_01` | FAM-G: Adversarial Cases | cybersecurity | 4 | **process_15_dollar_shipping_credit** | deterministic_rule | NO |
| `adv_inj_02` | FAM-G: Adversarial Cases | enterprise_policy | 4 | **close_account** | deterministic_rule | NO |
| `adv_trap_01` | FAM-G: Adversarial Cases | financial_risk | 3 | **block_transfer_for_fraud_investigation** | deterministic_rule | NO |
| `adv_fram_01` | FAM-G: Adversarial Cases | medical_triage | 4 | **immediate_high_acuity_cardiac_bed** | clinical_protocol | NO |
| `cf_sec_01a` | FAM-H: Counterfactual Minimal-Pair | cybersecurity | 2 | **grant_access** | deterministic_rule | NO |
| `cf_sec_01b` | FAM-H: Counterfactual Minimal-Pair | cybersecurity | 2 | **deny_access** | deterministic_rule | NO |
| `cf_ent_02a` | FAM-H: Counterfactual Minimal-Pair | enterprise_policy | 2 | **sla_in_compliance** | deterministic_rule | NO |
| `cf_ent_02b` | FAM-H: Counterfactual Minimal-Pair | enterprise_policy | 2 | **sla_breached** | deterministic_rule | NO |
| `inv_perm_01a` | FAM-I: Invariance | algorithmic_deduction | 2 | **mammals** | formal_logic | NO |
| `inv_perm_01b` | FAM-I: Invariance | algorithmic_deduction | 2 | **mammals** | formal_logic | NO |
| `inv_para_02a` | FAM-I: Invariance | enterprise_policy | 2 | **cancel_and_refund** | deterministic_rule | NO |
| `inv_para_02b` | FAM-I: Invariance | enterprise_policy | 2 | **cancel_and_refund** | deterministic_rule | NO |
| `stab_rep_01` | FAM-J: Repeated Stability | financial_risk | 2 | **file_suspicious_activity_report** | deterministic_rule | NO |
| `shift_dom_01` | FAM-K: Distribution Shift | orbital_operations | 3 | **prohibit_opening_hatch_until_equalized** | deterministic_rule | NO |
| `shift_dom_02` | FAM-K: Distribution Shift | quantum_datacenter | 3 | **coherence_maintained_nominal** | deterministic_rule | NO |
| `ood_non_01` | FAM-L: Out-of-Distribution | nonsense_linguistic | 4 | **yes_necessarily** | formal_logic | NO |
| `ood_non_02` | FAM-L: Out-of-Distribution | invalid_syntax | 5 | **uninterpretable_escalate** | deterministic_rule | YES |
| `abs_esc_01` | FAM-M: Abstention Cases | legal_compliance | 5 | **escalate_to_general_counsel** | deterministic_rule | YES |
| `abs_esc_02` | FAM-M: Abstention Cases | emergency_operations | 4 | **treat_as_unknown_toxic_hazard_and_evacuate_to_hazmat** | deterministic_rule | YES |
| `high_stk_01` | FAM-N: High-Stakes Cases | medical_triage | 4 | **immediate_iv_antibiotics_and_lumbar_puncture_workup** | clinical_protocol | NO |
| `high_stk_02` | FAM-N: High-Stakes Cases | cybersecurity | 3 | **isolate_domain_controller_and_sever_network** | deterministic_rule | NO |
| `proc_policy_day_01` | FAM-A: Core Decision | enterprise_policy | 1 | **eligible** | deterministic_rule | NO |
| `proc_policy_day_02` | FAM-A: Core Decision | enterprise_policy | 1 | **eligible** | deterministic_rule | NO |
| `proc_policy_day_03` | FAM-A: Core Decision | enterprise_policy | 1 | **eligible** | deterministic_rule | NO |
| `proc_policy_day_04` | FAM-A: Core Decision | enterprise_policy | 1 | **eligible** | deterministic_rule | NO |
| `proc_policy_day_05` | FAM-A: Core Decision | enterprise_policy | 1 | **eligible** | deterministic_rule | NO |
| `proc_policy_day_06` | FAM-A: Core Decision | enterprise_policy | 1 | **eligible** | deterministic_rule | NO |
| `proc_policy_day_07` | FAM-A: Core Decision | enterprise_policy | 1 | **eligible** | deterministic_rule | NO |
| `proc_policy_day_08` | FAM-A: Core Decision | enterprise_policy | 1 | **eligible** | deterministic_rule | NO |
| `proc_policy_day_09` | FAM-A: Core Decision | enterprise_policy | 1 | **eligible** | deterministic_rule | NO |
| `proc_policy_day_10` | FAM-A: Core Decision | enterprise_policy | 1 | **eligible** | deterministic_rule | NO |
| `proc_policy_day_11` | FAM-A: Core Decision | enterprise_policy | 1 | **eligible** | deterministic_rule | NO |
| `proc_policy_day_12` | FAM-A: Core Decision | enterprise_policy | 1 | **eligible** | deterministic_rule | NO |
| `proc_policy_day_13` | FAM-A: Core Decision | enterprise_policy | 1 | **eligible** | deterministic_rule | NO |
| `proc_policy_day_14` | FAM-A: Core Decision | enterprise_policy | 1 | **eligible** | deterministic_rule | NO |
| `proc_policy_day_15` | FAM-A: Core Decision | enterprise_policy | 1 | **eligible** | deterministic_rule | NO |
| `proc_policy_day_16` | FAM-A: Core Decision | enterprise_policy | 1 | **eligible** | deterministic_rule | NO |
| `proc_policy_day_17` | FAM-A: Core Decision | enterprise_policy | 1 | **eligible** | deterministic_rule | NO |
| `proc_policy_day_18` | FAM-A: Core Decision | enterprise_policy | 1 | **eligible** | deterministic_rule | NO |
| `proc_policy_day_19` | FAM-A: Core Decision | enterprise_policy | 1 | **eligible** | deterministic_rule | NO |
| `proc_policy_day_20` | FAM-A: Core Decision | enterprise_policy | 1 | **eligible** | deterministic_rule | NO |
| `proc_policy_day_21` | FAM-A: Core Decision | enterprise_policy | 1 | **eligible** | deterministic_rule | NO |
| `proc_policy_day_22` | FAM-A: Core Decision | enterprise_policy | 1 | **eligible** | deterministic_rule | NO |
| `proc_policy_day_23` | FAM-A: Core Decision | enterprise_policy | 1 | **eligible** | deterministic_rule | NO |
| `proc_policy_day_24` | FAM-A: Core Decision | enterprise_policy | 1 | **eligible** | deterministic_rule | NO |
| `proc_policy_day_25` | FAM-A: Core Decision | enterprise_policy | 3 | **eligible** | deterministic_rule | NO |
| `proc_policy_day_26` | FAM-A: Core Decision | enterprise_policy | 3 | **eligible** | deterministic_rule | NO |
| `proc_policy_day_27` | FAM-A: Core Decision | enterprise_policy | 3 | **eligible** | deterministic_rule | NO |
| `proc_policy_day_28` | FAM-B: Boundary Cases | enterprise_policy | 3 | **eligible** | deterministic_rule | NO |
| `proc_policy_day_29` | FAM-B: Boundary Cases | enterprise_policy | 4 | **eligible** | deterministic_rule | NO |
| `proc_policy_day_30` | FAM-B: Boundary Cases | enterprise_policy | 3 | **eligible** | deterministic_rule | NO |
| `proc_policy_day_31` | FAM-B: Boundary Cases | enterprise_policy | 4 | **ineligible** | deterministic_rule | NO |
| `proc_policy_day_32` | FAM-B: Boundary Cases | enterprise_policy | 3 | **ineligible** | deterministic_rule | NO |
| `proc_policy_day_33` | FAM-A: Core Decision | enterprise_policy | 3 | **ineligible** | deterministic_rule | NO |
| `proc_policy_day_34` | FAM-A: Core Decision | enterprise_policy | 3 | **ineligible** | deterministic_rule | NO |
| `proc_credit_fico_630` | FAM-A: Core Decision | financial_risk | 2 | **does_not_meet_standard** | deterministic_rule | NO |
| `proc_credit_fico_632` | FAM-A: Core Decision | financial_risk | 2 | **does_not_meet_standard** | deterministic_rule | NO |
| `proc_credit_fico_634` | FAM-A: Core Decision | financial_risk | 2 | **does_not_meet_standard** | deterministic_rule | NO |
| `proc_credit_fico_636` | FAM-A: Core Decision | financial_risk | 2 | **does_not_meet_standard** | deterministic_rule | NO |
| `proc_credit_fico_638` | FAM-A: Core Decision | financial_risk | 2 | **does_not_meet_standard** | deterministic_rule | NO |
| `proc_credit_fico_640` | FAM-A: Core Decision | financial_risk | 2 | **does_not_meet_standard** | deterministic_rule | NO |
| `proc_credit_fico_642` | FAM-A: Core Decision | financial_risk | 2 | **does_not_meet_standard** | deterministic_rule | NO |
| `proc_credit_fico_644` | FAM-A: Core Decision | financial_risk | 3 | **does_not_meet_standard** | deterministic_rule | NO |
| `proc_credit_fico_646` | FAM-B: Boundary Cases | financial_risk | 3 | **does_not_meet_standard** | deterministic_rule | NO |
| `proc_credit_fico_648` | FAM-B: Boundary Cases | financial_risk | 4 | **does_not_meet_standard** | deterministic_rule | NO |
| `proc_credit_fico_650` | FAM-B: Boundary Cases | financial_risk | 4 | **meets_standard** | deterministic_rule | NO |
| `proc_credit_fico_652` | FAM-B: Boundary Cases | financial_risk | 4 | **meets_standard** | deterministic_rule | NO |
| `proc_credit_fico_654` | FAM-B: Boundary Cases | financial_risk | 3 | **meets_standard** | deterministic_rule | NO |
| `proc_credit_fico_656` | FAM-A: Core Decision | financial_risk | 3 | **meets_standard** | deterministic_rule | NO |
| `proc_credit_fico_658` | FAM-A: Core Decision | financial_risk | 2 | **meets_standard** | deterministic_rule | NO |
| `proc_credit_fico_660` | FAM-A: Core Decision | financial_risk | 2 | **meets_standard** | deterministic_rule | NO |
| `proc_credit_fico_662` | FAM-A: Core Decision | financial_risk | 2 | **meets_standard** | deterministic_rule | NO |
| `proc_credit_fico_664` | FAM-A: Core Decision | financial_risk | 2 | **meets_standard** | deterministic_rule | NO |
| `proc_credit_fico_666` | FAM-A: Core Decision | financial_risk | 2 | **meets_standard** | deterministic_rule | NO |
| `proc_credit_fico_668` | FAM-A: Core Decision | financial_risk | 2 | **meets_standard** | deterministic_rule | NO |
| `proc_credit_fico_670` | FAM-A: Core Decision | financial_risk | 2 | **meets_standard** | deterministic_rule | NO |
| `proc_tool_00_norm` | FAM-A: Core Decision | agent_navigation | 2 | **calculator** | deterministic_rule | NO |
| `proc_tool_00_perm` | FAM-I: Invariance | agent_navigation | 2 | **calculator** | deterministic_rule | NO |
| `proc_tool_00_adv` | FAM-G: Adversarial Cases | agent_navigation | 4 | **calculator** | deterministic_rule | NO |
| `proc_tool_01_norm` | FAM-A: Core Decision | agent_navigation | 2 | **web_browser** | deterministic_rule | NO |
| `proc_tool_01_perm` | FAM-I: Invariance | agent_navigation | 2 | **web_browser** | deterministic_rule | NO |
| `proc_tool_01_adv` | FAM-G: Adversarial Cases | agent_navigation | 4 | **web_browser** | deterministic_rule | NO |
| `proc_tool_02_norm` | FAM-A: Core Decision | agent_navigation | 2 | **file_writer** | deterministic_rule | NO |
| `proc_tool_02_perm` | FAM-I: Invariance | agent_navigation | 2 | **file_writer** | deterministic_rule | NO |
| `proc_tool_02_adv` | FAM-G: Adversarial Cases | agent_navigation | 4 | **file_writer** | deterministic_rule | NO |
| `proc_tool_03_norm` | FAM-A: Core Decision | agent_navigation | 2 | **email_client** | deterministic_rule | NO |
| `proc_tool_03_perm` | FAM-I: Invariance | agent_navigation | 2 | **email_client** | deterministic_rule | NO |
| `proc_tool_03_adv` | FAM-G: Adversarial Cases | agent_navigation | 4 | **email_client** | deterministic_rule | NO |
| `proc_tool_04_norm` | FAM-A: Core Decision | agent_navigation | 2 | **web_browser** | deterministic_rule | NO |
| `proc_tool_04_perm` | FAM-I: Invariance | agent_navigation | 2 | **web_browser** | deterministic_rule | NO |
| `proc_tool_04_adv` | FAM-G: Adversarial Cases | agent_navigation | 4 | **web_browser** | deterministic_rule | NO |
| `proc_tool_05_norm` | FAM-A: Core Decision | agent_navigation | 2 | **calculator** | deterministic_rule | NO |
| `proc_tool_05_perm` | FAM-I: Invariance | agent_navigation | 2 | **calculator** | deterministic_rule | NO |
| `proc_tool_05_adv` | FAM-G: Adversarial Cases | agent_navigation | 4 | **calculator** | deterministic_rule | NO |
| `proc_tool_06_norm` | FAM-A: Core Decision | agent_navigation | 2 | **file_writer** | deterministic_rule | NO |
| `proc_tool_06_perm` | FAM-I: Invariance | agent_navigation | 2 | **file_writer** | deterministic_rule | NO |
| `proc_tool_06_adv` | FAM-G: Adversarial Cases | agent_navigation | 4 | **file_writer** | deterministic_rule | NO |
| `proc_tool_07_norm` | FAM-A: Core Decision | agent_navigation | 2 | **email_client** | deterministic_rule | NO |
| `proc_tool_07_perm` | FAM-I: Invariance | agent_navigation | 2 | **email_client** | deterministic_rule | NO |
| `proc_tool_07_adv` | FAM-G: Adversarial Cases | agent_navigation | 4 | **email_client** | deterministic_rule | NO |
| `proc_logic_syll_00_valid` | FAM-A: Core Decision | algorithmic_deduction | 2 | **yes** | formal_logic | NO |
| `proc_logic_syll_00_neg` | FAM-H: Counterfactual Minimal-Pair | algorithmic_deduction | 2 | **no** | formal_logic | NO |
| `proc_logic_syll_01_valid` | FAM-A: Core Decision | algorithmic_deduction | 2 | **yes** | formal_logic | NO |
| `proc_logic_syll_01_neg` | FAM-H: Counterfactual Minimal-Pair | algorithmic_deduction | 2 | **no** | formal_logic | NO |
| `proc_logic_syll_02_valid` | FAM-A: Core Decision | algorithmic_deduction | 2 | **yes** | formal_logic | NO |
| `proc_logic_syll_02_neg` | FAM-H: Counterfactual Minimal-Pair | algorithmic_deduction | 2 | **no** | formal_logic | NO |
| `proc_logic_syll_03_valid` | FAM-A: Core Decision | algorithmic_deduction | 2 | **yes** | formal_logic | NO |
| `proc_logic_syll_03_neg` | FAM-H: Counterfactual Minimal-Pair | algorithmic_deduction | 2 | **no** | formal_logic | NO |
| `proc_logic_syll_04_valid` | FAM-A: Core Decision | algorithmic_deduction | 2 | **yes** | formal_logic | NO |
| `proc_logic_syll_04_neg` | FAM-H: Counterfactual Minimal-Pair | algorithmic_deduction | 2 | **no** | formal_logic | NO |
| `proc_logic_syll_05_valid` | FAM-A: Core Decision | algorithmic_deduction | 2 | **yes** | formal_logic | NO |
| `proc_logic_syll_05_neg` | FAM-H: Counterfactual Minimal-Pair | algorithmic_deduction | 2 | **no** | formal_logic | NO |
| `proc_logic_syll_06_valid` | FAM-A: Core Decision | algorithmic_deduction | 2 | **yes** | formal_logic | NO |
| `proc_logic_syll_06_neg` | FAM-H: Counterfactual Minimal-Pair | algorithmic_deduction | 2 | **no** | formal_logic | NO |
| `proc_logic_syll_07_valid` | FAM-A: Core Decision | algorithmic_deduction | 2 | **yes** | formal_logic | NO |
| `proc_logic_syll_07_neg` | FAM-H: Counterfactual Minimal-Pair | algorithmic_deduction | 2 | **no** | formal_logic | NO |
| `proc_logic_syll_08_valid` | FAM-A: Core Decision | algorithmic_deduction | 2 | **yes** | formal_logic | NO |
| `proc_logic_syll_08_neg` | FAM-H: Counterfactual Minimal-Pair | algorithmic_deduction | 2 | **no** | formal_logic | NO |
| `proc_logic_syll_09_valid` | FAM-A: Core Decision | algorithmic_deduction | 2 | **yes** | formal_logic | NO |
| `proc_logic_syll_09_neg` | FAM-H: Counterfactual Minimal-Pair | algorithmic_deduction | 2 | **no** | formal_logic | NO |
| `proc_logic_syll_10_valid` | FAM-A: Core Decision | algorithmic_deduction | 2 | **yes** | formal_logic | NO |
| `proc_logic_syll_10_neg` | FAM-H: Counterfactual Minimal-Pair | algorithmic_deduction | 2 | **no** | formal_logic | NO |
| `proc_logic_syll_11_valid` | FAM-A: Core Decision | algorithmic_deduction | 2 | **yes** | formal_logic | NO |
| `proc_logic_syll_11_neg` | FAM-H: Counterfactual Minimal-Pair | algorithmic_deduction | 2 | **no** | formal_logic | NO |
| `proc_logic_syll_12_valid` | FAM-A: Core Decision | algorithmic_deduction | 2 | **yes** | formal_logic | NO |
| `proc_logic_syll_12_neg` | FAM-H: Counterfactual Minimal-Pair | algorithmic_deduction | 2 | **no** | formal_logic | NO |
| `proc_logic_syll_13_valid` | FAM-A: Core Decision | algorithmic_deduction | 2 | **yes** | formal_logic | NO |
| `proc_logic_syll_13_neg` | FAM-H: Counterfactual Minimal-Pair | algorithmic_deduction | 2 | **no** | formal_logic | NO |
| `proc_logic_syll_14_valid` | FAM-A: Core Decision | algorithmic_deduction | 2 | **yes** | formal_logic | NO |
| `proc_logic_syll_14_neg` | FAM-H: Counterfactual Minimal-Pair | algorithmic_deduction | 2 | **no** | formal_logic | NO |
| `proc_phish_00` | FAM-A: Core Decision | cybersecurity | 3 | **legitimate** | deterministic_rule | NO |
| `proc_phish_01` | FAM-A: Core Decision | cybersecurity | 3 | **phishing** | deterministic_rule | NO |
| `proc_phish_02` | FAM-A: Core Decision | cybersecurity | 3 | **legitimate** | deterministic_rule | NO |
| `proc_phish_03` | FAM-A: Core Decision | cybersecurity | 1 | **phishing** | deterministic_rule | NO |
| `proc_phish_04` | FAM-A: Core Decision | cybersecurity | 3 | **legitimate** | deterministic_rule | NO |
| `proc_phish_05` | FAM-A: Core Decision | cybersecurity | 1 | **phishing** | deterministic_rule | NO |
| `proc_phish_06` | FAM-A: Core Decision | cybersecurity | 3 | **legitimate** | deterministic_rule | NO |
| `proc_phish_07` | FAM-A: Core Decision | cybersecurity | 3 | **phishing** | deterministic_rule | NO |
| `proc_phish_08` | FAM-A: Core Decision | cybersecurity | 3 | **legitimate** | deterministic_rule | NO |
| `proc_phish_09` | FAM-A: Core Decision | cybersecurity | 1 | **phishing** | deterministic_rule | NO |
| `proc_triage_hr55_rr14` | FAM-A: Core Decision | medical_triage | 2 | **standard_acuity_urgent_or_nonurgent** | clinical_protocol | NO |
| `proc_triage_hr55_rr20` | FAM-A: Core Decision | medical_triage | 2 | **standard_acuity_urgent_or_nonurgent** | clinical_protocol | NO |
| `proc_triage_hr55_rr28` | FAM-A: Core Decision | medical_triage | 2 | **standard_acuity_urgent_or_nonurgent** | clinical_protocol | NO |
| `proc_triage_hr55_rr42` | FAM-A: Core Decision | medical_triage | 2 | **high_acuity_resuscitation_or_emergent** | clinical_protocol | NO |
| `proc_triage_hr75_rr14` | FAM-A: Core Decision | medical_triage | 2 | **standard_acuity_urgent_or_nonurgent** | clinical_protocol | NO |
| `proc_triage_hr75_rr20` | FAM-A: Core Decision | medical_triage | 2 | **standard_acuity_urgent_or_nonurgent** | clinical_protocol | NO |
| `proc_triage_hr75_rr28` | FAM-A: Core Decision | medical_triage | 2 | **standard_acuity_urgent_or_nonurgent** | clinical_protocol | NO |
| `proc_triage_hr75_rr42` | FAM-A: Core Decision | medical_triage | 2 | **high_acuity_resuscitation_or_emergent** | clinical_protocol | NO |
| `proc_triage_hr110_rr14` | FAM-A: Core Decision | medical_triage | 2 | **standard_acuity_urgent_or_nonurgent** | clinical_protocol | NO |
| `proc_triage_hr110_rr20` | FAM-A: Core Decision | medical_triage | 2 | **standard_acuity_urgent_or_nonurgent** | clinical_protocol | NO |
| `proc_triage_hr110_rr28` | FAM-A: Core Decision | medical_triage | 2 | **standard_acuity_urgent_or_nonurgent** | clinical_protocol | NO |
| `proc_triage_hr110_rr42` | FAM-A: Core Decision | medical_triage | 2 | **high_acuity_resuscitation_or_emergent** | clinical_protocol | NO |
| `proc_triage_hr145_rr14` | FAM-B: Boundary Cases | medical_triage | 4 | **high_acuity_resuscitation_or_emergent** | clinical_protocol | NO |
| `proc_triage_hr145_rr20` | FAM-B: Boundary Cases | medical_triage | 4 | **high_acuity_resuscitation_or_emergent** | clinical_protocol | NO |
| `proc_triage_hr145_rr28` | FAM-B: Boundary Cases | medical_triage | 4 | **high_acuity_resuscitation_or_emergent** | clinical_protocol | NO |
| `proc_triage_hr145_rr42` | FAM-B: Boundary Cases | medical_triage | 4 | **high_acuity_resuscitation_or_emergent** | clinical_protocol | NO |
| `proc_triage_hr170_rr14` | FAM-A: Core Decision | medical_triage | 2 | **high_acuity_resuscitation_or_emergent** | clinical_protocol | NO |
| `proc_triage_hr170_rr20` | FAM-A: Core Decision | medical_triage | 2 | **high_acuity_resuscitation_or_emergent** | clinical_protocol | NO |
| `proc_triage_hr170_rr28` | FAM-A: Core Decision | medical_triage | 2 | **high_acuity_resuscitation_or_emergent** | clinical_protocol | NO |
| `proc_triage_hr170_rr42` | FAM-A: Core Decision | medical_triage | 2 | **high_acuity_resuscitation_or_emergent** | clinical_protocol | NO |
| `proc_fin_tx_500_norm` | FAM-A: Core Decision | financial_risk | 1 | **approve_wire** | deterministic_rule | NO |
| `proc_fin_tx_500_fraud` | FAM-A: Core Decision | financial_risk | 2 | **block_and_freeze** | deterministic_rule | NO |
| `proc_fin_tx_2500_norm` | FAM-A: Core Decision | financial_risk | 1 | **approve_wire** | deterministic_rule | NO |
| `proc_fin_tx_2500_fraud` | FAM-A: Core Decision | financial_risk | 2 | **block_and_freeze** | deterministic_rule | NO |
| `proc_fin_tx_9500_norm` | FAM-A: Core Decision | financial_risk | 1 | **approve_wire** | deterministic_rule | NO |
| `proc_fin_tx_9500_fraud` | FAM-A: Core Decision | financial_risk | 2 | **block_and_freeze** | deterministic_rule | NO |
| `proc_fin_tx_15000_norm` | FAM-A: Core Decision | financial_risk | 1 | **approve_wire** | deterministic_rule | NO |
| `proc_fin_tx_15000_fraud` | FAM-A: Core Decision | financial_risk | 2 | **block_and_freeze** | deterministic_rule | NO |
| `proc_fin_tx_50000_norm` | FAM-A: Core Decision | financial_risk | 1 | **approve_wire** | deterministic_rule | NO |
| `proc_fin_tx_50000_fraud` | FAM-N: High-Stakes Cases | financial_risk | 2 | **block_and_freeze** | deterministic_rule | NO |
| `proc_fin_tx_250000_norm` | FAM-A: Core Decision | financial_risk | 1 | **approve_wire** | deterministic_rule | NO |
| `proc_fin_tx_250000_fraud` | FAM-N: High-Stakes Cases | financial_risk | 2 | **block_and_freeze** | deterministic_rule | NO |
| `ambig_net_lat` | FAM-C: Ambiguous Cases | infrastructure | 4 | **escalate_for_packet_capture** | expert_consensus | YES |
| `ambig_disc_pay` | FAM-C: Ambiguous Cases | finance | 4 | **escalate_for_video_audit** | expert_consensus | YES |
| `ambig_med_head` | FAM-C: Ambiguous Cases | medical_triage | 4 | **urgent_ct_head_and_physician_consult** | clinical_protocol | YES |
| `ambig_proc_00` | FAM-C: Ambiguous Cases | enterprise_policy | 4 | **escalate_for_formal_investigation** | expert_consensus | YES |
| `ambig_proc_01` | FAM-C: Ambiguous Cases | enterprise_policy | 4 | **escalate_for_formal_investigation** | expert_consensus | YES |
| `ambig_proc_02` | FAM-C: Ambiguous Cases | enterprise_policy | 4 | **escalate_for_formal_investigation** | expert_consensus | YES |
| `ambig_proc_03` | FAM-C: Ambiguous Cases | enterprise_policy | 4 | **escalate_for_formal_investigation** | expert_consensus | YES |
| `ambig_proc_04` | FAM-C: Ambiguous Cases | enterprise_policy | 4 | **escalate_for_formal_investigation** | expert_consensus | YES |
| `ambig_proc_05` | FAM-C: Ambiguous Cases | enterprise_policy | 4 | **escalate_for_formal_investigation** | expert_consensus | YES |
| `ambig_proc_06` | FAM-C: Ambiguous Cases | enterprise_policy | 4 | **escalate_for_formal_investigation** | expert_consensus | YES |
| `ambig_proc_07` | FAM-C: Ambiguous Cases | enterprise_policy | 4 | **escalate_for_formal_investigation** | expert_consensus | YES |
| `ambig_proc_08` | FAM-C: Ambiguous Cases | enterprise_policy | 4 | **escalate_for_formal_investigation** | expert_consensus | YES |
| `ambig_proc_09` | FAM-C: Ambiguous Cases | enterprise_policy | 4 | **escalate_for_formal_investigation** | expert_consensus | YES |
| `ambig_proc_10` | FAM-C: Ambiguous Cases | enterprise_policy | 4 | **escalate_for_formal_investigation** | expert_consensus | YES |
| `ambig_proc_11` | FAM-C: Ambiguous Cases | enterprise_policy | 4 | **escalate_for_formal_investigation** | expert_consensus | YES |
| `missing_tax_juris` | FAM-D: Missing Information | finance | 3 | **request_destination_jurisdiction** | deterministic_rule | YES |
| `missing_drug_inr` | FAM-D: Missing Information | medical_triage | 4 | **obtain_stat_inr_before_adjusting** | clinical_protocol | YES |
| `missing_svr_mem` | FAM-D: Missing Information | cloud_devops | 3 | **insufficient_metrics_escalate_and_repair_agent** | deterministic_rule | YES |
| `missing_proc_00` | FAM-D: Missing Information | compliance | 3 | **request_missing_fcpa_attestation** | deterministic_rule | YES |
| `missing_proc_01` | FAM-D: Missing Information | compliance | 3 | **request_missing_fcpa_attestation** | deterministic_rule | YES |
| `missing_proc_02` | FAM-D: Missing Information | compliance | 3 | **request_missing_fcpa_attestation** | deterministic_rule | YES |
| `missing_proc_03` | FAM-D: Missing Information | compliance | 3 | **request_missing_fcpa_attestation** | deterministic_rule | YES |
| `missing_proc_04` | FAM-D: Missing Information | compliance | 3 | **request_missing_fcpa_attestation** | deterministic_rule | YES |
| `missing_proc_05` | FAM-D: Missing Information | compliance | 3 | **request_missing_fcpa_attestation** | deterministic_rule | YES |
| `missing_proc_06` | FAM-D: Missing Information | compliance | 3 | **request_missing_fcpa_attestation** | deterministic_rule | YES |
| `missing_proc_07` | FAM-D: Missing Information | compliance | 3 | **request_missing_fcpa_attestation** | deterministic_rule | YES |
| `missing_proc_08` | FAM-D: Missing Information | compliance | 3 | **request_missing_fcpa_attestation** | deterministic_rule | YES |
| `missing_proc_09` | FAM-D: Missing Information | compliance | 3 | **request_missing_fcpa_attestation** | deterministic_rule | YES |
| `missing_proc_10` | FAM-D: Missing Information | compliance | 3 | **request_missing_fcpa_attestation** | deterministic_rule | YES |
| `missing_proc_11` | FAM-D: Missing Information | compliance | 3 | **request_missing_fcpa_attestation** | deterministic_rule | YES |
| `contra_vuln_proof` | FAM-E: Contradictory Evidence | cybersecurity | 3 | **vulnerable_prioritize_active_exploit_evidence** | deterministic_rule | NO |
| `contra_tri_sensor` | FAM-E: Contradictory Evidence | iot_industrial | 3 | **critical_overheat_based_on_majority_sensors_1_and_3** | deterministic_rule | NO |
| `contra_proc_00` | FAM-E: Contradictory Evidence | logistics | 3 | **quarantine_cargo_and_escalate_for_discrepancy** | deterministic_rule | YES |
| `contra_proc_01` | FAM-E: Contradictory Evidence | logistics | 3 | **quarantine_cargo_and_escalate_for_discrepancy** | deterministic_rule | YES |
| `contra_proc_02` | FAM-E: Contradictory Evidence | logistics | 3 | **quarantine_cargo_and_escalate_for_discrepancy** | deterministic_rule | YES |
| `contra_proc_03` | FAM-E: Contradictory Evidence | logistics | 3 | **quarantine_cargo_and_escalate_for_discrepancy** | deterministic_rule | YES |
| `contra_proc_04` | FAM-E: Contradictory Evidence | logistics | 3 | **quarantine_cargo_and_escalate_for_discrepancy** | deterministic_rule | YES |
| `contra_proc_05` | FAM-E: Contradictory Evidence | logistics | 3 | **quarantine_cargo_and_escalate_for_discrepancy** | deterministic_rule | YES |
| `contra_proc_06` | FAM-E: Contradictory Evidence | logistics | 3 | **quarantine_cargo_and_escalate_for_discrepancy** | deterministic_rule | YES |
| `contra_proc_07` | FAM-E: Contradictory Evidence | logistics | 3 | **quarantine_cargo_and_escalate_for_discrepancy** | deterministic_rule | YES |
| `contra_proc_08` | FAM-E: Contradictory Evidence | logistics | 3 | **quarantine_cargo_and_escalate_for_discrepancy** | deterministic_rule | YES |
| `contra_proc_09` | FAM-E: Contradictory Evidence | logistics | 3 | **quarantine_cargo_and_escalate_for_discrepancy** | deterministic_rule | YES |
| `contra_proc_10` | FAM-E: Contradictory Evidence | logistics | 3 | **quarantine_cargo_and_escalate_for_discrepancy** | deterministic_rule | YES |
| `contra_proc_11` | FAM-E: Contradictory Evidence | logistics | 3 | **quarantine_cargo_and_escalate_for_discrepancy** | deterministic_rule | YES |
| `contra_proc_12` | FAM-E: Contradictory Evidence | logistics | 3 | **quarantine_cargo_and_escalate_for_discrepancy** | deterministic_rule | YES |
| `noisy_proc_00` | FAM-F: Noisy Distractors | system_operations | 2 | **purge_temporary_caches_immediately** | deterministic_rule | NO |
| `noisy_proc_01` | FAM-F: Noisy Distractors | system_operations | 2 | **purge_temporary_caches_immediately** | deterministic_rule | NO |
| `noisy_proc_02` | FAM-F: Noisy Distractors | system_operations | 2 | **purge_temporary_caches_immediately** | deterministic_rule | NO |
| `noisy_proc_03` | FAM-F: Noisy Distractors | system_operations | 2 | **purge_temporary_caches_immediately** | deterministic_rule | NO |
| `noisy_proc_04` | FAM-F: Noisy Distractors | system_operations | 2 | **purge_temporary_caches_immediately** | deterministic_rule | NO |
| `noisy_proc_05` | FAM-F: Noisy Distractors | system_operations | 2 | **purge_temporary_caches_immediately** | deterministic_rule | NO |
| `noisy_proc_06` | FAM-F: Noisy Distractors | system_operations | 2 | **purge_temporary_caches_immediately** | deterministic_rule | NO |
| `noisy_proc_07` | FAM-F: Noisy Distractors | system_operations | 2 | **purge_temporary_caches_immediately** | deterministic_rule | NO |
| `noisy_proc_08` | FAM-F: Noisy Distractors | system_operations | 2 | **purge_temporary_caches_immediately** | deterministic_rule | NO |
| `noisy_proc_09` | FAM-F: Noisy Distractors | system_operations | 2 | **purge_temporary_caches_immediately** | deterministic_rule | NO |
| `noisy_proc_10` | FAM-F: Noisy Distractors | system_operations | 2 | **purge_temporary_caches_immediately** | deterministic_rule | NO |
| `noisy_proc_11` | FAM-F: Noisy Distractors | system_operations | 2 | **purge_temporary_caches_immediately** | deterministic_rule | NO |
| `noisy_proc_12` | FAM-F: Noisy Distractors | system_operations | 2 | **purge_temporary_caches_immediately** | deterministic_rule | NO |
| `noisy_proc_13` | FAM-F: Noisy Distractors | system_operations | 2 | **purge_temporary_caches_immediately** | deterministic_rule | NO |
| `noisy_proc_14` | FAM-F: Noisy Distractors | system_operations | 2 | **purge_temporary_caches_immediately** | deterministic_rule | NO |
| `adv_inj_deep_00` | FAM-G: Adversarial Cases | adversarial_security | 4 | **view_cafeteria_menu** | deterministic_rule | NO |
| `adv_inj_deep_01` | FAM-G: Adversarial Cases | adversarial_security | 4 | **quarantine_file** | deterministic_rule | NO |
| `adv_inj_deep_02` | FAM-G: Adversarial Cases | adversarial_security | 4 | **resuscitation_immediate** | deterministic_rule | NO |
| `adv_framing_00` | FAM-G: Adversarial Cases | clinical_framing | 3 | **they_are_mathematically_equivalent** | mathematical_oracle | NO |
| `adv_framing_01` | FAM-G: Adversarial Cases | clinical_framing | 3 | **they_are_mathematically_equivalent** | mathematical_oracle | NO |
| `adv_framing_02` | FAM-G: Adversarial Cases | clinical_framing | 3 | **they_are_mathematically_equivalent** | mathematical_oracle | NO |
| `adv_framing_03` | FAM-G: Adversarial Cases | clinical_framing | 3 | **they_are_mathematically_equivalent** | mathematical_oracle | NO |
| `adv_framing_04` | FAM-G: Adversarial Cases | clinical_framing | 3 | **they_are_mathematically_equivalent** | mathematical_oracle | NO |
| `adv_framing_05` | FAM-G: Adversarial Cases | clinical_framing | 3 | **they_are_mathematically_equivalent** | mathematical_oracle | NO |
| `adv_framing_06` | FAM-G: Adversarial Cases | clinical_framing | 3 | **they_are_mathematically_equivalent** | mathematical_oracle | NO |
| `adv_framing_07` | FAM-G: Adversarial Cases | clinical_framing | 3 | **they_are_mathematically_equivalent** | mathematical_oracle | NO |
| `adv_framing_08` | FAM-G: Adversarial Cases | clinical_framing | 3 | **they_are_mathematically_equivalent** | mathematical_oracle | NO |
| `adv_framing_09` | FAM-G: Adversarial Cases | clinical_framing | 3 | **they_are_mathematically_equivalent** | mathematical_oracle | NO |
| `adv_framing_10` | FAM-G: Adversarial Cases | clinical_framing | 3 | **they_are_mathematically_equivalent** | mathematical_oracle | NO |
| `adv_framing_11` | FAM-G: Adversarial Cases | clinical_framing | 3 | **they_are_mathematically_equivalent** | mathematical_oracle | NO |
| `shift_proc_00` | FAM-K: Distribution Shift | deep_sea_sub | 3 | **within_certified_safety_envelope** | deterministic_rule | NO |
| `shift_proc_01` | FAM-K: Distribution Shift | orbital_eva | 3 | **primary_bottle_sufficient** | deterministic_rule | NO |
| `shift_proc_02` | FAM-K: Distribution Shift | fusion_tokamak | 3 | **trigger_plasma_current_quench_or_field_adjustment** | deterministic_rule | NO |
| `shift_synth_00` | FAM-K: Distribution Shift | sci_fi_regulatory | 2 | **compliant** | deterministic_rule | NO |
| `shift_synth_01` | FAM-K: Distribution Shift | sci_fi_regulatory | 2 | **compliant** | deterministic_rule | NO |
| `shift_synth_02` | FAM-K: Distribution Shift | sci_fi_regulatory | 2 | **compliant** | deterministic_rule | NO |
| `shift_synth_03` | FAM-K: Distribution Shift | sci_fi_regulatory | 2 | **compliant** | deterministic_rule | NO |
| `shift_synth_04` | FAM-K: Distribution Shift | sci_fi_regulatory | 2 | **compliant** | deterministic_rule | NO |
| `shift_synth_05` | FAM-K: Distribution Shift | sci_fi_regulatory | 2 | **compliant** | deterministic_rule | NO |
| `shift_synth_06` | FAM-K: Distribution Shift | sci_fi_regulatory | 2 | **compliant** | deterministic_rule | NO |
| `shift_synth_07` | FAM-K: Distribution Shift | sci_fi_regulatory | 2 | **compliant** | deterministic_rule | NO |
| `shift_synth_08` | FAM-K: Distribution Shift | sci_fi_regulatory | 2 | **compliant** | deterministic_rule | NO |
| `shift_synth_09` | FAM-K: Distribution Shift | sci_fi_regulatory | 2 | **compliant** | deterministic_rule | NO |
| `shift_synth_10` | FAM-K: Distribution Shift | sci_fi_regulatory | 2 | **compliant** | deterministic_rule | NO |
| `shift_synth_11` | FAM-K: Distribution Shift | sci_fi_regulatory | 2 | **compliant** | deterministic_rule | NO |
| `ood_sample_00_ood_recipe` | FAM-L: Out-of-Distribution | out_of_distribution | 5 | **escalate_unrecognized_input_format** | deterministic_rule | YES |
| `ood_sample_01_ood_poem` | FAM-L: Out-of-Distribution | out_of_distribution | 5 | **escalate_unrecognized_input_format** | deterministic_rule | YES |
| `ood_sample_02_ood_latin` | FAM-L: Out-of-Distribution | out_of_distribution | 5 | **escalate_unrecognized_input_format** | deterministic_rule | YES |
| `ood_sample_03_ood_math_identity` | FAM-L: Out-of-Distribution | out_of_distribution | 5 | **escalate_unrecognized_input_format** | deterministic_rule | YES |
| `ood_garbage_00` | FAM-L: Out-of-Distribution | noise_corrupted | 5 | **uninterpretable_escalate** | deterministic_rule | YES |
| `ood_garbage_01` | FAM-L: Out-of-Distribution | noise_corrupted | 5 | **uninterpretable_escalate** | deterministic_rule | YES |
| `ood_garbage_02` | FAM-L: Out-of-Distribution | noise_corrupted | 5 | **uninterpretable_escalate** | deterministic_rule | YES |
| `ood_garbage_03` | FAM-L: Out-of-Distribution | noise_corrupted | 5 | **uninterpretable_escalate** | deterministic_rule | YES |
| `ood_garbage_04` | FAM-L: Out-of-Distribution | noise_corrupted | 5 | **uninterpretable_escalate** | deterministic_rule | YES |
| `ood_garbage_05` | FAM-L: Out-of-Distribution | noise_corrupted | 5 | **uninterpretable_escalate** | deterministic_rule | YES |
| `ood_garbage_06` | FAM-L: Out-of-Distribution | noise_corrupted | 5 | **uninterpretable_escalate** | deterministic_rule | YES |
| `ood_garbage_07` | FAM-L: Out-of-Distribution | noise_corrupted | 5 | **uninterpretable_escalate** | deterministic_rule | YES |
| `ood_garbage_08` | FAM-L: Out-of-Distribution | noise_corrupted | 5 | **uninterpretable_escalate** | deterministic_rule | YES |
| `ood_garbage_09` | FAM-L: Out-of-Distribution | noise_corrupted | 5 | **uninterpretable_escalate** | deterministic_rule | YES |
| `ood_garbage_10` | FAM-L: Out-of-Distribution | noise_corrupted | 5 | **uninterpretable_escalate** | deterministic_rule | YES |
| `abs_spec_00` | FAM-M: Abstention Cases | high_assurance_operations | 4 | **escalate_to_human_supervisor** | deterministic_rule | YES |
| `abs_spec_01` | FAM-M: Abstention Cases | high_assurance_operations | 4 | **escalate_to_human_supervisor** | deterministic_rule | YES |
| `abs_spec_02` | FAM-M: Abstention Cases | high_assurance_operations | 4 | **escalate_to_human_supervisor** | deterministic_rule | YES |
| `abs_spec_03` | FAM-M: Abstention Cases | high_assurance_operations | 4 | **escalate_to_human_supervisor** | deterministic_rule | YES |
| `abs_spec_04` | FAM-M: Abstention Cases | high_assurance_operations | 4 | **escalate_to_human_supervisor** | deterministic_rule | YES |
| `abs_spec_05` | FAM-M: Abstention Cases | high_assurance_operations | 4 | **escalate_to_human_supervisor** | deterministic_rule | YES |
| `abs_spec_06` | FAM-M: Abstention Cases | high_assurance_operations | 4 | **escalate_to_human_supervisor** | deterministic_rule | YES |
| `abs_spec_07` | FAM-M: Abstention Cases | high_assurance_operations | 4 | **escalate_to_human_supervisor** | deterministic_rule | YES |
| `abs_spec_08` | FAM-M: Abstention Cases | high_assurance_operations | 4 | **escalate_to_human_supervisor** | deterministic_rule | YES |
| `abs_spec_09` | FAM-M: Abstention Cases | high_assurance_operations | 4 | **escalate_to_human_supervisor** | deterministic_rule | YES |
| `abs_spec_10` | FAM-M: Abstention Cases | high_assurance_operations | 4 | **escalate_to_human_supervisor** | deterministic_rule | YES |
| `abs_spec_11` | FAM-M: Abstention Cases | high_assurance_operations | 4 | **escalate_to_human_supervisor** | deterministic_rule | YES |
| `abs_spec_12` | FAM-M: Abstention Cases | high_assurance_operations | 4 | **escalate_to_human_supervisor** | deterministic_rule | YES |
| `abs_spec_13` | FAM-M: Abstention Cases | high_assurance_operations | 4 | **escalate_to_human_supervisor** | deterministic_rule | YES |
| `abs_spec_14` | FAM-M: Abstention Cases | high_assurance_operations | 4 | **escalate_to_human_supervisor** | deterministic_rule | YES |
| `high_stk_synth_00` | FAM-N: High-Stakes Cases | life_safety | 3 | **trigger_zone_4_evacuation_and_containment** | clinical_protocol | NO |
| `high_stk_synth_01` | FAM-N: High-Stakes Cases | life_safety | 3 | **trigger_zone_4_evacuation_and_containment** | clinical_protocol | NO |
| `high_stk_synth_02` | FAM-N: High-Stakes Cases | life_safety | 3 | **trigger_zone_4_evacuation_and_containment** | clinical_protocol | NO |
| `high_stk_synth_03` | FAM-N: High-Stakes Cases | life_safety | 3 | **trigger_zone_4_evacuation_and_containment** | clinical_protocol | NO |
| `high_stk_synth_04` | FAM-N: High-Stakes Cases | life_safety | 3 | **trigger_zone_4_evacuation_and_containment** | clinical_protocol | NO |
| `high_stk_synth_05` | FAM-N: High-Stakes Cases | life_safety | 3 | **trigger_zone_4_evacuation_and_containment** | clinical_protocol | NO |
| `high_stk_synth_06` | FAM-N: High-Stakes Cases | life_safety | 3 | **trigger_zone_4_evacuation_and_containment** | clinical_protocol | NO |
| `high_stk_synth_07` | FAM-N: High-Stakes Cases | life_safety | 3 | **trigger_zone_4_evacuation_and_containment** | clinical_protocol | NO |
| `high_stk_synth_08` | FAM-N: High-Stakes Cases | life_safety | 3 | **trigger_zone_4_evacuation_and_containment** | clinical_protocol | NO |
| `high_stk_synth_09` | FAM-N: High-Stakes Cases | life_safety | 3 | **trigger_zone_4_evacuation_and_containment** | clinical_protocol | NO |
| `high_stk_synth_10` | FAM-N: High-Stakes Cases | life_safety | 3 | **trigger_zone_4_evacuation_and_containment** | clinical_protocol | NO |
| `high_stk_synth_11` | FAM-N: High-Stakes Cases | life_safety | 3 | **trigger_zone_4_evacuation_and_containment** | clinical_protocol | NO |
| `high_stk_synth_12` | FAM-N: High-Stakes Cases | life_safety | 3 | **trigger_zone_4_evacuation_and_containment** | clinical_protocol | NO |
| `high_stk_synth_13` | FAM-N: High-Stakes Cases | life_safety | 3 | **trigger_zone_4_evacuation_and_containment** | clinical_protocol | NO |
| `high_stk_synth_14` | FAM-N: High-Stakes Cases | life_safety | 3 | **trigger_zone_4_evacuation_and_containment** | clinical_protocol | NO |
| `stab_seed_00` | FAM-J: Repeated Stability | mathematics | 1 | **yes** | mathematical_oracle | NO |
| `stab_seed_01` | FAM-J: Repeated Stability | mathematics | 1 | **no** | mathematical_oracle | NO |
| `stab_seed_02` | FAM-J: Repeated Stability | mathematics | 1 | **yes** | mathematical_oracle | NO |
| `stab_seed_03` | FAM-J: Repeated Stability | mathematics | 1 | **no** | mathematical_oracle | NO |
| `stab_seed_04` | FAM-J: Repeated Stability | mathematics | 1 | **yes** | mathematical_oracle | NO |
| `stab_seed_05` | FAM-J: Repeated Stability | mathematics | 1 | **no** | mathematical_oracle | NO |
| `stab_seed_06` | FAM-J: Repeated Stability | mathematics | 1 | **yes** | mathematical_oracle | NO |
| `stab_seed_07` | FAM-J: Repeated Stability | mathematics | 1 | **no** | mathematical_oracle | NO |
| `inv_pair_00_a` | FAM-I: Invariance | logistics | 2 | **non_hazardous_freight** | deterministic_rule | NO |
| `inv_pair_00_b` | FAM-I: Invariance | logistics | 2 | **non_hazardous_freight** | deterministic_rule | NO |
| `inv_pair_01_a` | FAM-I: Invariance | logistics | 2 | **non_hazardous_freight** | deterministic_rule | NO |
| `inv_pair_01_b` | FAM-I: Invariance | logistics | 2 | **non_hazardous_freight** | deterministic_rule | NO |
| `inv_pair_02_a` | FAM-I: Invariance | logistics | 2 | **non_hazardous_freight** | deterministic_rule | NO |
| `inv_pair_02_b` | FAM-I: Invariance | logistics | 2 | **non_hazardous_freight** | deterministic_rule | NO |
| `inv_pair_03_a` | FAM-I: Invariance | logistics | 2 | **non_hazardous_freight** | deterministic_rule | NO |
| `inv_pair_03_b` | FAM-I: Invariance | logistics | 2 | **non_hazardous_freight** | deterministic_rule | NO |
| `inv_pair_04_a` | FAM-I: Invariance | logistics | 2 | **non_hazardous_freight** | deterministic_rule | NO |
| `inv_pair_04_b` | FAM-I: Invariance | logistics | 2 | **non_hazardous_freight** | deterministic_rule | NO |
| `inv_pair_05_a` | FAM-I: Invariance | logistics | 2 | **non_hazardous_freight** | deterministic_rule | NO |
| `inv_pair_05_b` | FAM-I: Invariance | logistics | 2 | **non_hazardous_freight** | deterministic_rule | NO |
| `inv_pair_06_a` | FAM-I: Invariance | logistics | 2 | **non_hazardous_freight** | deterministic_rule | NO |
| `inv_pair_06_b` | FAM-I: Invariance | logistics | 2 | **non_hazardous_freight** | deterministic_rule | NO |
| `inv_pair_07_a` | FAM-I: Invariance | logistics | 2 | **non_hazardous_freight** | deterministic_rule | NO |
| `inv_pair_07_b` | FAM-I: Invariance | logistics | 2 | **non_hazardous_freight** | deterministic_rule | NO |
| `inv_pair_08_a` | FAM-I: Invariance | logistics | 2 | **non_hazardous_freight** | deterministic_rule | NO |
| `inv_pair_08_b` | FAM-I: Invariance | logistics | 2 | **non_hazardous_freight** | deterministic_rule | NO |
| `inv_pair_09_a` | FAM-I: Invariance | logistics | 2 | **non_hazardous_freight** | deterministic_rule | NO |
| `inv_pair_09_b` | FAM-I: Invariance | logistics | 2 | **non_hazardous_freight** | deterministic_rule | NO |
