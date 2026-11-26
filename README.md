# TalentBridge — Resume Screening and Role Matching

An end-to-end Python data science and NLP capstone project that validates, cleans, normalizes, scores, and analyzes parsed candidate resumes (`parsed_resumes.csv`) for recruiter decision support.

The pipeline separates **transparent requirement + TF-IDF ranking** from an **ML policy-reproduction experiment**, ensuring model probabilities are never blended into candidate rankings and explicit governance guardrails are enforced throughout.

---

## Key Highlights

- **Conservative Data Validation & Auditing:** Checks schema requirements, enforces unique non-null `candidate_id` values, validates numeric bounds and binary flags, flags anomalies and duplicates (`data_review_flag`) without inventing missing values, and avoids arbitrary penalties for resume length or institution tier.
- **Deterministic Skill Normalization:** Tokenizes raw technical skills and tools against a frozen 37-term canonical vocabulary (`VOCAB`) plus explicit aliases (`ALIASES`), automatically repairing only unique single adjacent-letter transpositions and whitespace variants (e.g., `padnas` $\rightarrow$ `pandas`, `dcoker` $\rightarrow$ `docker`, `powre bi` $\rightarrow$ `power bi`) while routing ambiguous or unresolved tokens to review.
- **Transparent Rule + TF-IDF Matching:** Evaluates eligibility (100% mandatory skill match + minimum experience), computes a 100-point structural score (`50%` mandatory skills, `20%` experience alignment, `15%` domain match, `10%` related title match, `5%` delivery/leadership indicators), blends `80%` structural score with `20%` TF-IDF cosine similarity (`1–2` n-grams, `max_features=5000`), and exports row-level explanations, shortlists, and review queues.
- **Leak-Free Classifier Policy Experiment:** Splits data (`SEED=42`) into train (`3,000`), validation (`1,000`), and test (`1,000`) partitions and evaluates `Logistic Regression`, `Random Forest`, and `Gradient Boosting` pipelines using `TextAndStructured` (fitting TF-IDF strictly inside each pipeline fit with no cross-split vocabulary leakage). Models are selected on validation macro F1 and refitted on train+validation for final test reporting.
- **Sensitivity & Governance Reporting:** Tests ranking stability across text weights (`0.1`, `0.2`, `0.3`) and `High Fit` score thresholds (`75`, `80`, `85`), exporting 22 audit, scoring, model, chart, configuration, and report files into `capstone_outputs_final/`.

---

## Project Architecture & Pipeline Stages

1. **Setup & Data Ingestion (Steps 1–3):** Imports Python and `scikit-learn` libraries, sets `SEED=42` and output path `capstone_outputs_final/`, and loads `parsed_resumes.csv` (`5,000` rows $\times$ `34` columns).
2. **Validation & Conservative Cleaning (Steps 4–9):** Validates required fields and unique `candidate_id` values; sets missing `highest_education` and `education_field` to `Unknown`; checks numeric ranges (`years_experience` in `[0, 60]`, `resume_length_words >= 1`, `keyword_density_score` and `profile_completeness_score` in `[0, 1]`) and binary `*_flag` values (`0` or `1`); assigns `seniority_bucket` (`Entry`, `Mid`, `Senior`, `Lead`, `Unknown`); and sets `data_review_flag` for duplicates, short summaries (`<10` words), and role/experience anomalies. Exports `validation_summary.csv` and `cleaned_parsed_resumes.csv`.
3. **Conservative Skill Normalization (Steps 10–13):** Parses comma/semicolon/pipe-separated tokens from `technical_skills_raw` and `tools_platforms_raw`, normalizes against `VOCAB` and `ALIASES` or unique single transposition/spacing matches, records unresolved terms (`skill_review_flag`), and exports `skill_normalization_audit.csv`.
4. **Exploratory Data Analysis (Steps 14–15):** Plots domain counts, experience distribution, and reported resume word count to `eda.png`, and prints dataset findings (`6` domains, `6` years median experience, `48–63` reported resume words, `4,749` repeated experience summaries, `182` skill variants corrected, `0` unresolved unique terms).
5. **Rule + TF-IDF Role Matching (Steps 16–23):** Scores `senior_data_scientist` (`14` eligible) and `software_engineer` (`516` eligible; top `50` shortlisted) using:
   $$\text{structural\_score} = 100 \times \sum_f w_f \cdot \text{feature}_f$$
   $$\text{ranking\_score} = 0.8 \times \text{structural\_score} + 20 \times \text{nlp\_similarity}$$
   Exports per-role all-scores, eligible shortlist, and requirements-review CSVs alongside `role_summary.csv` and `candidate_case_studies.csv`.
6. **Classifier Policy Reproduction (Steps 24–31):** Evaluates `Logistic Regression`, `Random Forest`, and `Gradient Boosting` on `synthetic_label` (`High Fit`, `Medium Fit`, `Low Fit`) for `JOBS[0]` (`Senior Data Scientist`), selecting by validation macro F1 (`Gradient Boosting`) and refitting on train+validation for final test metrics, confusion matrix (`confusion_matrix.png`), and Random Forest policy feature importance (`policy_feature_importance.png`).
7. **Sensitivity & Final Exports (Steps 32–38):** Evaluates top-K overlap, rank correlation, and mean absolute rank displacement across `text_weight` values (`0.1`, `0.2`, `0.3`) and `High Fit` thresholds (`75`, `80`, `85`), then writes `feature_dictionary.csv`, `configuration.json`, and `business_report.md` (`22` total output files).

---

## Repository Contents

```text
├── Final_Expanded_Resume_Screening_Capstone.ipynb
├── parsed_resumes.csv
├── requirements.txt
├── .gitignore
├── LICENSE
└── capstone_outputs_final/
    ├── cleaned_parsed_resumes.csv
    ├── validation_summary.csv
    ├── skill_normalization_audit.csv
    ├── senior_data_scientist_all_scores.csv
    ├── senior_data_scientist_eligible_shortlist.csv
    ├── senior_data_scientist_requirements_review.csv
    ├── software_engineer_all_scores.csv
    ├── software_engineer_eligible_shortlist.csv
    ├── software_engineer_requirements_review.csv
    ├── role_summary.csv
    ├── candidate_case_studies.csv
    ├── model_comparison.csv
    ├── per_class_metrics.csv
    ├── policy_feature_importance.csv
    ├── ranking_sensitivity.csv
    ├── label_threshold_sensitivity.csv
    ├── feature_dictionary.csv
    ├── configuration.json
    ├── business_report.md
    ├── eda.png
    ├── confusion_matrix.png
    └── policy_feature_importance.png
