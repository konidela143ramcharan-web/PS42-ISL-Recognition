# Problem Statement 42 — Rubric Assessment

**Assessment date:** 23 October 2026  
**Scope:** Updated assessment of the project files against the uploaded *Image Classification Challenge* document. The ISL training pipeline and notebook were added after the initial folder review.  
**Overall status:** **Not compliant — substantial rework is required before submission.**

## 1. Correct assignment and requirements

The PDF defines **PS-42 as Indian Sign Language (ISL) Recognition**: classify static ISL signs for English alphabets and numbers. The listed dataset is **ISLAN (Mendeley Data), approximately 700 sign images**. Motion signs J and Z are excluded. The statement specifically calls out the small dataset, signs made with one or two hands, and the need to justify augmentation choices, including whether horizontal flipping changes a sign.

The initial project described blood-cell classification. The project code and notebook have since been changed to target ISL and now expect the dataset under `data\ISLAN` (or a path supplied on the command line).

## 2. Rubric score (provisional)

| Rubric criterion | Score | Evidence and rationale |
|---|---:|---|
| Problem Understanding & Planning | **5/10** | The updated objective matches PS-42, and code now counts classes and saves sample images. Actual dataset EDA and discussion of small-data risks, duplicates/leakage, and metric selection are still missing. |
| Technical Approach & Implementation | **7/10** | The code now implements seeded stratified train/validation/test splits, preprocessing, conservative augmentation without horizontal flips, class weights, early stopping, and held-out test metrics. It has only been smoke-tested on synthetic images; actual ISLAN execution and saved notebook outputs are not verified. |
| Innovation & Use of AI/ML Techniques | **3/5** | A standard CNN is supplemented with class weighting and small geometric augmentation, but there is no comparison or evidence demonstrating an improvement on ISL data. |
| Results, Evaluation & Impact | **0/3** | No valid ISL test results or student-written error/impact analysis are available. The pipeline can generate test metrics and diagnostic plots once run on ISLAN. |
| Presentation & Documentation | **0/2** | The updated notebook is a runnable template, but it has no saved outputs. The required two-page report PDF and GitHub link are missing, and the notebook filename does not include the register number. |
| **Total** | **15/30** | Provisional estimate based on the code and documents currently available; actual dataset results and final submission documents were not supplied. |

The score is an evidence-based estimate, not an official grade. The Python pipeline passed a one-epoch smoke test using generated synthetic images; it has **not** been run on the ISLAN dataset, so this does not count as PS-42 model evidence.

## 3. Folder verification

- `main.py` now provides command-line options for the ISL dataset, output folder, image size, batch size, epochs, and seed.
- `problem_statement_42_solution.py` now provides dataset counts/sample plots, deterministic stratified splits, CNN training, and held-out metrics/artifacts.
- `problem_statement_42.ipynb` calls the shared ISL pipeline but has no saved outputs (`execution_count` is unset and outputs are empty).
- `README.md` documents the ISL dataset layout and Windows run instructions; `requirements.txt` includes scikit-learn.
- The pipeline was verified with synthetic images, not with the assignment dataset. No ISLAN dataset, two-page report PDF, or repository URL is present in the folder.
- The current notebook filename does not meet the required `RegisterNo_PS42.ipynb` naming format.

## 4. Work required for a compliant submission

1. Obtain the ISLAN dataset and arrange it in the documented class-folder layout. Verify class names, image counts, and that motion signs J and Z are excluded.
2. Run the notebook or `main.py` on ISLAN and save its outputs. Review class counts and samples, split sizes, learning curves, held-out test accuracy, per-class precision/recall/F1, macro F1, and confusion matrix.
3. Explain at least four misclassified test examples and discuss the dataset's small size and any imbalance, duplicates, or leakage risks.
4. Consider a second model or robustness experiment and report evidence of its effect; discuss real-world relevance, limitations, and ethical concerns.
5. Submit the prescribed report: no more than two A4 pages, with the required sections, hyperparameter table, results table, readable labelled figures, dataset citation, and AI-use declaration. Include name, register number, section, date, PS number/title, and notebook GitHub link. Use the required `RegisterNo_PS42_Report.pdf` and `RegisterNo_PS42.ipynb` filenames.

## 5. Source

Uploaded assignment document: *CBL_DL_Image_Classification_Problem_Statements (1).pdf*, PS-42 description and Parts C–E (instructions, report format, and evaluation rubric).
