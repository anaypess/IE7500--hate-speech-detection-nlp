# Hate Speech and Offensive Language Detection on Social Media Using NLP

**Course:** IE7500 — Applied Natural Language Processing in Engineering
**Team:** Ana Luiza Young Pessoa, Mus Ab Wilmaz, Liang Wang

## Project Overview

This project builds an NLP model to detect and classify hate speech and offensive
language in social media posts (Twitter data). Text is classified into three
categories: **hate speech**, **offensive language**, and **neither**. Automating
this task supports content moderation efforts and contributes to safer online
environments.

The project compares a **TF-IDF + Logistic Regression** baseline against a
**fine-tuned DistilBERT** model, and goes beyond raw classification performance by
evaluating the model for demographic bias and framing it as a **human-in-the-loop**
support tool rather than an autonomous moderation system.

## Problem Statement

Manual moderation of hate speech and offensive content at scale is infeasible given
the volume of content generated daily. This is a multi-class text classification
problem under conditions of class imbalance and linguistic ambiguity — hateful
content often uses coded language, sarcasm, and slang, and the boundary between
hate speech and merely offensive language is a well-known primary source of
classifier error (Davidson et al., 2017).

## Dataset

**Hate Speech and Offensive Language Dataset** (Davidson et al., 2017)

- ~24,802 tweets labeled by crowdsourced annotators (majority vote)
- Classes: hate speech (5.77%), offensive language (77.43%), neither (16.80%)
- Original source: https://github.com/t-davidson/hate-speech-and-offensive-language
- Mirrored on Hugging Face: https://huggingface.co/datasets/tdavidson/hate_speech_offensive

Loaded directly in the notebook via:

```python
import pandas as pd
df = pd.read_parquet("hf://datasets/tdavidson/hate_speech_offensive/data/train-00000-of-00001.parquet")
```

## Repository Structure

```
hate-speech-detection-nlp/
├── README.md
├── requirements.txt
├── notebooks/
│   └── milestone2_hate_speech_detection.ipynb
├── src/                      # (future) modularized scripts extracted from the notebook
├── experiments/
│   └── results/               # saved metrics, confusion matrices, model checkpoints
└── docs/
    └── Milestone1_Proposal.docx
```

## Methodology

1. **EDA** — class distribution, tweet length statistics, duplicate/missing value checks
2. **Preprocessing** — lowercasing, URL/mention removal, text normalization
3. **Split** — stratified 70% train / 15% validation / 15% test
4. **Baseline model** — TF-IDF features + Logistic Regression (`class_weight="balanced"`, with SMOTE considered as an alternative)
5. **Transformer model** — DistilBERT fine-tuned for 3-class sequence classification
6. **Evaluation** — macro F1-score (primary metric), per-class precision/recall/F1, confusion matrix, ROC-AUC where applicable
7. **Bias analysis** — checks for disparate misclassification across demographic/identity term groups
8. **Error analysis** — manual inspection of borderline hate-speech-vs-offensive predictions

## Setup Instructions - How to run this code

```bash
git clone https://github.com/anaypess/IE7500--hate-speech-detection-nlp.git
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook notebooks/milestone2_hate_speech_detection.ipynb
```

## References

- Bespalov, D., Bhabesh, S., Xiang, Y., Zhou, L., & Qi, Y. (2023). Towards building a robust toxicity predictor. *ACL 2023 Industry Track*, 581–598. https://aclanthology.org/2023.acl-industry.56/
- Cao, Y. T., Domingo, L.-F., Gilbert, S., Mazurek, M. L., Shilton, K., & Daumé III, H. (2024). Toxicity detection is NOT all you need. *EMNLP 2024*, 3567–3587. https://doi.org/10.18653/v1/2024.emnlp-main.209
- Davidson, T., Warmsley, D., Macy, M., & Weber, I. (2017). Automated hate speech detection and the problem of offensive language. *ICWSM*, 512–515. https://doi.org/10.1609/icwsm.v11i1.14955
- Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. *NAACL-HLT 2019*, 4171–4186. https://doi.org/10.18653/v1/N19-1423
- Fortuna, P., & Nunes, S. (2018). A survey on automatic detection of hate speech in text. *ACM Computing Surveys*, 51(4), 1–30. https://doi.org/10.1145/3232676
- Jahan, M. S., & Oussalah, M. (2023). A systematic review of hate speech automatic detection using natural language processing. *Neurocomputing*, 546, 126232. https://doi.org/10.1016/j.neucom.2023.126232
