## 1. Research and Selection of Methods

**Define Objectives:** Our project's NLP task is multi-class text classification —
we want to classify tweets into three categories (hate speech, offensive language,
or neither). This aligns directly with the project's goal of building a
responsible, human-in-the-loop tool to support content moderation.

**Literature Review:** Before implementing anything, we based our approach on
existing research. Davidson et al. (2017), whose dataset we use, already showed
that the hardest part of this task is telling hate speech apart from offensive
language, which is exactly the pattern we later confirmed in our own error
analysis. We also relied on Devlin et al. (2019) to justify testing a transformer
model (DistilBERT), since BERT-based models are known to capture context better
than simple word-frequency models, and on Cao et al. (2024) and Bespalov et al.
(2023) to support treating this as a human-assisted tool rather than a fully
automated one.

**Benchmarking:** We compared two approaches with very different trade-offs:
a lightweight TF-IDF + Logistic Regression baseline (fast, cheap, easy to run on
CPU) versus a fine-tuned DistilBERT model (heavier, needs GPU, but theoretically
more accurate since it understands context). This let us weigh accuracy against
computational cost directly, instead of just assuming the more complex model
would automatically be better.

**Preliminary Experiments:** Since we didn't have GPU access at first, we ran a
small-scale CPU version of DistilBERT (a reduced sample of ~3,000 tweets, small
batch size, 1 epoch) just to confirm the training pipeline worked before
committing to a full run. Once we got GPU access (via Google Colab), we scaled
this up to 8,000 training samples and 2 epochs for the results we report.

## 2. Model Implementation

**Framework Selection:** We used scikit-learn for the baseline model (TF-IDF
vectorizer + Logistic Regression), since it's simple, fast, and well suited for a
linear baseline. For the transformer model, we used Hugging Face's
`transformers` library with PyTorch as the backend, since Hugging Face gives us
a pretrained DistilBERT model and a ready-to-use `Trainer` API, so we didn't have
to write a training loop from scratch.

**Dataset Preparation:** We loaded the Davidson et al. (2017) dataset directly
from Hugging Face Datasets as a parquet file. Before training, we checked for
missing values and duplicates (there were none), looked at class distribution
and tweet length by class, and cleaned the text (lowercasing, removing URLs and
@mentions, removing non-alphabetic characters). We then split the data into
train (70%), validation (15%), and test (15%) sets, stratified by class so the
class proportions stay consistent across all three splits.

**Model Development:** We built the baseline as a standard TF-IDF +
LogisticRegression pipeline. For DistilBERT, we wrote a custom PyTorch
`Dataset` class to wrap our tokenized tweets and labels so they could be fed into
Hugging Face's `Trainer`, keeping preprocessing and model code reusable rather
than hard-coded for one run.

**Training & Fine-Tuning:** The baseline was trained with `class_weight="balanced"`
to help account for the class imbalance (offensive language makes up ~77% of the
data). DistilBERT was fine-tuned using transfer learning, starting from the
pretrained `distilbert-base-uncased` weights instead of training from scratch.
Since we first only had CPU access, we tuned batch size, number of epochs, and
sample size down to keep training feasible, then increased them once GPU access
was available (final run: 8,000 training tweets, batch size 16, 2 epochs).

**Evaluation & Metrics:** We used macro F1-score as our main metric, since it
treats all three classes equally and isn't misleading like plain accuracy would
be on this imbalanced dataset. We also looked at per-class precision, recall, and
F1, plus confusion matrices for both models, to understand exactly where each
model was making mistakes rather than just relying on one overall score. This
led us to notice that DistilBERT's higher accuracy didn't translate into a much
better macro F1, and that it actually did worse on recall for hate speech than
the baseline, which we could only catch by looking beyond the single aggregate
number.