# AmyloFuse

*A protein language model(pLM) embedding-fusion framework for binary amyloid sequence classification.*

AmyloFuse combines configurable pLM representations with pLM-wise attention and a sequence backbone. The default configuration fuses ProtT5, ProtBERT, and ESM-1 features, processes them with a bidirectional LSTM, and predict whether an input protein sequence is amyloid or non-amyloid.

The repository provides end-to-end embedding extraction, training, evaluation, SHAP analysis, model checkpointing, masking experiment, feature caching, and artifact export for the `Amy` and `Cross_Beta_DB` datasets.

---

## 📋 Overview

-   [Architecture](#-architecture)
-   [Repository structure](#-repository-structure)
-   [Installation](#-installation)
-   [Data preparation](#-data-preparation)
-   [Configuration](#-configuration)
-   [Training and evaluation](#-training-and-evaluation)
-   [Outputs and metrics](#-outputs-and-metrics)
-   [Troubleshooting](#-troubleshooting)
-   [References](#-references)

The default pipeline uses protein representations from the ProtTrans family and ESM.[^1].[^2] Each representation is reduced to a shared latent dimension before adaptive embedding reweighting and sequence modeling.

Key capabilities include:

-   Configurable fusion of `prot-t5`, `prot-bert`, `esm-1`, and `esm-2` embeddings
-   Optional sample-wise attention across embedding pLMs
-   `bi-lstm`, `lstm`, and `transformer` sequence backbones
-   Seeded training, early stopping, and gradient clipping
-   Cached feature reuse and export of losses, probabilities, attention weights,and learned visual features
-   SHAP explanations and inference-time masking metrics for each configured pLM

## 📊 Architecture

![AmyloFuse architecture](AmyloFuse.png) *Figure 1: AmyloFuse architecture for pLM feature fusion and amyloid classification.*

The implemented model follows five stages:

1.  ProtT5, ProtBERT, and ESM produce one mean-pooled representation per protein sequence.
2.  A separate linear layer projects each representation to `64` dimensions and stack.
3.  When `use_attention: true`, `AttentionModule` computes sample-wise softmax weights across the selected embedding pLMs and reweights them.
4.  The configured sequence backbone processes the reweighted fusion embeddings. The default backbone is a two-layer stacked bidirectional LSTM.
5.  Residual addition, mean pooling, and an MLP head produce one binary logit.

## 📁 Repository structure

```text
AmyloFuse/
├── main.py                       # Training and evaluation entry point
├── config.yaml                   # Project configuration file
├── requirements.txt              
├── AmyloFuse.png                 # Architecture figure
├── AmyloFuse.log                 # Run logs
├── huggingface/                  # Local ProtTrans pretrained model directories
├── data/
│   ├── Amy/                      # Amy FASTA files
│   └── Cross_Beta_DB/            # Cross-Beta DB FASTA files
├── Cross_Beta_DB_test_set/       # Ten predefined CSV test sets provided in Cross-Beta DB paper
├── dataset/                      # Cache metadata and labels
├── features/                     # Cached pLM representations
├── model/                        # Model checkpoints
├── saved/                        # Exported experiment artifacts
└── src/
    ├── config.py                 # Typed configuration and validation
    ├── core/                     # Classification and analysis pipeline
    │   ├── attention.py          # AttentionModule implementation
    │   ├── dataset.py            # Labels, feature extraction, and cache loading
    │   ├── model.py              # Fusion network, classifier, and source masking
    │   ├── shap.py               # SHAP analysis implementation
    │   └── train.py              # Training, evaluation, SHAP, and masking workflows
    ├── plm/                      # Pretrained protein language model extractors
    │   ├── extract_features_bert.py    # ProtBERT embeddings extracting
    │   ├── extract_features_esm_1.py   # ESM-1 embeddings extracting
    │   ├── extract_features_esm_2.py   # ESM-2 embeddings extracting
    │   └── extract_features_protT5.py  # ProtT5 embeddings extracting
    └── utils/                    # FASTA, split, test-set, and metric utilities
```

The `features/`, `model/`, and `saved/` directories contain cached or generated artifacts. Files under `dataset/` store labels, the seed, and the feature-source names used to decide whether an existing feature cache can be reused.

Within `src/core/`, `shap.py` computes exact SHAP values for groups of pLM embeddings, `train.py` runs training and evaluation pipeline along with the SHAP and masking analyses, and `model.py` removes embedding from a selected pLM during masking. The `saved/shap/` and `saved/masking/` directories contain their respective outputs. The `huggingface/` directory shown above is a suggested location for local pretrained weights, you can replace with your specified paths set in `config.yaml`.

## 🔧 Installation

Create a virtual environment and install the pinned dependencies from the repository root:

```bash
conda create -n AmyloFuse -y python=3.10
conda activate AmyloFuse
cd your filefold
pip install -r requirements.txt
```

Feature extraction can run on CPU, but a CUDA-capable environment is strongly recommended for the pretrained pLMs. Follow the official PyTorch installation guidance if the pinned PyTorch packages need to be matched to a particular CUDA environment.[^3]

### Pretrained model setup

Download pretrained model weights file and place them in `huggingface` cache directory before running:

-   `t5_path` must point to a local Hugging Face-compatible ProtT5 model directory.
-   `bert_path` must point to a local Hugging Face-compatible ProtBERT model directory.
-   Both extractors pass `local_files_only=True`, so they do not download missing models automatically.
-   ESM-1 is loaded through `esm.pretrained.esm1_t34_670M_UR50S()` while ESM-2 is loaded through `esm.pretrained.esm2_t33_650M_UR50D()`. Their weights must already be cached or be obtainable through the fair-esm download mechanism.

## 📦 Data preparation

The supported dataset names are `Amy` and `Cross_Beta_DB`. Store source FASTA files under the matching directory:

```text
data/<dataset>/
├── pos.fasta
├── neg.fasta
├── train_pos.fasta
├── train_neg.fasta
├── test_pos.fasta
└── test_neg.fasta
```

The source files use standard FASTA records. Positive sequences receive label `1`, and negative sequences receive label `0`. Feature extraction truncates every sequence to its first `512` residues.

### Amy dataset

For `dataset: Amy`, each run shuffles `pos.fasta` and `neg.fasta` under the configured global seed and writes an `80:20` train/test split to the four `train_*` and `test_*` FASTA files.

For comparison experiments, `data/Amy/comparison/` contains the fixed training and independent-test split from the AMYPred-FRL study by Charoenkwan *et al.*[^4] It has 132 positive and 305 negative training sequences, plus 33 positive and 77 negative test sequences. The default `main.py` workflow does not read this subdirectory; it creates its own `80:20` split from `data/Amy/pos.fasta` and `data/Amy/neg.fasta` as described above.

### Cross-Beta DB dataset

For `dataset: Cross_Beta_DB`, the entry point processes `Cross_Beta_DB_test_set/test_set_1.csv` through `test_set_10.csv`. Each CSV defines positive and negative external test sequences built by Gonay *et al.*[^5]; the remaining source sequences are sampled to build the training set.

Each test-set CSV must contain `Positive` and `Negative` columns. The conversion uppercases test sequences and removes characters outside the 20 standard amino-acid letters. For each of the ten sets, the code samples 48 remaining sequences per class for training and rewrites the generated FASTA files under `data/Cross_Beta_DB/`.

## ⚙️ Configuration

The repository currently ships with this configuration:

```yaml
dataset: Amy
seed: 4294967291

pretrained_models:
  t5_path: huggingface/prot
  bert_path: huggingface/bert

model:
  feature: ['prot-t5', 'prot-bert', 'esm-1']
  use_attention: true
  sequence_model: bi-lstm
  residual_add: true

training:
  batch_size: 16
  init_learning_rate: 0.005
  epochs: 30
  early_stopping_patience: 10

save:
  model_path: model
  shap_path: saved/shap
  losses_path: saved/losses
  probs_path: saved/probs
  attn_weights_path: saved/attn_weights
  visual_features_path: saved/visual_features
  masking_path: saved/masking
```

The pretrained-model paths in this example are illustrative. The checked-in `config.yaml` currently uses `/huggingface/prot` and `/huggingface/bert`; change them to paths available on your machine. The `save` values are output roots: the program appends `<dataset>/<seed>/` to each root. `src/config.py` requires both new keys, `shap_path` and `masking_path`, when loading a configuration file.

| Section             | Key                       | Default                 | Description                                                     |
|---------------------|---------------------------|-------------------------|-----------------------------------------------------------------|
| Root                | `dataset`                 | `Amy`                   | Selects `Amy` or `Cross_Beta_DB` dataset to run                 |
| Root                | `seed`                    | `4294967291`            | Seeds Python, NumPy, PyTorch, CUDA, data shuffling, and loaders |
| `pretrained_models` | `t5_path`                 | `huggingface/prot`      | Local ProtT5 model directory                                    |
| `pretrained_models` | `bert_path`               | `huggingface/bert`      | Local ProtBERT model directory                                  |
| `model`             | `feature`                 | Three pLMs              | Ordered embedding pLMs to fuse                                  |
| `model`             | `use_attention`           | `true`                  | Enables attention module or not                                 |
| `model`             | `sequence_model`          | `bi-lstm`               | Selects `bi-lstm`, `lstm`, or `transformer`                     |
| `model`             | `residual_add`            | `true`                  | Enables residual connection or not                              |
| `training`          | `batch_size`              | `16`                    | Training and evaluation batch size                              |
| `training`          | `init_learning_rate`      | `0.005`                 | Initial AdamW learning rate                                     |
| `training`          | `epochs`                  | `30`                    | Maximum number of training epochs                               |
| `training`          | `early_stopping_patience` | `10`                    | Epochs without training-loss improvement before stopping        |
| `save`              | `model_path`              | `model`                 | Model checkpoint root                                           |
| `save`              | `shap_path`               | `saved/shap`            | SHAP explanation artifact root                                  |
| `save`              | `losses_path`             | `saved/losses`          | Training-loss artifact root                                     |
| `save`              | `probs_path`              | `saved/probs`           | Prediction-probability artifact root                            |
| `save`              | `attn_weights_path`       | `saved/attn_weights`    | Attention weights artifact root                                 |
| `save`              | `visual_features_path`    | `saved/visual_features` | Feature for UMAP visualization artifact root                    |
| `save`              | `masking_path`            | `saved/masking`         | Per-pLM masking metrics artifact root                           |

For the Amy dataset, select esm-1 as the feature extractor; for the Cross-Beta DB dataset, select esm-2

Recognized feature names are `prot-t5`, `prot-bert`, and `esm-2`. Any other feature name currently falls back to the ESM-1 extractor, so use custom names only after checking the dispatch behavior in `src/core/dataset.py`.

Use `esm-1` explicitly when selecting the ESM-1 extractor. The feature list order also determines the attention-weight columns, SHAP-value columns, and masking-result keys.

## 🚀 Training and evaluation

Run the complete pipeline from the repository root:

```bash
python main.py --config config.yaml
```

The selected dataset determines the workflow:

-   **`Amy`:** Create the seeded `80:20` split, extract or load features, train one model, evaluate one test set, and save its artifacts.
-   **`Cross_Beta_DB`:** Generate ten predefined train/test combinations, train and evaluate each one, and report the mean of every metric.

After the ordinary test prediction, the run performs SHAP analysis for each test sequence and a separate evaluation with each configured pLM embedding removed. These analyses run automatically. SHAP estimates each pLM embedding source's contribution to the model logit. Masking removes one pLM's projected embedding at inference time and reports the resulting classification metrics without retraining.

Early stopping monitors training loss rather than a validation-set metric.

Feature tensors are cached under `features/<dataset>/<set_number>/`. A cache is reused only when its metadata has the same seed, records the requested feature name, and the corresponding tensor exists.

The actual feature paths are `features/Amy/{train,test}/` for `Amy` and `features/Cross_Beta_DB/<set_number>/{train,test}/` for Cross-Beta DB. Cache metadata is stored in `dataset/<dataset>/{train,test}_set.pt`; its validity check does not compare FASTA contents.

## 📈 Outputs and metrics

Outputs are grouped by dataset and seed:

```text
model/<dataset>/<seed>/model_params.pth
saved/losses/<dataset>/<seed>/losses.npy
saved/probs/<dataset>/<seed>/probs.npy
saved/shap/<dataset>/<seed>/shap.pkl
saved/attn_weights/<dataset>/<seed>/attn_weights.npy
saved/visual_features/<dataset>/<seed>/visual_features.npy
saved/masking/<dataset>/<seed>/masking.pkl
AmyloFuse.log
```

For `Amy`, `shap.pkl` contains one `shap.Explanation` whose values have one row per test sample and one column per configured pLM. `masking.pkl` is a dictionary keyed by removed pLM name; each value is a metric dictionary. For `Cross_Beta_DB`, `shap.pkl` contains a list of ten explanations in test-set order, while `masking.pkl` contains per-source metrics averaged across the ten sets. These files use Python pickle, so load them only from trusted runs. The helper `save_explanation()` in `src/core/shap.py` can also write a CSV summary, but the current `main.py` does not call that helper.

Cross-Beta DB checkpoints add the set number (`model/<dataset>/<seed>/<set_number>/model_params.pth`). Depending on whether a collection forms a regular numeric array, multi-set artifacts such as losses can be saved as `.npy` or `.pkl`. Attention weights are absent when `use_attention: false`.

The evaluation threshold is `0.5`. The pipeline reports:

Predicted probabilities must be strictly greater than `0.5` to receive the positive label.

| Metric key   | Meaning                               |
|--------------|---------------------------------------|
| `accuracy`   | Classification accuracy               |
| `sn(recall)` | Sensitivity or recall                 |
| `sp`         | Specificity                           |
| `mcc`        | Matthews correlation coefficient      |
| `auc`        | Area under the ROC curve              |
| `precision`  | Positive predictive value             |
| `f1_score`   | Harmonic mean of precision and recall |

Attention weights are not exported when `use_attention: false`. Numeric artifacts are normally saved as `.npy`; collections that cannot be converted to a regular numeric array are serialized as `.pkl` instead.

## 🔍 Troubleshooting

### ProtT5 or ProtBERT cannot be loaded

Confirm that `t5_path` and `bert_path` point to complete local model directories containing the Hugging Face model and tokenizer files. The project intentionally uses offline loading for these extractors. See the Transformers documentation for model-loading and cache behavior.[^6]

### ESM-2 weights are unavailable

Ensure the environment can access the fair-esm pretrained model cache or permit the initial model download. The ESM project publishes its pretrained model interface and available checkpoints in its official repository.[^7]

### CUDA runs out of memory

Reduce `training.batch_size` in `config.yaml`. If the error occurs during PLM extraction, reduce the extractor `batch_size` in the relevant file under `src/plm/`.

### Cached features no longer match edited FASTA data

The cache validity check uses the seed and feature names; it does not hash FASTA contents. Move the relevant `features/<dataset>/<set_number>/` directory and matching `dataset/<dataset>/*.pt` metadata to a backup location, then rerun to regenerate them. Avoid removing unrelated experiment artifacts.

## 🔗 References

[^1]: Elnaggar, A. et al. (2021). “ProtTrans: Towards Cracking the Language of Life's Code Through Self-Supervised Deep Learning and High Performance Computing.” *arXiv*. [https://arxiv.org/abs/2007.06225](https://arxiv.org/abs/2007.06225)

[^2]: Rives, A. et al. (2021). “Biological Structure and Function Emerge from Scaling Unsupervised Learning to 250 Million Protein Sequences.” *Proceedings of the National Academy of Sciences*. [https://doi.org/10.1073/pnas.2016239118](https://doi.org/10.1073/pnas.2016239118)

[^3]: PyTorch. “PyTorch Documentation.” [https://docs.pytorch.org/docs/stable/](https://docs.pytorch.org/docs/stable/)

[^4]: Charoenkwan, P.; Ahmed, S.; Nantasenamat, C.; Quinn, J. M. W.; Moni, M. A.; Lió, P.; Shoombuatong, W. “AMYPred-FRL Is a Novel Approach for Accurate Prediction of Amyloid Proteins by Using Feature Representation Learning.” *Scientific Reports* **2022**, *12* (1), 7697. [https://doi.org/10.1038/s41598-022-11897-z](https://doi.org/10.1038/s41598-022-11897-z)

[^5]: Gonay, V.; Dunne, M. P.; Caceres‐Delpiano, J.; Kajava, A. V. Developing Machine‐learning‐based Amyloidogenicity Predictors with Cross‐Beta DB. Alzheimer’s & Dementia 2025, 21 (2), e14510. https://doi.org/10.1002/alz.14510

[^6]: Hugging Face. “Transformers Documentation.” [https://huggingface.co/docs/transformers/index](https://huggingface.co/docs/transformers/index)

[^7]: Meta AI Research. “Evolutionary Scale Modeling.” [https://github.com/facebookresearch/esm](https://github.com/facebookresearch/esm)


