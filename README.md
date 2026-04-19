# AB-Former: Learning Directed Item Relationships for Content‑Based Recommendation

This repository extends [beeFormer](https://github.com/recombee/beeformer) – a framework that aligns semantic item representations with interaction data by fine‑tuning sentence Transformers on implicit feedback.

We introduce **AB-Former** (Asymmetric Beeformer), which replaces the symmetric similarity from a single embedding (each item scores others via the same row–column outer product) with an asymmetric setup: separate **Q** and **V** embeddings, and similarity is **QV^T**. That lets **“B given A”** differ from **“A given B”**, decouples an item as a *trigger* from the same item as a *target*, and reduces over‑recommending near‑duplicate items.

### Asymmetry types (`--asym_type`)

| Type | Description |
|------|-------------|
| `mlp` | After mean pooling, two independent MLPs produce Q and V. |
| `qformer` | Learned query tokens attend to Transformer hidden states. |
| `prepend` | Learnable prefix tokens prepended to input text; two forward passes with shared weights. |
| `dual` | Two complete sentence Transformer copies (initialised identically, then fine‑tuned separately). |

## Main features

- **Asymmetric beeFormer** – fine‑tune a sentence Transformer with an asymmetric objective (**QV^T**) instead of the symmetric **AA^T** pattern.
- **Asymmetric L³AE** – hybrid model that first learns a semantic prior matrix **S** from asymmetric embeddings, then solves for the collaborative weight matrix **B** in closed form.

## Installation

```bash
python3.10 -m venv abf_env
source abf_env/bin/activate
git clone https://github.com/u1783076/ab-former.git
cd ab-former
pip install -r requirements.txt
```

## Data preparation

The repository expects datasets under `_datasets/`. For the original beeFormer datasets:

- **Goodbooks‑10k** – run `source download_data` inside `_datasets/goodbooks`
- **ML‑20M** – inside `_datasets/ml20m`
- **Amazon Books** – inside `_datasets/amazbooks`, then `python preprocess.py`

For Amazon Grocery / Beauty, prepare the data similarly.

## Training AB-Former

Basic training with the **MLP** separation strategy:

```bash
python train.py \
  --dataset goodbooks \
  --sbert sentence-transformers/all-mpnet-base-v2 \
  --max_seq_length 384 \
  --batch_size 1024 \
  --lr 1e-5 \
  --epochs 5 \
  --use_asym_model true \
  --asym_type mlp \
  --model_name my_asym_model
```

Other asymmetry types: `mlp`, `qformer`, `prepend`, `dual`. See the paper for hyperparameter recommendations per dataset.

## Training asymmetric L³AE

Train a semantic prior matrix **S** from an asymmetric text encoder, then solve for the collaborative weight matrix **B** in closed form:

```bash
python train.py \
  --use_l3ae_model true \
  --use_asym_model true \
  --asym_type mlp \
  --dataset goodbooks \
  --sbert /path/to/checkpoint \
  --model_name my_l3ae_model
```

If `--use_asym_model false`, L³AE uses a symmetric encoder (one embedding space, same **AA^T**-style similarity as in the original setup).

## Evaluation

After training, evaluate on the item‑split or time‑split setting:

```bash
python evaluate_itemsplit.py --model_path my_asym_model --dataset goodbooks
python evaluate_timesplit.py --model_path my_asym_model --dataset amazbooks
```
