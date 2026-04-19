import os

os.environ["KERAS_BACKEND"] = "torch"

import argparse
import subprocess

from time import time

from callbacks import _build_eval_model

parser = argparse.ArgumentParser()

parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--device", default=None, type=str, help="Limit device to run on")
parser.add_argument("--flag", default="none", type=str, help="flag for distinction of experiments, default none")

# dataset
parser.add_argument("--dataset", default="-", type=str, help="Dataset to run on")

# sentence transformer details
parser.add_argument("--sbert", default="none", type=str, help="Input sentence transformer model to train")
parser.add_argument("--max_seq_length", default=0, type=int, help="Maximum sequece length for sbert")
parser.add_argument("--image_model", default="none", type=str, help="Input image model to test")
parser.add_argument("--is_asym", default="false", type=str)

parser.add_argument(
    "--use_l3ae_model",
    default="false",
    type=str,
    help="Load SparseKerasEASE from --ease_model path: true/false, default false",
)
parser.add_argument(
    "--ease_model",
    default="none",
    type=str,
    help="Path to saved SparseKerasEASE checkpoint (B.npy + items_idx.npy)",
)

args = parser.parse_args([] if "__file__" not in globals() else None)
print(args)

if args.device is not None:
    print(f"Limiting devices to {args.device}")
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.device}"

import keras
import math
import numpy as np
import torch

from models import SparseKerasELSA, SparseKerasEASE
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from config import config
from utils import *

import images

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device {DEVICE}")


def main(args):
    # prepare logging folder
    folder = f"results/{str(pd.Timestamp('today'))} {9*int(1e6)+np.random.randint(999999)}".replace(" ", "_")
    if not os.path.exists(folder):
        os.makedirs(folder)
    vargs = vars(args)
    vargs["cuda_or_cpu"] = DEVICE
    pd.Series(vargs).to_csv(f"{folder}/setup.csv")
    print(folder)

    # set random seeds for reproducibility
    torch.manual_seed(args.seed)
    keras.utils.set_random_seed(args.seed)
    np.random.seed(args.seed)

    # read data
    if args.dataset not in config.keys():
        print("Unknown dataset. List of available datsets: \n")
        for x in config.keys():
            print(x)
        return

    dataset, params = config[args.dataset]
    dataset.load_interactions(**params)
    csev = TimeBasedEvaluation(dataset)

    print(dataset)

    if args.sbert != "none":
        sbert = SentenceTransformer(args.sbert, device=DEVICE, trust_remote_code=True)
        module_with_tokenize = next(
            (m for m in sbert._modules.values()
            if callable(getattr(m, 'tokenize', None))),
            None,
        )
        sbert.tokenize = module_with_tokenize.tokenize
        if args.max_seq_length > 0:
            sbert.max_seq_length = args.max_seq_length

        if args.use_l3ae_model == "true":
            model = SparseKerasEASE.load(args.ease_model, DEVICE)
        else:
            model = _build_eval_model(
                sbert,
                is_asym=args.is_asym == "true",
                texts=dataset.texts,
                items_idx=dataset.all_interactions.item_id.cat.categories,
                device=DEVICE,
            )
    elif args.image_model != "none":
        image_model = images.ImageModel(args.image_model, device=DEVICE)

        tokenized_images_dict = images.read_images_into_dict(
            dataset.all_interactions.item_id.cat.categories,
            fn=image_model.tokenize,
            path=dataset.images_dir,
            suffix=dataset.images_suffix,
        )
        tokenized_test_images = images.read_images_from_dict(
            dataset.all_interactions.item_id.cat.categories, tokenized_images_dict
        )

        embs = image_model.encode(tokenized_test_images)
        model = SparseKerasELSA(
            len(dataset.all_interactions.item_id.cat.categories),
            embs.shape[1],
            dataset.all_interactions.item_id.cat.categories,
            device=DEVICE,
        )
        model.to(DEVICE)
        model.set_weights([embs])

    else:
        print("Model not specified.")


    df_preds = model.predict_df(csev.test_src)
    results = csev(df_preds)
    print(results)

    # final logs
    pd.Series(results).to_csv(f"{folder}/result.csv")
    print("results file written")

    pd.Series(0).to_csv(f"{folder}/timer.csv")
    print("timer written")


if __name__ == "__main__":
    main(args)
