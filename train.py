import os
import argparse 

# need to parse arguments before loading pytorch
parser = argparse.ArgumentParser()

parser.add_argument("--seed", default=42, type=int, help="Random seed, default 42")
parser.add_argument("--device", default=None, type=str, help="Limit device to run on, default None (no limit)")
parser.add_argument(
    "--devices", default=None, type=str, help="Devices for multi-device training ex. [0,1,2,3], default None "
)
parser.add_argument("--flag", default="none", type=str, help="flag for distinction of experiments, default none")
parser.add_argument("--validation", default="false", type=str, help="Use validation split: true/false, default false")

# lr
parser.add_argument(
    "--lr", default=1e-5, type=float, help="Learning rate for model training, only if scheduler is none, default 1e-5"
)
parser.add_argument(
    "--scheduler", default="none", type=str, help="Scheduler: LinearWarmup, CosineDecay or none, default none"
)
parser.add_argument("--init_lr", default=0.0, type=float, help="starting lr, only if scheduler is not none, default 0")
parser.add_argument(
    "--warmup_lr", default=1e-4, type=float, help="max warmup lr, only if scheduler is not none, default 1e-4"
)
parser.add_argument(
    "--target_lr", default=1e-6, type=float, help="final lr, only if scheduler is LinearWarmup, default 1e-6"
)
parser.add_argument(
    "--warmup_epochs", default=1, type=int, help="Warmup epochs, only if scheduler is not none, default 1"
)
parser.add_argument(
    "--decay_epochs", default=3, type=int, help="Decay epochs, only if scheduler is not none, default 3"
)
parser.add_argument(
    "--tuning_epochs", default=1, type=int, help="Final lr epochs, only if scheduler is LinearWarmup, default 1"
)
parser.add_argument("--epochs", default=5, type=int, help="Training epochs, only if scheduler is none, default 5")

# dataset
parser.add_argument("--dataset", default="-", type=str, help="Dataset to run on")
parser.add_argument("--use_cold_start", default="false", type=str, help="Use cold start evaluation, default false")
parser.add_argument("--use_time_split", default="false", type=str, help="Use time split evaluation, default false")
parser.add_argument(
    "--prefix",
    default=None,
    type=str,
    help="Add prefix to every item description (example for e5 models add query: as prefix to every item description - see https://huggingface.co/intfloat/multilingual-e5-base#faq), default None",
)

# sentence transformer details
parser.add_argument("--sbert", default=None, type=str, help="Input sentence transformer model to train")
parser.add_argument("--image_model", default=None, type=str, help="Input image model model to train")

parser.add_argument(
    "--max_seq_length",
    default=None,
    type=int,
    help="Maximum sequence length, default None (use original value from sbert)",
)
parser.add_argument(
    "--preproces_html",
    default="false",
    type=str,
    help="whether to get rid of html inside descriptions (not relevant for LLM generated descriptions), default false",
)

# model hyperparams
parser.add_argument(
    "--max_output",
    default=None,
    type=int,
    help="Max number of items on output (m parameter from paper), default 10000",
)
parser.add_argument(
    "--batch_size", default=1024, type=int, help="Batch size of sampled users per training step, default 1024"
)
parser.add_argument(
    "--top_k",
    default=0,
    type=int,
    help="Optimize only for top-k predictions on the output of the model. May bring some improvement for large, sparse datasets (in theory). Default 0 (not use)",
)
parser.add_argument(
    "--sbert_batch_size",
    default=200,
    type=int,
    help="Batch size for computing embeddings with sentence transformer, default 200",
)
parser.add_argument(
    "--use_asym_model",
    default="false",
    type=str,
    help="Whether to append asymmetric Q/V projection heads to the SBERT pipeline: true/false, default false",
)
parser.add_argument(
    "--asym_type",
    default=None,
    type=str,
    help="Type of asymmetric projection module: mlp | qformer | prepend | dual, default mlp",
)
parser.add_argument(
    "--qformer_layers",
    default=2,
    type=int,
    help="Number of QFormer cross-attention layers, default 2",
)
parser.add_argument(
    "--qformer_heads",
    default=8,
    type=int,
    help="Number of attention heads in QFormer, default 8",
)
parser.add_argument(
    "--qformer_ffn_ratio",
    default=4.0,
    type=float,
    help="FFN hidden dim ratio relative to sbert_dim in QFormer layers. Default 4.0",
)
parser.add_argument(
    "--prepend_n_tokens",
    default=1,
    type=int,
    help="Asym prepend: number of learnable prefix tokens prepended for Q and for V. Default: 1",
)
parser.add_argument(
    "--mlp_hidden_dim",
    default=None,
    type=int,
    help="Hidden dimension for Q/V MLP heads. Default None (same as sbert embedding dim)",
)
parser.add_argument(
    "--asym_params_lr_scaling",
    default=1.0,
    type=float,
    help="Asym MLP/QFormer: multiply those grads by this before Nadam (higher effective LR on heads). "
         "1.0 = off. Default: 1.0",
)
parser.add_argument(
    "--asym_qvt_l2_reg",
    default="false",
    type=str,
    help="Asym beeformer (GD): add λ·||Q V^T||_F^2 on the batch item block. true/false, default false",
)
parser.add_argument(
    "--asym_qvt_l2_weight",
    default=1e-6,
    type=float,
    help="Weight λ for Q V^T Frobenius penalty when --asym_qvt_l2_reg true. Default: 1e-6",
)

parser.add_argument(
    "--use_l3ae_model",
    default="false",
    type=str,
    help="Use AsymBeeformerL3AE (frozen beeformer + closed-form S + B): true/false, default false",
)

parser.add_argument(
    "--lambda_s",
    default=10.0,
    type=float,
    help="Regularization for semantic S matrix in AsymBeeformerL3AE, default 1.0",
)
parser.add_argument(
    "--lambda_b",
    default=100.0,
    type=float,
    help="L2 regularization for collaborative B matrix in AsymBeeformerL3AE, default 500.0",
)
parser.add_argument(
    "--lambda_r",
    default=150.0,
    type=float,
    help="||B - S|| regularization strength in AsymBeeformerL3AE, default 10.0",
)

# output model name
parser.add_argument(
    "--model_name",
    default="my_model",
    type=str,
    help="Output sentence transformer model name to train, default my_model",
)

# evaluate
parser.add_argument(
    "--evaluate", default="false", type=str, help="final evaluation after training [true/false], default false"
)
parser.add_argument(
    "--evaluate_epoch", default="false", type=str, help="evaluation after every epoch [true/false], default false"
)
parser.add_argument(
    "--save_every_epoch", default="true", type=str, help="save after every epoch [true/false], default true"
)
parser.add_argument(
    "--use_elsa_model",
    default="false",
    type=str,
    help="Train direct ELSA baseline (learn A) without SBERT: true/false, default false",
)
parser.add_argument(
    "--use_asym_elsa_model",
    default="false",
    type=str,
    help="Train direct AsymELSA baseline (learn Q,V) without SBERT: true/false, default false",
)
parser.add_argument(
    "--elsa_dim",
    default=256,
    type=int,
    help="Embedding dimension for direct (Asym)ELSA parameters when training without SBERT, default 256",
)

args = parser.parse_args([] if "__file__" not in globals() else None)
print(args)

# limit visible devces for pytorch
if args.device is not None:
    print(f"Limiting devices to {args.device}")
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.device}"

# force the usage of pytorch backend in keras
os.environ["KERAS_BACKEND"] = "torch"

# now we can finally import modules

import keras
import math
import numpy as np
import subprocess
import time
import torch
from copy import deepcopy

from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from callbacks import evaluateWriter
from config import config
from dataloaders import beeformerDataset, ELSADataset
from add_modules import (add_asym_projection, add_qformer_projection,
                        add_prepend_projection, add_dual_encoder, build_sbert)
from models import (
    NMSEbeeformer,
    AsymNMSEbeeformer,
    AsymBeeformerL3AE,
    L3AE,
    SparseKerasAsymELSA,
    SparseKerasELSA,
)
from schedules import LinearWarmup
from utils import *

import images

torch.set_float32_matmul_precision('medium')

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device {DEVICE}")

def load_data(args):
    if args.validation == "true":
        what = "val"
    else:
        what = "test"

    # read data
    if args.dataset in config.keys():
        dataset, params = config[args.dataset]
        dataset.load_interactions(**params)

        if args.use_time_split == "true":
            evaluator = TimeBasedEvaluation(dataset, what=what)
        elif args.use_cold_start == "true":
            evaluator = ColdStartEvaluation(dataset, what=what)
        else:
            # user-based split strategy (default)
            evaluator = Evaluation(dataset, what=what)

        items_d = dataset.items_texts
        items_d["asin"] = items_d.item_id
        if args.validation == "true":
            _train_interactions = dataset.train_interactions
        else:
            _train_interactions = dataset.full_train_interactions

    elif args.dataset == "goodlens":
        # todo: should be rewritten to combine any two datasets (not as simple as it looks)
        dataset, params = config["ml20m"]
        dataset.load_interactions(**params)

        if args.use_cold_start == "true":
            evaluator = ColdStartEvaluation(dataset, what=what)
        elif args.use_time_split == "true":
            evaluator = TimeBasedEvaluation(dataset, what=what)
        else:
            evaluator = Evaluation(dataset, what=what)

        dataset2, params2 = config["goodbooks"]
        dataset2.load_interactions(**params2)
        if args.use_cold_start == "true":
            # this must be done, because init in evaluator is modifiyng dataset object
            # it also mean that evaluation will be eventually done on movielens
            evaluator2 = ColdStartEvaluation(dataset2)
        else:
            evaluator2 = Evaluation(dataset2)

        # merge the two datasets
        if args.validation == "true":
            df = dataset.train_interactions.copy()
        else:
            df = dataset.full_train_interactions.copy()

        it = dataset.items_texts.copy()
        df["user_id"] = df.user_id.apply(lambda x: "m" + x)
        df["item_id"] = df.item_id.apply(lambda x: "m" + x)
        it["item_id"] = it.item_id.apply(lambda x: "m" + x)

        if args.validation == "true":
            df2 = dataset2.train_interactions.copy()
        else:
            df2 = dataset2.full_train_interactions.copy()

        it2 = dataset2.items_texts.copy()
        df2["user_id"] = df2.user_id.apply(lambda x: "g" + x)
        df2["item_id"] = df2.item_id.apply(lambda x: "g" + x)
        it2["item_id"] = it2.item_id.apply(lambda x: "g" + x)
        _train_interactions = pd.concat([df, df2])
        items_texts = pd.concat([it, it2])
        _train_interactions["item_id"] = _train_interactions["item_id"].astype("category")
        _train_interactions["user_id"] = _train_interactions["user_id"].astype("category")
        items_d = items_texts
        items_d["asin"] = items_d.item_id
    else:
        print("Unknown dataset. List of available datsets: \n")
        for x in config.keys():
            print(x)
        print("goodlens")
        print()
        return None, None, None
    
    return dataset, evaluator, _train_interactions, items_d

def load_text_model(args, items_d, dataset, _train_interactions):
    # load and preprocess text side information
    print("Preprocessing texts.")
    if args.evaluate == "true" or args.evaluate_epoch == "true":
        am_itemids = items_d.asin.to_numpy()
        cc = np.array(dataset.all_interactions.item_id.cat.categories)
        ccdf = pd.Series(cc).to_frame()
        ccdf.columns = ["item_id"]
        amdf = pd.Series(am_itemids).to_frame().reset_index()
        amdf.columns = ["idx", "item_id"]
        am_locator = pd.merge(how="inner", left=ccdf, right=amdf).idx.to_numpy()

        if args.dataset in config.keys():
            am_texts = items_d._text_attributes
        elif args.preproces_html == "true":
            am_texts = items_d.fillna(0).apply(
                lambda row: f"{row.title}: {preproces_html('. '.join(eval(row.description)))}", axis=1
            )
        else:
            print("using html preprocessing")
            am_texts = items_d.fillna(0).apply(lambda row: f"{row.title}: {'. '.join(eval(row.description))}", axis=1)

        am_texts_all = am_texts.to_numpy()[am_locator]  # evaluation texts
    else:
        am_texts_all = None

    am_itemids = items_d.asin.to_numpy()
    cc = np.array(_train_interactions.item_id.cat.categories)
    ccdf = pd.Series(cc).to_frame()
    ccdf.columns = ["item_id"]
    amdf = pd.Series(am_itemids).to_frame().reset_index()
    amdf.columns = ["idx", "item_id"]
    am_locator = pd.merge(how="inner", left=ccdf, right=amdf).idx.to_numpy()
    am_texts = items_d._text_attributes
    am_texts = am_texts.to_numpy()[am_locator]  # training texts

    # for e5 models
    if args.prefix is not None:
        print("adding prefix", args.prefix, "to all texts")
        am_texts = np.array([args.prefix + x for x in am_texts])
        print(am_texts[:10])

    # create sentence Transformer that will be trained
    print("Creating sbert")
    sbert = build_sbert(
        args.sbert, DEVICE,
        asym_type=args.asym_type if args.use_asym_model == "true" else None,
        max_seq_length=args.max_seq_length,
        mlp_hidden_dim=args.mlp_hidden_dim,
        qformer_layers=args.qformer_layers,
        qformer_heads=args.qformer_heads,
        qformer_ffn_ratio=args.qformer_ffn_ratio,
        prepend_n_tokens=args.prepend_n_tokens,
    )

    # tokenize item text side information (descriptions)
    am_tokenized = sbert.tokenize(am_texts)
    if am_texts_all is None:
        am_texts_all = am_texts
        
    return am_texts_all, am_tokenized, sbert

def load_image_model(args, items_d, dataset, _train_interactions):
    image_model = images.ImageModel(args.image_model, device=DEVICE)

    tokenized_images_dict = images.read_images_into_dict(dataset.all_interactions.item_id.cat.categories, fn=image_model.tokenize, path=dataset.images_dir, suffix=dataset.images_suffix)
    tokenized_train_images = images.read_images_from_dict(_train_interactions.item_id.cat.categories, tokenized_images_dict)
    tokenized_test_images = images.read_images_from_dict(dataset.all_interactions.item_id.cat.categories, tokenized_images_dict)

    return tokenized_test_images, tokenized_train_images, image_model

def prepare_schedule(args, steps_per_epoch):
    # prepare training schedule
    if args.scheduler == "CosineDecay":
        schedule = keras.optimizers.schedules.CosineDecay(
            0.0,
            steps_per_epoch * (args.decay_epochs + args.warmup_epochs),
            alpha=0.0,
            name="CosineDecay",
            warmup_target=args.warmup_lr,
            warmup_steps=steps_per_epoch * args.warmup_epochs,
        )
        epochs = args.warmup_epochs + args.decay_epochs + args.tuning_epochs
        print("Using schedule with config", schedule.get_config())
    elif args.scheduler == "LinearWarmup":
        schedule = LinearWarmup(
            warmup_steps=steps_per_epoch * args.warmup_epochs,
            decay_steps=steps_per_epoch * args.decay_epochs,
            starting_lr=args.init_lr,
            warmup_lr=args.warmup_lr,
            final_lr=args.target_lr,
        )
        print("Using schedule with config", schedule.get_config())
        epochs = args.warmup_epochs + args.decay_epochs + args.tuning_epochs
    else:
        schedule = args.lr
        epochs = args.epochs
        print("Using constant learning rate of", schedule)
    
    return schedule, epochs

def _maybe_eval_writer(
    *,
    args,
    logdir: str,
    evaluator,
    items_idx_for_eval,
    sbert,
    am_texts_all,
    is_asym_sbert: bool,
    eval_model = None,
):
    if not (args.evaluate == "true" or args.evaluate_epoch == "true" or args.save_every_epoch == "true"):
        return None
    return evaluateWriter(
        items_idx=items_idx_for_eval,
        sbert=sbert,
        evaluator=evaluator,
        logdir=logdir,
        DEVICE=DEVICE,
        texts=am_texts_all,
        is_asym_sbert=is_asym_sbert,
        sbert_name=args.model_name,
        evaluate_epoch=args.evaluate_epoch,
        save_every_epoch=args.save_every_epoch,
        eval_model=eval_model,
    )


def _train_keras_nmse_loop(
    *,
    model: keras.models.Model,
    datal,
    epochs: int,
    schedule,
    callbacks,
    folder: str,
    preamble: str,
    show_summary: bool,
    save_sbert: bool,
    sbert,
    sbert_save_path,
    model_save_path,
    final_eval_cb,
    final_eval_model= None,
    final_eval_kwargs= None,
    write_nvidia_smi_log = True,
    history_agg = "single",
    fits_for_history = None,
    loss = "nmse"
) -> float:
    model.compile(
        optimizer=keras.optimizers.Nadam(learning_rate=schedule),
        loss=NMSE if loss == "nmse" else cosine_loss,
        metrics=[keras.metrics.CosineSimilarity()],
    )
    print(preamble)
    print('FINAL EVAL MODEL', final_eval_model)
    model.train_step(datal[0])
    model.built = True
    if show_summary:
        model.summary()

    train_time = time.time()
    f = model.fit(datal, epochs=epochs, callbacks=callbacks)
    train_time = time.time() - train_time

    if model_save_path is not None:
        model.save(model_save_path)

    if save_sbert and sbert is not None and sbert_save_path is not None:
        sbert.save(sbert_save_path)

    if args.evaluate == "true" and final_eval_cb is not None:
        kwargs = dict(final_eval_kwargs or {})
        if final_eval_model is not None:
            final_eval_cb.on_epoch_end(epoch="final", eval_model=final_eval_model, **kwargs)
        else:
            final_eval_cb.on_epoch_end(epoch="final", **kwargs)

    fits = None
    if history_agg == "concat_fits":
        fits = (list(fits_for_history) if fits_for_history is not None else []) + [f]
        ks = list(fits[0].history.keys())
        if len(fits) > 1:
            dc = {k: np.array([(h.history[k]) for h in fits]).flatten() for k in ks}
        else:
            dc = {k: np.array(fits[0].history[k]) for k in ks}
    else:
        ks = list(f.history.keys())
        dc = {k: np.array(f.history[k]) for k in ks}
    dc["epoch"] = np.arange(len(dc[list(dc.keys())[0]])) + 1
    df_hist = pd.DataFrame(dc)
    # Match legacy column ordering used in the main Beeformer path.
    if history_agg == "concat_fits" and fits is not None and len(fits) > 1 and len(df_hist.columns) > 1:
        df_hist = df_hist[list(df_hist.columns[-1:]) + list(df_hist.columns[:-1])]
    df_hist.to_csv(f"{folder}/history.csv")

    if final_eval_cb is not None:
        try:
            pd.concat([pd.Series(x).to_frame().T for x in final_eval_cb.results_list]).to_csv(
                f"{folder}/results-history.csv"
            )
        except Exception:
            pass

    pd.Series(train_time).to_csv(f"{folder}/timer.csv")
    print("timer written")

    if write_nvidia_smi_log:
        try:
            out = subprocess.check_output(["nvidia-smi"])
            with open(os.path.join(folder, f"{args.dataset}_{args.flag}.log"), "w") as f_log:
                f_log.write(out.decode("utf-8"))
        except Exception:
            pass

    return float(train_time)



def main(args):
    # prepare logging folder
    folder = os.path.join(
        "results", f"{str(pd.Timestamp('today'))} {9*int(1e6)+np.random.randint(999999)}".replace(" ", "_")
    )
    if not os.path.exists(folder):
        os.makedirs(folder)
    vargs = vars(args)
    vargs["cuda_or_cpu"] = DEVICE
    pd.Series(vargs).to_csv(f"{folder}/setup.csv")
    print(f"Saving results to {folder}")

    # set random seeds for reproducibility
    torch.manual_seed(args.seed)
    keras.utils.set_random_seed(args.seed)
    np.random.seed(args.seed)
    print(f"seeds set to {args.seed}")

    if args.validation == "true":
        what = "val"
    else:
        what = "test"

    # read data

    dataset, evaluator, _train_interactions, items_d = load_data(args)
    
    if dataset is None:
        return

    if args.use_elsa_model == "true" or args.use_asym_elsa_model == "true":
        print("Creating interaction matrix for ELSA training")
        X = get_sparse_matrix_from_dataframe(_train_interactions)
        n_items = X.shape[1]
        items_idx = _train_interactions.item_id.cat.categories
        datal = ELSADataset(
            X,
            DEVICE,
            shuffle=True,
            max_output=args.max_output,
            batch_size=args.batch_size,
        )
        steps_per_epoch = len(datal)
        schedule, epochs = prepare_schedule(args, steps_per_epoch)

        if args.use_asym_elsa_model == "true":
            model = SparseKerasAsymELSA(
                n_items=n_items,
                n_dims=args.elsa_dim,
                items_idx=items_idx,
                device=DEVICE,
                top_k=args.top_k,
            )
        else:
            model = SparseKerasELSA(
                n_items=n_items,
                n_dims=args.elsa_dim,
                items_idx=items_idx,
                device=DEVICE,
                top_k=args.top_k,
            )

        eval_cb = _maybe_eval_writer(
            args=args,
            logdir=folder,
            evaluator=evaluator,
            items_idx_for_eval=items_idx,
            sbert=None,
            am_texts_all=None,
            is_asym_sbert=False,
            eval_model=model,
        )
        cbs = [eval_cb] if eval_cb is not None else []

        model.to(DEVICE)
        _train_keras_nmse_loop(
            model=model,
            datal=datal,
            epochs=epochs,
            schedule=schedule,
            callbacks=cbs,
            folder=folder,
            preamble="Building direct (Asym)ELSA model …",
            show_summary=False,
            save_sbert=False,
            sbert=None,
            sbert_save_path=None,
            model_save_path=args.model_name,
            final_eval_cb=eval_cb,
            final_eval_model=model if args.evaluate == "true" else None,
            write_nvidia_smi_log=False,
            loss = "cosine",
        )
        return

    if args.sbert is not None: 
        # load and preprocess text side information
        am_texts_all, am_tokenized, sbert = load_text_model(args, items_d, dataset, _train_interactions)
    elif args.image_model is not None:
        am_texts_all, am_tokenized, sbert = load_image_model(args, items_d, dataset, _train_interactions)
    else:
        print("Dont know what to train. Please specify the --sbert argument.")

    # training in paralel on multiple gpus
    if args.devices is not None:
        print(f"Will run sbert on devices {args.devices}")
        devices_to_run = eval(args.devices)
        module_sbert = torch.nn.DataParallel(sbert, device_ids=devices_to_run, output_device=devices_to_run[0])
    else:
        module_sbert = sbert

    # create X train
    print("Creating interaction matrix for training")
    X = get_sparse_matrix_from_dataframe(_train_interactions)

    if args.use_l3ae_model == "true":
        if args.use_asym_model == "true":
            model = AsymBeeformerL3AE(
                sbert=module_sbert,
                device=DEVICE,
                tokenized_sentences=am_tokenized,
                items_idx=_train_interactions.item_id.cat.categories,
                lambda_s=args.lambda_s,
                lambda_b=args.lambda_b,
                lambda_r=args.lambda_r,
                sbert_batch_size=args.sbert_batch_size,
            )
        else:
            model = L3AE(
                sbert=module_sbert,
                device=DEVICE,
                tokenized_sentences=am_tokenized,
                items_idx=_train_interactions.item_id.cat.categories,
                lambda_s=args.lambda_s,
                lambda_b=args.lambda_b,
                lambda_r=args.lambda_r,
                sbert_batch_size=args.sbert_batch_size,
            )

        eval_cb = evaluateWriter(
            items_idx=_train_interactions.item_id.cat.categories,
            sbert=sbert,
            evaluator=evaluator,
            logdir=folder,
            DEVICE=DEVICE,
            texts=am_texts_all,
            is_asym_sbert=args.use_asym_model == "true",
            sbert_name=args.model_name,
            evaluate_epoch=args.evaluate,
            save_every_epoch="false",
        )

        train_time = time.time()
        ease_model = model.fit_closed_form(X)
        train_time = time.time() - train_time

        sbert.save(args.model_name)
        ease_model.save(args.model_name)

        if args.evaluate == "true":
            eval_cb.on_epoch_end(epoch="final", eval_model=ease_model)

        pd.Series(train_time).to_csv(f"{folder}/timer.csv")
        out = subprocess.check_output(["nvidia-smi"])
        with open(os.path.join(folder, f"{args.dataset}_{args.flag}.log"), "w") as f:
            f.write(out.decode("utf-8"))
        return

    # prepare dataloader
    print("Creating dataloader")
    datal = beeformerDataset(
        X, am_tokenized, DEVICE, shuffle=True, max_output=args.max_output, batch_size=args.batch_size
    )
    steps_per_epoch = len(datal)

    print(sbert)

    # create trainable keras model
    model_kwargs = dict(
        tokenized_sentences=am_tokenized,
        items_idx=_train_interactions.item_id.cat.categories,
        sbert=module_sbert,
        device=DEVICE,
        top_k=args.top_k,
        sbert_batch_size=args.sbert_batch_size,
    )
    if args.use_asym_model == "true":
        model = AsymNMSEbeeformer(
            **model_kwargs,
            asym_params_lr_scaling=args.asym_params_lr_scaling,
            qvt_l2_reg=(args.asym_qvt_l2_reg == "true"),
            qvt_l2_weight=float(args.asym_qvt_l2_weight),
        )
    else:
        model = NMSEbeeformer(**model_kwargs)

    # prepare lr schedule 
    schedule, epochs = prepare_schedule(args, steps_per_epoch)

    model.to(DEVICE)

    # create callback object to monitor the training procedure
    cbs = []
    if args.evaluate == "true" or args.evaluate_epoch == "true" or args.save_every_epoch == "true":

        train_items = _train_interactions[["user_id", "item_id"]].drop_duplicates()["item_id"].unique()
        test_items = evaluator.test_target.item_id.unique().tolist()
        cold_items = list(set(test_items) - set(train_items))
        cold_items_evaluator = deepcopy(evaluator)
        cold_items_evaluator.test_target = cold_items_evaluator.test_target[cold_items_evaluator.test_target.item_id.isin(cold_items)]
        cold_items_evaluator.test_src = cold_items_evaluator.test_src[cold_items_evaluator.test_src.user_id.isin(cold_items_evaluator.test_target.user_id)]
        print(f"Print num items {len(cold_items)} and users {len(cold_items_evaluator.test_src.user_id.unique())} for coldstart evaluation")

        eval_cb = evaluateWriter(
            items_idx=dataset.all_interactions.item_id.cat.categories,
            sbert=sbert,
            evaluator=evaluator,
            logdir=folder,
            DEVICE=DEVICE,
            texts=am_texts_all,
            is_asym_sbert=(args.use_asym_model == "true"),
            sbert_name=args.model_name,
            evaluate_epoch=args.evaluate_epoch,
            save_every_epoch=args.save_every_epoch,
            coldstart_evaluator=cold_items_evaluator,
        )
        cbs.append(eval_cb)

    # build the model
    model.compile(
        optimizer=keras.optimizers.Nadam(learning_rate=schedule), loss=NMSE, metrics=[keras.metrics.CosineSimilarity()]
    )
    print("Building the model")
    model.train_step(datal[0])
    model.built = True
    model.summary()
    print("Starting training loop")
    train_time = 0

    # training
    fits = []
    # print(f"Training for {args.warmup_epochs+args.decay_epochs+args.tuning_epochs} epochs.")
    f = model.fit(
        datal,
        epochs=epochs,
        callbacks=cbs,
    )
    fits.append(f)
    train_time = time.time() - train_time

    # save resulting model
    sbert.save(args.model_name)

    # final evaluation — reuse the callback that already has all the logic
    if args.evaluate == "true":
        eval_cb.on_epoch_end(epoch="final")

    # final logs
    ks = list(f.history.keys())
    dc = {k: np.array([(f.history[k]) for f in fits]).flatten() for k in ks}
    dc["epoch"] = np.arange(len(dc[list(dc.keys())[0]])) + 1
    df = pd.DataFrame(dc)
    df[list(df.columns[-1:]) + list(df.columns[:-1])]

    df.to_csv(f"{folder}/history.csv")
    print("history file written")

    try:
        pd.concat([pd.Series(x).to_frame().T for x in eval_cb.results_list]).to_csv(f"{folder}/results-history.csv")
    except:
        print("eval_cb not exist")

    pd.Series(train_time).to_csv(f"{folder}/timer.csv")
    print("timer written")

    out = subprocess.check_output(["nvidia-smi"])

    with open(os.path.join(folder, f"{args.dataset}_{args.flag}.log"), "w") as f:
        f.write(out.decode("utf-8"))


if __name__ == "__main__":
    main(args)
