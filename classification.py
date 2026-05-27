import argparse
import gc
import multiprocessing
import os
import pathlib
import pickle
import platform
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy
import pandas as pd
import seaborn as sn
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from tqdm import tqdm

try:
    from lightning.pytorch import Trainer, seed_everything
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
    from lightning.pytorch.loggers import TensorBoardLogger
except ImportError:
    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
    from pytorch_lightning.loggers import TensorBoardLogger
    from pytorch_lightning.utilities.seed import seed_everything

from utils.datasets.assemblygraphs import AssemblyGraphs
from utils.models.models import ClassificationGNN


def parse_devices(devices):
    if isinstance(devices, int):
        return devices
    if devices is None:
        return 1

    devices = str(devices).strip()
    return int(devices) if devices.isdigit() else devices


def get_parser():
    """Obtain argument parser."""
    parser = argparse.ArgumentParser("Classification Model")
    parser.add_argument("traintest", choices=("train", "test"), default="train", help="Whether to train or test")
    parser.add_argument("--dataset_path", type=str, help="Path to dataset")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint file to load weights from")

    # Trainer/runtime arguments.
    parser.add_argument("--accelerator", choices=("auto", "cpu", "gpu"), default="auto")
    parser.add_argument("--devices", type=str, default="1")
    parser.add_argument("--gpus", type=int, default=None, help="Backward-compatible alias for --devices")
    parser.add_argument("--max_epochs", "--max_epoch", dest="max_epochs", type=int, default=100)
    parser.add_argument("--precision", type=str, default="32-true")
    parser.add_argument("--accumulate_grad_batches", type=int, default=32)
    parser.add_argument("--log_every_n_steps", type=int, default=10)

    # Global Parameters (populated on-run and shared across modules)
    parser.add_argument("--experiment_id", type=str)
    parser.add_argument("--node_dim", type=int)
    parser.add_argument("--edge_dim", type=int)
    parser.add_argument("--gnn_type", type=str, default="sage", choices=["sage", "gat", "gin"])
    parser.add_argument("--hid_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=7)
    parser.add_argument("--train_set", type=list)
    parser.add_argument("--val_set", type=list)
    parser.add_argument("--test_set", type=list)
    parser.add_argument("--vocab", type=dict)

    # Customized Parameters (for feature engineering)
    parser.add_argument("--node_drop", type=bool, default=True)
    parser.add_argument("--UV_Net", type=bool, default=True)
    parser.add_argument("--image_fingerprint", type=bool, default=False)
    parser.add_argument("--MVCNN_embedding", type=bool, default=True)
    parser.add_argument("--random_seed", type=int)
    parser.add_argument("--os", type=str)
    parser.add_argument("--single_node_prediction", type=bool, default=True)
    parser.add_argument("--fixed_split", type=bool, default=False)

    return parser.parse_args()


def normalize_runtime_args(args):
    if args.gpus is not None:
        args.accelerator = "gpu"
        args.devices = str(args.gpus)

    args.devices = parse_devices(args.devices)
    args.os = platform.system()
    return args


def save_results_to_csv(args, cf_acc, experiment_id, f1):
    """Organize all experimental results inside a single CSV file for easier comparison."""
    if args.os == "Windows":
        csv_dir = Path("results\\organized_results.csv")
        if not csv_dir.exists():
            df = pd.DataFrame(list(), columns=["Experiment ID", "Random Seed", "Ablation",
                                               "Micro. F1", "Confusion Acc."])
            df.to_csv("results\\organized_results.csv", index=False)

        csv_row = [experiment_id, args.random_seed, args.ablation, f1, cf_acc]
        dataframe = pd.DataFrame([csv_row], columns=["Experiment ID", "Random Seed", "Ablation",
                                                     "Micro. F1", "Confusion Acc."])
        dataframe.to_csv("results\\organized_results.csv", mode="a", header=False, index=False)
    else:
        csv_dir = Path("results/organized_results.csv")
        if not csv_dir.exists():
            df = pd.DataFrame(list(), columns=["Experiment ID", "Random Seed", "Ablation",
                                               "Micro. F1", "Confusion Acc."])
            df.to_csv("results/organized_results.csv", index=False)

        csv_row = [experiment_id, args.random_seed, args.ablation, f1, cf_acc]
        dataframe = pd.DataFrame([csv_row], columns=["Experiment ID", "Random Seed", "Ablation",
                                                     "Micro. F1", "Confusion Acc."])
        dataframe.to_csv("results/organized_results.csv", mode="a", header=False, index=False)


def build_trainer(args, callbacks, logger):
    return Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        max_epochs=args.max_epochs,
        precision=args.precision,
        callbacks=callbacks,
        logger=logger,
        accumulate_grad_batches=args.accumulate_grad_batches,
        log_every_n_steps=args.log_every_n_steps,
    )


def initialization(args):
    """Initialization of the trainer & the dataset."""
    random.seed(args.random_seed)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    results_path = pathlib.Path(__file__).parent.joinpath("results/checkpoints")
    if not results_path.exists():
        results_path.mkdir(parents=True, exist_ok=True)
        os.makedirs("results/confusion_matrices")
        os.makedirs("results/classification_reports")

    month_day = time.strftime("%m%d")
    hour_min_second = time.strftime("%H%M%S")

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=str(results_path.joinpath(month_day + "_" + hour_min_second)),
        filename="best",
        save_last=True,
    )
    early_stop_callback = EarlyStopping(monitor="val_loss", patience=30, mode="min")

    args.experiment_id = str(results_path.joinpath(month_day + "_" + hour_min_second))

    trainer = build_trainer(
        args,
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=TensorBoardLogger(str(results_path), name=month_day + "_" + hour_min_second, version="logger"),
    )

    return trainer, AssemblyGraphs


def get_num_workers(args):
    if args.os == "Windows":
        return 0
    return multiprocessing.cpu_count()


def train_test(args, trainer, dataset_cls):
    """Train & Test."""
    seed_everything(seed=args.random_seed, workers=True)

    train_data = dataset_cls(args, root_dir=args.dataset_path, split="train")
    val_data = dataset_cls(args, root_dir=args.dataset_path, split="val")

    args.node_dim = train_data.node_dim()
    if args.UV_Net:
        args.node_dim += 128
    if args.single_node_prediction:
        if args.node_dropping:
            args.node_dim += 6
        else:
            args.node_dim += 8

    args.edge_dim = train_data.edge_dim()

    train_loader = train_data.get_dataloader(
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=get_num_workers(args),
    )
    val_loader = val_data.get_dataloader(
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=get_num_workers(args),
    )

    experiment_id = args.experiment_id.split("\\")[-1] if args.os == "Windows" else args.experiment_id.split("/")[-1]

    if args.os == "Windows":
        if not os.path.exists(f"results\\checkpoints\\{experiment_id}"):
            os.makedirs(f"results\\checkpoints\\{experiment_id}")

        with open(f"results\\checkpoints\\{experiment_id}\\random_seed.txt", "w") as f:
            f.write(str(args.random_seed))

        with open(f"results\\checkpoints\\{experiment_id}\\vocab.pickle", "wb") as f:
            pickle.dump(args.vocab, f, pickle.HIGHEST_PROTOCOL)

        with open(f"results\\checkpoints\\{experiment_id}\\train_set.txt", "w") as f:
            for assembly in args.train_set:
                f.write(f"{assembly}\n")
        with open(f"results\\checkpoints\\{experiment_id}\\val_set.txt", "w") as f:
            for assembly in args.val_set:
                f.write(f"{assembly}\n")
    else:
        if not os.path.exists(f"results/checkpoints/{experiment_id}"):
            os.makedirs(f"results/checkpoints/{experiment_id}")

        with open(f"results/checkpoints/{experiment_id}/random_seed.txt", "w") as f:
            f.write(str(args.random_seed))

        with open(f"results/checkpoints/{experiment_id}/vocab.pickle", "wb") as f:
            pickle.dump(args.vocab, f, pickle.HIGHEST_PROTOCOL)

        with open(f"results/checkpoints/{experiment_id}/train_set.txt", "w") as f:
            for assembly in args.train_set:
                f.write(f"{assembly}\n")
        with open(f"results/checkpoints/{experiment_id}/val_set.txt", "w") as f:
            for assembly in args.val_set:
                f.write(f"{assembly}\n")

    if args.checkpoint:
        print("Loading from existing checkpoint - continuing previous training")
        model = ClassificationGNN.load_from_checkpoint(args.checkpoint)
    else:
        model = ClassificationGNN(args)

    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    args.checkpoint = args.experiment_id + ("\\best.ckpt" if args.os == "Windows" else "/best.ckpt")
    test(args, dataset_cls)


def test(args, dataset_cls):
    """Test Only."""
    assert args.checkpoint is not None, "Expected the --checkpoint argument to be provided"

    test_data = dataset_cls(args, root_dir=args.dataset_path, split="test")
    test_loader = test_data.get_dataloader(
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=get_num_workers(args),
    )

    args.node_dim = test_data.node_dim()
    if args.UV_Net:
        args.node_dim += 128
    if args.single_node_prediction:
        if args.node_dropping:
            args.node_dim += 6
        else:
            args.node_dim += 8

    args.edge_dim = test_data.edge_dim()

    model = ClassificationGNN.load_from_checkpoint(args.checkpoint)

    predictions, ground_truths = [], []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Inference on test loader"):
            preds, labels = model.test_step(batch, None)
            predictions.append(preds.tolist())
            ground_truths.append(labels.tolist())

    predictions = list(numpy.concatenate(predictions).flat)
    ground_truths = list(numpy.concatenate(ground_truths).flat)

    if args.traintest == "train":
        experiment_id = args.experiment_id.split("\\")[-1] if args.os == "Windows" else args.experiment_id.split("/")[-1]

    print(classification_report(y_pred=predictions, y_true=ground_truths))
    if args.traintest == "train":
        if args.os == "Windows":
            with open(f"results\\classification_reports\\{experiment_id}.txt", "w") as f:
                f.write(str(classification_report(y_pred=predictions, y_true=ground_truths)))
            with open(f"results\\checkpoints\\{experiment_id}\\test_set.txt", "w") as f:
                for assembly in args.test_set:
                    f.write(f"{assembly}\n")
        else:
            with open(f"results/classification_reports/{experiment_id}.txt", "w") as f:
                f.write(str(classification_report(y_pred=predictions, y_true=ground_truths)))
            with open(f"results/checkpoints/{experiment_id}/test_set.txt", "w") as f:
                for assembly in args.test_set:
                    f.write(f"{assembly}\n")
    else:
        with open("test_classification_report.txt", "w") as f:
            f.write(str(classification_report(y_pred=predictions, y_true=ground_truths)))

    cf = confusion_matrix(y_pred=predictions, y_true=ground_truths, normalize="true")
    cf_acc = round(sum(cf.diagonal() / cf.sum(axis=1)) / len(cf), 3)
    print("Confusion Acc = ", cf_acc)

    plt.figure(figsize=(24, 18))
    if args.node_dropping:
        label = ["Metal_Aluminum", "Metal_Ferrous", "Metal_Non-Ferrous", "Other", "Plastic", "Wood"]
    else:
        label = ["Metal_Aluminum", "Metal_Ferrous", "Metal_Ferrous_Steel", "Metal_Non-Ferrous",
                 "Other", "Paint", "Plastic", "Wood"]

    sn.heatmap(cf, annot=True, fmt=".2f", cmap="Blues", xticklabels=label, yticklabels=label,
               annot_kws={"size": 25})
    plt.xticks(size="xx-large", rotation=45)
    plt.yticks(size="xx-large", rotation=45)
    plt.tight_layout()

    if args.traintest == "train":
        if args.os == "Windows":
            plt.savefig(fname=f"results\\confusion_matrices\\{experiment_id}.png", format="png")
            plt.savefig(fname=f"results\\confusion_matrices\\{experiment_id}.pdf", format="pdf")
        else:
            plt.savefig(fname=f"results/confusion_matrices/{experiment_id}.png", format="png")
            plt.savefig(fname=f"results/confusion_matrices/{experiment_id}.pdf", format="pdf")
    else:
        plt.savefig(fname="test_confusion_matrix.png", format="png")
        plt.savefig(fname="test_confusion_matrix.pdf", format="pdf")

    if args.traintest == "train":
        f1 = round(f1_score(y_true=ground_truths, y_pred=predictions, average="micro", zero_division=0), 3)
        save_results_to_csv(args, cf_acc, experiment_id, f1)


if __name__ == "__main__":
    args = normalize_runtime_args(get_parser())

    # Model settings
    args.node_dropping = True
    args.UV_Net = True
    args.single_node_prediction = False
    args.fixed_split = True

    # Feature engineering settings
    args.image_fingerprint = False
    args.MVCNN_embedding = True

    ablations = [["body_name", "occ_name"]]

    for ablation in ablations:
        args.ablation = ablation

        if args.random_seed is None:
            args.random_seed = random.randint(0, 99999999)
            print(f"[Note] Generated new random seed = {args.random_seed}")
        else:
            print(f"[Note] Using existing random seed = {args.random_seed}")

        if args.traintest == "train":
            trainer, dataset_cls = initialization(args)
            train_test(args, trainer, dataset_cls)
        else:
            random.seed(args.random_seed)
            seed_everything(seed=args.random_seed, workers=True)
            test(args, AssemblyGraphs)
