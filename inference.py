import argparse
import os
import pickle
import platform
import random

import numpy
import torch
from tqdm import tqdm

try:
    from lightning.pytorch import seed_everything
except ImportError:
    from pytorch_lightning.utilities.seed import seed_everything

from utils.datasets.assemblygraphs import AssemblyGraphs
from utils.models.models import ClassificationGNN


def get_parser():
    """Obtain argument parser."""
    parser = argparse.ArgumentParser("UV-Net solid model classification")
    parser.add_argument("type", choices=("single_sample", "multiple_sample"), default="multiple",
                        help="single or multiple inference")
    parser.add_argument("--inference_sample", type=str, help="Path to assembly sample to perform inference")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint file to load weights from")
    parser.add_argument("--vocab", type=str, default=None, help="Vocab pickle file to load one-hot encoding metrics")

    parser.add_argument("--node_dim", type=int)
    parser.add_argument("--edge_dim", type=int)
    parser.add_argument("--gnn_type", type=str, default="sage", choices=["sage"])
    parser.add_argument("--hid_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=7)
    parser.add_argument("--ablation", type=list, default=[])

    parser.add_argument("--node_drop", type=bool, default=True)
    parser.add_argument("--UV_Net", type=bool, default=True)
    parser.add_argument("--image_fingerprint", type=bool, default=False)
    parser.add_argument("--MVCNN_embedding", type=bool, default=False)
    parser.add_argument("--random_seed", type=int)
    parser.add_argument("--os", type=str)
    parser.add_argument("--single_node_prediction", type=bool, default=True)

    return parser.parse_args()


def load_trusted_classification_checkpoint(checkpoint_path):
    """Load a trusted Lightning checkpoint across PyTorch >= 2.6."""
    if hasattr(torch.serialization, "add_safe_globals"):
        torch.serialization.add_safe_globals([argparse.Namespace])

    try:
        return ClassificationGNN.load_from_checkpoint(checkpoint_path, weights_only=False)
    except TypeError:
        return ClassificationGNN.load_from_checkpoint(checkpoint_path)


def inference(args):
    """Inference on one sample."""
    assert args.checkpoint is not None, "Expected the --checkpoint argument to be provided"
    assert args.inference_sample is not None, "Expected the --inference_sample argument to be provided"
    assert args.vocab is not None, "Expected the --vocab argument to be provided"

    model = load_trusted_classification_checkpoint(args.checkpoint)

    with open(args.vocab, "rb") as f:
        args.vocab = pickle.load(f)

    if args.type == "single_sample":
        print("Performing inference on a single assembly sample - all node prediction for the sample")
        assert len(os.listdir(args.inference_sample)) == 1

        inference_data = AssemblyGraphs(args, root_dir=args.inference_sample, split="inference")
        inference_loader = inference_data.get_dataloader(batch_size=1, shuffle=False)

        model.eval()
        predictions, truths, all_body_ids = [], [], []
        correct, all_count = 0, 0

        for batch in inference_loader:
            num_nodes = batch["assembly_graph"].x.shape[0]

        for i in tqdm(range(num_nodes), desc="Inference on nodes of the sample assembly"):
            with torch.no_grad():
                for batch in inference_loader:
                    body_ids = batch["body_ids"][i]
                    preds, labels = model.inference_step(batch, i)
                    predictions.append(preds.tolist())
                    truths.append(labels.tolist())
                    all_body_ids.append(body_ids)

                    all_count += 1
                    if preds.tolist() == labels.tolist():
                        correct += 1

        predictions = list(numpy.concatenate(predictions).flat)
        truths = list(numpy.concatenate(truths).flat)

        print("Body IDs:", all_body_ids)
        print("Ground Truths:", truths)
        print("Predictions:", predictions)
        print(f"Inference accuracy = {round(correct / all_count, 2)}")
    else:
        print("Performing inference on multiple assembly samples - one node prediction per sample")

        inference_data = AssemblyGraphs(args, root_dir=args.inference_sample, split="inference")
        inference_loader = inference_data.get_dataloader(batch_size=16, shuffle=False)

        model.eval()
        predictions, truths = [], []
        correct, all_count = 0, 0

        for batch in tqdm(inference_loader, desc="Inference on sample assemblies"):
            preds, labels = model.test_step(batch, None)
            predictions.append(preds.tolist())
            truths.append(labels.tolist())

        predictions = list(numpy.concatenate(predictions).flat)
        truths = list(numpy.concatenate(truths).flat)

        for pred, truth in zip(predictions, truths):
            all_count += 1
            if pred == truth:
                correct += 1

        print(predictions)
        print(truths)
        print(f"Inference accuracy = {round(correct / all_count, 2)}")


if __name__ == "__main__":
    args = get_parser()
    args.os = platform.system()

    args.node_dropping = True
    args.UV_Net = True
    args.single_node_prediction = True

    args.image_fingerprint = False
    args.MVCNN_embedding = True
    args.ablation = ["body_name", "occ_name"]

    if args.random_seed is None:
        args.random_seed = random.randint(0, 99999999)
        print(f"[Note] Generated NEW random seed = {args.random_seed}")
    else:
        print(f"[Note] Using EXISTING random seed = {args.random_seed}")

    random.seed(args.random_seed)
    seed_everything(seed=args.random_seed, workers=True)
    inference(args)
