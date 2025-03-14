import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch


def generate_plots(list_of_dirs, legend_names, save_path):
    """Generate plots according to log
    :param list_of_dirs: List of paths to log directories
    :param legend_names: List of legend names
    :param save_path: Path to save the figs
    """
    assert len(list_of_dirs) == len(
        legend_names
    ), "Names and log directories must have same length"
    data = {}
    for logdir, name in zip(list_of_dirs, legend_names):
        json_path = os.path.join(logdir, "results.json")
        assert os.path.exists(
            os.path.join(logdir, "results.json")
        ), f"No json file in {logdir}"
        with open(json_path, "r") as f:
            data[name] = json.load(f)

    for yaxis in ["train_accs", "valid_accs", "train_losses", "valid_losses"]:
        fig, ax = plt.subplots()
        for name in data:
            ax.plot(data[name][yaxis], label=name)
        ax.legend()
        ax.set_xlabel("epochs")
        ax.set_ylabel(yaxis.replace("_", " "))
        fig.savefig(os.path.join(save_path, f"{yaxis}.png"))


def seed_experiment(seed):
    """Seed the pseudorandom number generator, for repeatability.

    Args:
        seed (int): random seed
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True


def to_device(tensors, device):
    if isinstance(tensors, torch.Tensor):
        return tensors.to(device=device)
    elif isinstance(tensors, dict):
        return dict(
            (key, to_device(tensor, device)) for (key, tensor) in tensors.items()
        )
    elif isinstance(tensors, list):
        return list((to_device(tensors[0], device), to_device(tensors[1], device)))
    else:
        raise NotImplementedError("Unknown type {0}".format(type(tensors)))


def cross_entropy_loss(logits: torch.Tensor, labels: torch.Tensor):
    """Return the mean loss for this batch
    :param logits: [batch_size, num_class]
    :param labels: [batch_size]
    :return loss
    """

    # first one hot encode labels -> size [batch_size, num_classes]
    num_classes = logits.size(-1)
    labels = torch.nn.functional.one_hot(labels, num_classes=num_classes)

    # compute log softmax of logits -> size [batch_size, num_classes]
    # logits = torch.log(torch.exp(logits) / torch.exp(logits).sum(-1, keepdim=True))
    logits_normalized = logits - torch.max(logits, dim=-1, keepdim=True)[0]
    logits = logits_normalized - torch.log(
        torch.sum(torch.exp(logits_normalized), dim=-1, keepdim=True)
    )

    # now take elementwise product, and then sum over class axis
    return -(logits * labels).sum(dim=-1).mean()


def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor):
    """Compute the accuracy of the batch"""
    acc = (logits.argmax(dim=1) == labels).float().mean()
    return acc
