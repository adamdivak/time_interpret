import multiprocessing as mp
import numpy as np
import random
import torch as th
import torch.nn as nn
import os

from argparse import ArgumentParser
from captum.attr import DeepLift, GradientShap, IntegratedGradients, Lime
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from typing import List

import pickle as pkl

from tint.attr import (
    DynaMask,
    ExtremalMask,
    Fit,
    Retain,
    TemporalAugmentedOcclusion,
    TemporalOcclusion,
    TimeForwardTunnel,
)
from tint.attr.models import (
    ExtremalMaskNet,
    JointFeatureGeneratorNet,
    MaskNet,
    RetainNet,
)
from tint.datasets import HMM
from tint.metrics.white_box import (
    aup,
    aur,
    information,
    entropy,
    roc_auc,
    auprc,
)
from tint.models import MLP, RNN


from classifier import StateClassifierNet


def main(
    explainers: List[str],
    device: str = "cpu",
    fold: int = 0,
    seed: int = 42,
    deterministic: bool = False,
    lambda_1: float = 1.0,
    lambda_2: float = 1.0,
    output_file: str = "results.csv",
):
    # If deterministic, seed everything
    if deterministic:
        seed_everything(seed=seed, workers=True)

    # Get accelerator and device
    accelerator = device.split(":")[0]
    device_id = 1
    if len(device.split(":")) > 1:
        device_id = [int(device.split(":")[1])]

    # Create lock
    lock = mp.Lock()

    # Load data
    hmm = HMM(n_folds=5, fold=fold, seed=seed)
    hmm.prepare_data()

    print(f"Training classifier..")

    # Create classifier
    classifier = StateClassifierNet(
        feature_size=3,
        n_state=2,
        hidden_size=200,
        regres=True,
        loss="cross_entropy",
        lr=0.0001,
        l2=1e-3,
    )

    if args.classifier_checkpoint and not os.path.exists(args.classifier_checkpoint):
        print(f"Classifier checkpoint specified, but does not exist, re-training the classifier.")
        args.classifier_checkpoint = ""
    if args.classifier_checkpoint:
        classifier = StateClassifierNet.load_from_checkpoint(args.classifier_checkpoint)
        print(f"..pre-trained classifier loaded from {args.classifier_checkpoint}")
    else:
        # Train classifier
        trainer = Trainer(
            max_epochs=50,
            accelerator=accelerator,
            devices=device_id,
            deterministic=deterministic,
            enable_checkpointing=True,
            check_val_every_n_epoch=10,
            default_root_dir="HMM_classifier",
            logger=TensorBoardLogger(
                save_dir=".",
                name="HMM_classifier",
            ),
        )
        trainer.fit(classifier, datamodule=hmm)

        print(f"..training classifier finished.")

    # Get data for explainers
    with lock:
        x_train = hmm.preprocess(split="train")["x"].to(device)
        x_test = hmm.preprocess(split="test")["x"].to(device)
        y_test = hmm.preprocess(split="test")["y"].to(device)
        true_saliency = hmm.true_saliency(split="test").to(device)

    # Save classifier predictions for debug plotting
    y_hat_test = classifier.forward(x_test, return_all=True).argmax(dim=-1)
    with open(hmm.data_dir + "/classifier_predictions_test.npz", "wb") as fp:
        pkl.dump(obj=y_hat_test, file=fp)

    # Switch to eval
    classifier.eval()

    # Set model to device
    classifier.to(device)

    # Disable cudnn if using cuda accelerator.
    # Please see https://captum.ai/docs/faq#how-can-i-resolve-cudnn-rnn-backward-error-for-rnn-or-lstm-network
    # for more information.
    if accelerator == "cuda":
        th.backends.cudnn.enabled = False

    # Create dict of attributions
    attr = {}
    # perturbed_signals = {}

    if "deep_lift" in explainers:
        explainer = TimeForwardTunnel(DeepLift(classifier))
        attr["deep_lift"] = explainer.attribute(
            x_test,
            baselines=x_test * 0,
            task="binary",
            show_progress=True,
        ).abs()

    if "dyna_mask" in explainers:
        print(f"Training dynamask..")
        trainer = Trainer(
            max_epochs=1000,
            accelerator=accelerator,
            devices=device_id,
            log_every_n_steps=2,
            deterministic=deterministic,
            logger=TensorBoardLogger(
                save_dir=".",
                name="dyna_mask",
            ),
        )
        mask = MaskNet(
            forward_func=classifier,
            perturbation="gaussian_blur",
            sigma_max=1,
            keep_ratio=list(np.arange(0.25, 0.35, 0.01)),
            size_reg_factor_init=0.1,
            size_reg_factor_dilation=100,
            time_reg_factor=1.0,
        )
        explainer = DynaMask(classifier)
        _attr = explainer.attribute(
            x_test,
            additional_forward_args=(True,),
            trainer=trainer,
            mask_net=mask,
            batch_size=100,
            return_best_ratio=True,
        )
        print(f"Best keep ratio is {_attr[1]}")
        attr["dyna_mask"] = _attr[0].to(device)

    if "extremal_mask_preservation" in explainers:
        print(f"Training extremal_mask (preservation)..")
        trainer = Trainer(
            max_epochs=500,
            accelerator=accelerator,
            devices=device_id,
            log_every_n_steps=2,
            check_val_every_n_epoch=20,
            deterministic=deterministic,
            enable_checkpointing=True,
            default_root_dir="extremal_mask_preservation",
            logger=TensorBoardLogger(
                save_dir=".",
                name="HMM_extremal_mask_explainer",
                # version=random.getrandbits(128),
            ),
        )
        mask = ExtremalMaskNet(
            forward_func=classifier,
            model=nn.Sequential(
                RNN(
                    input_size=x_test.shape[-1],
                    rnn="gru",
                    hidden_size=x_test.shape[-1],
                    bidirectional=True,
                ),
                MLP([2 * x_test.shape[-1], x_test.shape[-1]]),
            ),
            lambda_1=lambda_1,
            lambda_2=lambda_2,
            optim="adam",
            lr=0.01,
        )
        explainer = ExtremalMask(classifier)
        _attr = explainer.attribute(
            x_test,
            additional_forward_args=(True,),
            trainer=trainer,
            mask_net=mask,
            batch_size=100,
        )
        attr["extremal_mask_preservation"] = _attr.to(device)
        # perturbed_signals["extremal_mask_preservation"] = mask.net.perturbed_signal

    if "extremal_mask_deletion" in explainers:
        print(f"Training extremal_mask (deletion)..")
        trainer = Trainer(
            max_epochs=500,
            accelerator=accelerator,
            devices=device_id,
            log_every_n_steps=2,
            check_val_every_n_epoch=20,
            deterministic=deterministic,
            enable_checkpointing=True,
            logger=TensorBoardLogger(
                save_dir=".",
                name="HMM_extremal_mask_explainer_deletion",
                # version=random.getrandbits(128),
            ),
        )
        mask = ExtremalMaskNet(
            forward_func=classifier,
            model=nn.Sequential(
                RNN(
                    input_size=x_test.shape[-1],
                    rnn="gru",
                    hidden_size=x_test.shape[-1],
                    bidirectional=True,
                ),
                MLP([2 * x_test.shape[-1], x_test.shape[-1]]),
            ),
            lambda_1=lambda_1,
            lambda_2=lambda_2,
            optim="adam",
            lr=0.01,
            preservation_mode=False
        )
        explainer = ExtremalMask(classifier)
        _attr = explainer.attribute(
            x_test,
            additional_forward_args=(True,),
            trainer=trainer,
            mask_net=mask,
            batch_size=100,
        )
        attr["extremal_mask_deletion"] = 1 - _attr.to(device)  # the returned mask by deletion is flipped
        # perturbed_signals["extremal_mask_deletion"] = mask.net.perturbed_signal

    if "fit" in explainers:
        try:
            generator = JointFeatureGeneratorNet(rnn_hidden_size=6)
            trainer = Trainer(
                max_epochs=300,
                accelerator=accelerator,
                devices=device_id,
                log_every_n_steps=10,
                deterministic=deterministic,
                logger=TensorBoardLogger(
                    save_dir=".",
                    version=random.getrandbits(128),
                ),
            )
            explainer = Fit(
                classifier,
                generator=generator,
                datamodule=hmm,
                trainer=trainer,
            )
            attr["fit"] = explainer.attribute(x_test, show_progress=True)
        except Exception as e:
            # In some cases thi explainer failed to run properly
            print(f"Failed to run the Fit explainer, excluding it")

    if "gradient_shap" in explainers:
        explainer = TimeForwardTunnel(GradientShap(classifier.cpu()))
        attr["gradient_shap"] = explainer.attribute(
            x_test.cpu(),
            baselines=th.cat([x_test.cpu() * 0, x_test.cpu()]),
            n_samples=50,
            stdevs=0.0001,
            task="binary",
            show_progress=True,
        ).abs()
        classifier.to(device)

    if "integrated_gradients" in explainers:
        explainer = TimeForwardTunnel(IntegratedGradients(classifier))
        attr["integrated_gradients"] = explainer.attribute(
            x_test,
            baselines=x_test * 0,
            internal_batch_size=200,
            task="binary",
            show_progress=True,
        ).abs()

    if "lime" in explainers:
        explainer = TimeForwardTunnel(Lime(classifier))
        attr["lime"] = explainer.attribute(
            x_test,
            task="binary",
            show_progress=True,
        ).abs()

    if "augmented_occlusion" in explainers:
        explainer = TimeForwardTunnel(
            TemporalAugmentedOcclusion(
                classifier, data=x_train, n_sampling=10, is_temporal=True
            )
        )
        attr["augmented_occlusion"] = explainer.attribute(
            x_test,
            sliding_window_shapes=(1,),
            attributions_fn=abs,
            task="binary",
            show_progress=True,
        ).abs()

    if "occlusion" in explainers:
        explainer = TimeForwardTunnel(TemporalOcclusion(classifier))
        attr["occlusion"] = explainer.attribute(
            x_test,
            sliding_window_shapes=(1,),
            baselines=x_train.mean(0, keepdim=True),
            attributions_fn=abs,
            task="binary",
            show_progress=True,
        ).abs()

    if "retain" in explainers:
        retain = RetainNet(
            dim_emb=128,
            dropout_emb=0.4,
            dim_alpha=8,
            dim_beta=8,
            dropout_context=0.4,
            dim_output=2,
            loss="cross_entropy",
        )
        explainer = Retain(
            datamodule=hmm,
            retain=retain,
            trainer=Trainer(
                max_epochs=50,
                accelerator=accelerator,
                devices=device_id,
                deterministic=deterministic,
                logger=TensorBoardLogger(
                    save_dir=".",
                    version=random.getrandbits(128),
                ),
            ),
        )
        attr["retain"] = (
            explainer.attribute(x_test, target=y_test).abs().to(device)
        )

    # Save all explanation for debug plotting
    with open(hmm.data_dir + "/hmm_saliency.npz", "wb") as fp:
        attr_numpy = {k: v.detach().numpy() for k, v in attr.items()}
        pkl.dump(attr_numpy, fp)
    # with open(hmm.data_dir + "/hmm_x_test_perturbed_signals.npz", "wb") as fp:
    #     perturbed_signals_numpy = {k: v.detach().numpy() for k, v in perturbed_signals.items()}
    #     pkl.dump(obj=perturbed_signals_numpy, file=fp)

    with open(output_file, "a") as fp, lock:
        for k, v in attr.items():
            fp.write(str(seed) + ",")
            fp.write(str(fold) + ",")
            fp.write(k + ",")
            fp.write(str(lambda_1) + ",")
            fp.write(str(lambda_2) + ",")
            fp.write(f"{aup(v, true_saliency):.4},")
            fp.write(f"{aur(v, true_saliency):.4},")
            fp.write(f"{information(v, true_saliency):.4},")
            fp.write(f"{entropy(v, true_saliency):.4},")
            fp.write(f"{roc_auc(v, true_saliency):.4},")
            fp.write(f"{auprc(v, true_saliency):.4}")
            fp.write("\n")


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--explainers",
        type=str,
        default=[
            "deep_lift",
            "dyna_mask",
            "extremal_mask_preservation",
            "extremal_mask_deletion",
            "fit",
            "gradient_shap",
            "integrated_gradients",
            "lime",
            "augmented_occlusion",
            "occlusion",
            "retain",
        ],
        nargs="+",
        metavar="N",
        help="List of explainer to use.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Which device to use.",
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=0,
        help="Fold of the cross-validation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for data generation.",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Whether to make training deterministic or not.",
    )
    parser.add_argument(
        "--lambda-1",
        type=float,
        default=1.0,
        help="Lambda 1 hyperparameter.",
    )
    parser.add_argument(
        "--lambda-2",
        type=float,
        default=1.0,
        help="Lambda 2 hyperparameter.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="results.csv",
        help="Where to save the results.",
    )
    parser.add_argument(
        "--classifier_checkpoint",
        type=str,
        default="HMM_classifier/version_47/checkpoints/epoch=49-step=1000.ckpt",
        help="A checkpoint to load the classifier from instead of re-training.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        explainers=args.explainers,
        device=args.device,
        fold=args.fold,
        seed=args.seed,
        deterministic=args.deterministic,
        lambda_1=args.lambda_1,
        lambda_2=args.lambda_2,
        output_file=args.output_file,
    )
