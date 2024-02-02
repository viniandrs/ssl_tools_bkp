#!/usr/bin/env python

# TODO: A way of removing the need to add the path to the root of
# the project
import sys
from jsonargparse import CLI
import lightning as L
import torch

sys.path.append("../../../")


from ssl_tools.experiments import SSLTrain, SSLTest
from ssl_tools.models.ssl.tnc import build_tnc
from ssl_tools.data.data_modules import (
    TNCHARDataModule,
    MultiModalHARSeriesDataModule,
)
from torchmetrics import Accuracy
from ssl_tools.models.ssl.classifier import SSLDiscriminator
from ssl_tools.models.ssl.tnc import TNCHead


class TNCTrain(SSLTrain):
    _MODEL_NAME = "TNC"

    def __init__(
        self,
        data: str,
        encoding_size: int = 10,
        in_channel: int = 6,
        window_size: int = 60,
        mc_sample_size: int = 20,
        w: float = 0.05,
        significance_level: float = 0.01,
        repeat: int = 5,
        pad_length: bool = True,
        num_classes: int = 6,
        update_backbone: bool = False,
        *args,
        **kwargs,
    ):
        """Trains the temporal neighbourhood coding (TNC) model

        Parameters
        ----------
        encoding_size : int, optional
            Size of the encoding (output of the linear layer)
        in_channel : int, optional
            Number of channels in the input data
        window_size : int, optional
            Size of the input windows (X_t) to be fed to the encoder
        mc_sample_size : bool, optional
            Number of pairs of samples to be generated for each sample in the batch
        w : float, optional
            Unlabeled data correction factor to compensate sampling bias // NEED TO CHECK
        significance_level : float, optional

        repeat : int, optional

        pad_length : bool, optional
            If True, the samples are padded to the length of the longest sample
            in the dataset.
        num_classes : int, optional
            Number of classes in the dataset. Only used in finetune mode.
        update_backbone : bool, optional
            If True, the backbone will be updated during training. Only used in
            finetune mode.
        """
        super().__init__(*args, **kwargs)
        self.data = data
        self.encoding_size = encoding_size
        self.in_channel = in_channel
        self.window_size = window_size
        self.mc_sample_size = mc_sample_size
        self.w = w
        self.significance_level = significance_level
        self.repeat = repeat
        self.pad_length = pad_length
        self.num_classes = num_classes
        self.update_backbone = update_backbone

    def _get_pretrain_model(self) -> L.LightningModule:
        model = build_tnc(
            encoding_size=self.encoding_size,
            in_channel=self.in_channel,
            mc_sample_size=self.mc_sample_size,
            w=self.w,
            learning_rate=self.learning_rate,
        )
        return model

    def _get_pretrain_data_module(self) -> L.LightningDataModule:
        data_module = TNCHARDataModule(
            self.data,
            pad=self.pad_length,
            window_size=self.window_size,
            mc_sample_size=self.mc_sample_size,
            significance_level=self.significance_level,
            repeat=self.repeat,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
        return data_module

    def _get_finetune_model(
        self, load_backbone: str = None
    ) -> L.LightningModule:
        model = self._get_pretrain_model()

        if load_backbone is not None:
            self._load_model(model, load_backbone)

        classifier = TNCHead(
            input_size=self.encoding_size,
            n_classes=self.num_classes,
        )

        task = "multiclass" if self.num_classes > 2 else "binary"
        model = SSLDiscriminator(
            backbone=model,
            head=classifier,
            loss_fn=torch.nn.CrossEntropyLoss(),
            learning_rate=self.learning_rate,
            metrics={"acc": Accuracy(task=task, num_classes=self.num_classes)},
            update_backbone=self.update_backbone,
        )
        return model

    def _get_finetune_data_module(self) -> L.LightningDataModule:
        data_module = MultiModalHARSeriesDataModule(
            self.data,
            batch_size=self.batch_size,
            label="standard activity code",
            features_as_channels=True,
        )

        return data_module


class TNCTest(SSLTest):
    _MODEL_NAME = "TNC"

    def __init__(
        self,
        data: str,
        encoding_size: int = 10,
        in_channel: int = 6,
        window_size: int = 60,
        mc_sample_size: int = 20,
        w: float = 0.05,
        num_classes: int = 6,
        *args,
        **kwargs,
    ):
        """Trains the constrastive predictive coding model

        Parameters
        ----------
        encoding_size : int, optional
            Size of the encoding (output of the linear layer)
        in_channel : int, optional
            Number of channels in the input data
        window_size : int, optional
            Size of the input windows (X_t) to be fed to the encoder
        pad_length : bool, optional
            If True, the samples are padded to the length of the longest sample
            in the dataset.
        """
        super().__init__(*args, **kwargs)
        self.data = data
        self.encoding_size = encoding_size
        self.in_channel = in_channel
        self.window_size = window_size
        self.mc_sample_size = mc_sample_size
        self.w = w
        self.num_classes = num_classes

    def _get_test_model(self) -> L.LightningModule:
        model = build_tnc(
            encoding_size=self.encoding_size,
            in_channel=self.in_channel,
            mc_sample_size=self.mc_sample_size,
            w=self.w,
        )
        classifier = TNCHead(
            input_size=self.encoding_size,
            n_classes=self.num_classes,
        )

        task = "multiclass" if self.num_classes > 2 else "binary"
        model = SSLDiscriminator(
            backbone=model,
            head=classifier,
            loss_fn=torch.nn.CrossEntropyLoss(),
            metrics={"acc": Accuracy(task=task, num_classes=self.num_classes)},
        )
        return model

    def _get_test_data_module(self) -> L.LightningDataModule:
        data_module = MultiModalHARSeriesDataModule(
            self.data,
            batch_size=self.batch_size,
            label="standard activity code",
            features_as_channels=True,
        )

        return data_module


def main():
    components = {
        "fit": TNCTrain,
        "test": TNCTest,
    }
    CLI(components=components, as_positional=False)()


if __name__ == "__main__":
    main()
