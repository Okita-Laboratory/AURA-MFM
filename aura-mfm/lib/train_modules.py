# Copyright (c) Meta Platforms, Inc. and affiliates.
# LICENSE file in the root directory of this source tree.

import pytorch_lightning as pl
import torch
import torchmetrics
from lib.loss import InfoNCE
import matplotlib.pyplot as plt
import os



class MultimodalContrastiveLearningModule(pl.LightningModule):
    def __init__(self, modality_to_encoder, source_modality, target_modalities, output_dir, learning_rate):
        """
        modality_to_encoder = {
                'imu': imu_encoder,
                'text': text_encoder,
                'video': video_encoder,
                'audio': audio_encoder,
            }
        """
        super().__init__()

        self.source_modality = source_modality
        self.target_modalities = target_modalities
        self.list_modalities = modality_to_encoder.keys()
        self.learning_rate = learning_rate

        self.loss = InfoNCE(symmetric_loss=True, learn_temperature=True)
        self.train_losses = []
        self.val_losses = []
        
        self.train_loss_history = []  # エポック毎のトレーニング損失を保存
        self.val_loss_history = []    # エポック毎のバリデーション損失を保存

        # エポックごとのプロットを保存するディレクトリを作成
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        if "imu" in self.list_modalities:
            self.imu_encoder = modality_to_encoder["imu"]

        if "text" in self.list_modalities:
            self.text_encoder = modality_to_encoder["text"]

        if "video" in self.list_modalities:
            self.video_encoder = modality_to_encoder["video"]

        if "audio" in self.list_modalities:
            self.audio_encoder = modality_to_encoder["audio"]

    def forward(self, batch):
        # x_imu: (batch_size x 6 x window_size)
        # x_narration: [ str ] with len == batch_size
        # y_*: B x size_embeddings
        """
        if len(batch["video"]) != len(batch["narration"]) or len(batch["video"]) != len(batch["imu"]):
            print("Weird!")
            min_size = min(min(len(batch["video"]), len(batch["narration"])), len(batch["imu"]))
            batch["imu"] = batch["imu"][:min_size]
            batch["video"] = batch["video"][:min_size]
            batch["audio"] = batch["audio"][:min_size]
        """

        out = {}

        if "imu" in self.list_modalities:
            x_imu = batch["imu"]
            # print(f"x_imu.dtype:{x_imu.dtype}")
            
            y_imu = self.imu_encoder(x_imu)
            out["imu"] = y_imu
            # print(f"x_imu:{x_imu}, y_imu:{y_imu}")

        if "text" in self.list_modalities:
            x_narration = batch["narration"]
            y_narration = self.text_encoder.get_text_embeddings(x_narration)
            out["text"] = y_narration

        if "video" in self.list_modalities:
            x_video = batch["video"]
            y_video = self.video_encoder.get_video_embeddings(x_video)
            out["video"] = y_video
            # print(f"x_video:{x_video.dtype}, y_video:{y_video.dtype}")

        if "audio" in self.list_modalities:
            x_audio = batch["audio"]
            y_audio = self.audio_encoder(x_audio)
            out["audio"] = y_audio

        return out

    def plot_and_save(self, data, filename, title, ylabel):
        plt.figure()
        plt.plot(range(1, len(data) + 1), data)
        plt.title(title)
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.savefig(os.path.join(self.output_dir, filename))
        plt.close()

    def on_train_epoch_end(self):
        avg_train_loss = torch.stack(self.train_losses).mean().item()  # エポックごとの平均を計算
        self.train_loss_history.append(avg_train_loss)  # 履歴に追加
        self.log("epoch_train_loss", avg_train_loss, logger=True,sync_dist=True)
        self.train_losses.clear()  # 次のエポックのためにリセット

        # トレーニング損失をプロットして保存
        self.plot_and_save(
            data=self.train_loss_history,
            filename="train_loss.png",
            title="Training Loss",
            ylabel="Loss"
        )

    def on_validation_epoch_end(self):
        avg_val_loss = torch.stack(self.val_losses).mean().item()
        self.val_loss_history.append(avg_val_loss)  # 履歴に追加
        self.log("epoch_val_loss", avg_val_loss, logger=True,sync_dist=True)
        self.val_losses.clear()
        print("validation loss: ", avg_val_loss)
        # バリデーション損失をプロットして保存
        self.plot_and_save(
            data=self.val_loss_history[1:],  # 最初のエポックは除外
            filename="val_loss.png",
            title="Validation Loss",
            ylabel="Loss"
        )

    def training_step(self, batch, batch_idx: int):
        loss = self._shared_eval(batch, batch_idx, "train")
        self.train_losses.append(loss.detach())  # ロスをリストに追加
        return loss

    def validation_step(self, batch, batch_idx: int):
        loss = self._shared_eval(batch, batch_idx, "val")
        self.val_losses.append(loss.detach())  # ロスをリストに追加
        return loss
    
    def test_step(self, batch, batch_idx: int):
        # y: {modality[str]: y_*} where y_*: B x size_embeddings
        y = self(batch)

        # Prepare metrics computation
        y_query_modality = y[self.source_modality]
        loss_output = 0.0
        metrics = {}

        # Compute metrics for source modality <> each target modality
        for target_modality in self.target_modalities:
            y_key_modality = y[target_modality]
            s2t_loss = self.loss(query=y_query_modality, positive_key=y_key_modality)
            loss_output += s2t_loss
            s_t_accuracy, t_s_accuracy = evaluate_batch_similarity(y_query_modality, y_key_modality, device=self.device)

            str_s2t = "{source_modality_initial}2{target_modality_initial}".format(
                source_modality_initial=self.source_modality[0],
                target_modality_initial=target_modality[0],
            )
            str_t2s = "{target_modality_initial}2{source_modality_initial}".format(
                target_modality_initial=target_modality[0],
                source_modality_initial=self.source_modality[0],
            )
            metrics[f"{str_s2t}_accuracy"] = s_t_accuracy
            metrics[f"{str_t2s}_accuracy"] = t_s_accuracy

        # Save and log metrics
        metrics["test_loss"] = loss_output
        self.log_dict(metrics, logger=True)
        return metrics

    def predict_step(self, batch, batch_idx: int):
        return self(batch)

    def _shared_eval(self, batch, batch_idx: int, prefix: str):
        # y: {modality[str]: y_*} where y_*: B x size_embeddings
        y = self(batch)

        # Use NCE loss
        y_query_modality = y[self.source_modality]
        loss_output = 0.0

        # Compute loss for source modality <> each target modality
        for target_modality in self.target_modalities:
            y_key_modality = y[target_modality]
            s2t_loss = self.loss(query=y_query_modality, positive_key=y_key_modality)

            # Log the loss
            str_s2t = "{source_modality_initial}2{target_modality_initial}".format(
                source_modality_initial=self.source_modality[0],
                target_modality_initial=target_modality[0],
            )
            # self.log(f"{prefix}_{str_s2t}_loss", s2t_loss, logger=True, sync_dist=True)
            loss_output += s2t_loss

        # self.log(f"{prefix}_loss", loss_output, logger=True, sync_dist=True)
        return loss_output

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


def evaluate_batch_similarity(source_embeddings, target_embeddings, device):
    """
    Given a batch matrix (size B) of paired embeddings,
    measure the accuracy of the predictions by checking nearest the neighbors
    """
    #  Compute similarity
    s = torch.nn.functional.normalize(source_embeddings, dim=1)
    t = torch.nn.functional.normalize(target_embeddings, dim=1)

    # similarities: B x B
    similarities = torch.mm(s, t.transpose(0, 1))

    # pred: 1 x B (ideally [0, 1, 2, 3, ..., B])
    s_t_pred = torch.argmax(similarities, dim=1)
    t_s_pred = torch.argmax(similarities, dim=0)
    B = len(s_t_pred)
    s_t_accuracy = sum(s_t_pred == torch.arange(B, device=device)) / B
    t_s_accuracy = sum(t_s_pred == torch.arange(B, device=device)) / B
    return s_t_accuracy, t_s_accuracy


class ClassificationModule(pl.LightningModule):
    def __init__(self, model):
        """
        Encoder + Head
        """
        super().__init__()

        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.model = model
        self.accuracy_train = torchmetrics.Accuracy(task="multiclass", num_classes=4)
        self.accuracy_valid = torchmetrics.Accuracy(task="multiclass", num_classes=4)
        self.accuracy_test = torchmetrics.Accuracy(task="multiclass", num_classes=4)
        self.f1_test = torchmetrics.F1Score(task="multiclass", num_classes=4, average="macro")

    def forward(self, batch):
        # x_imu: (batch_size x 6 x window_size)
        # x_narration: [ str ] with len == batch_size
        # y_*: B x size_embeddings
        """
        in: batch_size x 6 x window_size
        out: batch_size x 1
        """
        return self.model(batch)

    def training_step(self, batch, batch_idx: int):
        return self._shared_eval(batch, batch_idx, "train")

    def training_epoch_end(self, outs):
        # log epoch metric
        self.log("train_acc_epoch", self.accuracy_train)

    def validation_epoch_end(self, outs):
        # log epoch metric
        self.log("val_acc_epoch", self.accuracy_valid)

    def test_epoch_end(self, outs):
        # log epoch metric
        self.log("test_acc_epoch", self.accuracy_test)
        self.log("test_f1_epoch", self.f1_test)

    def validation_step(self, batch, batch_idx: int):
        return self._shared_eval(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx: int):
        return self._shared_eval(batch, batch_idx, "test")

    def predict_step(self, batch, batch_idx: int):
        return self(batch)

    def _shared_eval(self, batch, batch_idx: int, prefix: str):
        x, y = batch
        y_hat = self(x)
        loss_output = self.loss_fn(y_hat, y)
        if prefix == "train":
            self.accuracy_train(y_hat, y)
            self.log(f"{prefix}_acc_step", self.accuracy_train, logger=True)
        if prefix == "val":
            self.accuracy_valid(y_hat, y)
            self.log(f"{prefix}_acc_step", self.accuracy_valid, logger=True)
        if prefix == "test":
            self.accuracy_test(y_hat, y)
            self.f1_test(y_hat, y)
            self.log(f"{prefix}_acc_step", self.accuracy_test, logger=True)
            self.log(f"{prefix}_f1_step", self.f1_test, logger=True)
        self.log(f"{prefix}_loss", loss_output, logger=True)
        return loss_output

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=5e-4)
