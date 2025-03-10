import zipfile
from typing import List
import editdistance
import json
import pytorch_lightning as pl
import torch
import torch.optim as optim
import torch.nn.functional as F # added 
from torch import FloatTensor, LongTensor

from tamer.datamodule import Batch, vocab
from tamer.model.tamer import TAMER
from tamer.utils.utils import (
    ExpRateRecorder, Hypothesis, ce_loss, to_bi_tgt_out, to_struct_output)

import random


class LitTAMER(pl.LightningModule):
    def __init__(
        self,
        d_model: int,
        # encoder
        growth_rate: int,
        num_layers: int,
        # decoder
        nhead: int,
        num_decoder_layers: int,
        dim_feedforward: int,
        dropout: float,
        dc: int,
        cross_coverage: bool,
        self_coverage: bool,
        # beam search
        beam_size: int,
        max_len: int,
        alpha: float,
        early_stopping: bool,
        temperature: float,
        # training
        learning_rate: float,
        patience: int,
        milestones: List[int] = [40, 55],
        vocab_size: int = 113,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.tamer_model = TAMER(
            d_model=d_model,
            growth_rate=growth_rate,
            num_layers=num_layers,
            nhead=nhead,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            dc=dc,
            cross_coverage=cross_coverage,
            self_coverage=self_coverage,
            vocab_size=vocab_size,
        )

        self.exprate_recorder = ExpRateRecorder()

    def forward(
        self, img: FloatTensor, img_mask: LongTensor, tgt: LongTensor,
        tgt_key_padding_mask: torch.BoolTensor = None
    ) -> FloatTensor:
        return self.tamer_model(
        img, img_mask, tgt, tgt_key_padding_mask=tgt_key_padding_mask
        )

    
    # Original Implementation:
    # -----------------------

    # def training_step(self, batch: Batch, _):
    #     tgt, out = to_bi_tgt_out(batch.indices, self.device)
    #     struct_out, _ = to_struct_output(batch.indices, self.device)
    #     out_hat, sim = self(batch.imgs, batch.mask, tgt)

    #     loss = ce_loss(out_hat, out)
    #     self.log("train_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
    #     struct_loss = ce_loss(sim, struct_out, ignore_idx=-1)
    #     self.log(
    #         "train/struct_loss",
    #         struct_loss,
    #         on_step=False,
    #         on_epoch=True,
    #         sync_dist=True,
    #     )

    #     return loss + struct_loss



    # Self-Critical Sequence Training:
    # -------------------------------

    # reward function
    # def compute_reward(self, pred_out, target_out):
    #     pred_seq = pred_out.argmax(dim=-1) 
    #     target_seq = target_out.argmax(dim=-1)

    #     # compute levenshein distance for reward signal
    #     reward = -editdistance.eval(pred_seq.tolist(), target_seq.tolist())  

    #     return torch.tensor(reward, dtype=torch.float, device=self.device)
    
    def compute_reward(self, pred_out, target_out):
        """
        Compute reward using Levenshtein distance (edit distance).
        
        Args:
        - pred_out (Tensor): [batch_size, seq_len] predicted token indices
        - target_out (Tensor): [batch_size, seq_len] target token indices
        
        Returns:
        - reward (Tensor): A tensor of shape [batch_size] containing the reward for each sequence
        """
        batch_size = pred_out.size(0)

        # List to store the rewards for each sequence
        rewards = []

        # Ensure both pred_out and target_out are tensors (convert them if they are not)
        if isinstance(pred_out, list):
            pred_out = torch.tensor(pred_out, dtype=torch.long, device=self.device)
        if isinstance(target_out, list):
            target_out = torch.tensor(target_out, dtype=torch.long, device=self.device)

        # Compute Levenshtein distance for each pair of generated (pred_out) and target (target_out)
        for i in range(batch_size):
            # Convert the sequences to lists of tokens for edit distance calculation
            pred_seq = pred_out[i].cpu().numpy().tolist()  # Convert to list for editdistance
            target_seq = target_out[i].cpu().numpy().tolist()

            # Compute edit distance (Levenshtein distance)
            dist = editdistance.eval(pred_seq, target_seq)

            # Reward is the negative of the distance (lower distance means better match)
            reward = -dist
            rewards.append(reward)

        # Convert the list of rewards to a tensor and return it
        return torch.tensor(rewards, dtype=torch.float, device=self.device)

    def training_step(self, batch: Batch, batch_idx: int):
        tgt, out = to_bi_tgt_out(batch.indices, self.device)
        struct_out, _ = to_struct_output(batch.indices, self.device)

        # Forward pass (generation)
        out_hat, sim = self(batch.imgs, batch.mask, tgt)

        # Compute the reward using the updated compute_reward function
        reward = self.compute_reward(out_hat.argmax(dim=-1), batch.indices)

        # Standard CE loss
        ce_loss_value = ce_loss(out_hat, out)
        self.log("train_loss", ce_loss_value, on_step=False, on_epoch=True, sync_dist=True)

        # Struct loss
        struct_loss_value = ce_loss(sim, struct_out, ignore_idx=-1)
        self.log("train/struct_loss", struct_loss_value, on_step=False, on_epoch=True, sync_dist=True)

        # SCST loss (scaled by reward)
        scst_loss = F.mse_loss(out_hat, out)  # Base SCST loss (you can change it if needed)
        scst_loss = scst_loss * reward.view(-1, 1)  # Scale by the reward for each sequence

        # Total loss: Combine CE loss, struct loss, and SCST loss
        total_loss = ce_loss_value + struct_loss_value + scst_loss.mean()

        return total_loss
    

    # def training_step(self, batch: Batch, _):
    #     tgt, out = to_bi_tgt_out(batch.indices, self.device)
    #     struct_out, _ = to_struct_output(batch.indices, self.device)

    #     # baseline output
    #     baseline_out, sim = self(batch.imgs, batch.mask, tgt)

    #     # sampled output
    #     sampled_tgt = self.sample_output(batch.imgs, batch.mask) 
    #     sampled_out, _ = self(batch.imgs, batch.mask, sampled_tgt)

    #     # cross-entropy loss
    #     ce_loss = ce_loss(baseline_out, out)

    #     # struct loss
    #     struct_loss = ce_loss(sim, struct_out, ignore_idx=-1)

    #     # reinforce loss
    #     baseline_reward = self.compute_reward(baseline_out, out)
    #     sampled_reward = self.compute_reward(sampled_out, out)
    #     reinforce_loss = (sampled_reward - baseline_reward) * self.compute_nll_loss(sampled_out, sampled_tgt)

    #     # combined loss
    #     loss = ce_loss + struct_loss + 0.5 * reinforce_loss

    #     return loss



    def validation_step(self, batch: Batch, _):
        tgt, out = to_bi_tgt_out(batch.indices, self.device)
        struct_out, _ = to_struct_output(batch.indices, self.device)
        out_hat, sim = self(batch.imgs, batch.mask, tgt)

        loss = ce_loss(out_hat, out)
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        struct_loss = ce_loss(sim, struct_out, ignore_idx=-1)
        self.log(
            "val/struct_loss",
            struct_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        # if self.current_epoch < self.hparams.milestones[0]:
        #     self.log(
        #         "val_ExpRate",
        #         self.exprate_recorder,
        #         prog_bar=True,
        #         on_step=False,
        #         on_epoch=True,
        #     )
        #     return

        hyps = self.approximate_joint_search(batch.imgs, batch.mask)

        self.exprate_recorder([h.seq for h in hyps], batch.indices)
        self.log(
            "val_ExpRate",
            self.exprate_recorder,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )

    def test_step(self, batch: Batch, _):
        hyps = self.approximate_joint_search(batch.imgs, batch.mask)
        self.exprate_recorder([h.seq for h in hyps], batch.indices)
        gts = [vocab.indices2words(ind) for ind in batch.indices]
        preds = [vocab.indices2words(h.seq) for h in hyps]

        return batch.img_bases, preds, gts

    def test_epoch_end(self, test_outputs) -> None:
        exprate = self.exprate_recorder.compute()
        print(f"Validation ExpRate: {exprate}")
        errors_dict = {}
        predictions_dict = {}
        with zipfile.ZipFile("result.zip", "w") as zip_f:
            for img_bases, preds, gts in test_outputs:
                for img_base, pred, gt in zip(img_bases, preds, gts):
                    content = f"%{img_base}\n${pred}$".encode()
                    with zip_f.open(f"{img_base}.txt", "w") as f:
                        f.write(content)
                    distance = editdistance.eval(pred, gt)
                    if distance > 0:
                        errors_dict[img_base] = {
                            "pred": " ".join(pred),
                            "gt": " ".join(gt),
                            "dist": distance,
                        }

                    predictions_dict[img_base] = {
                        "pred": " ".join(pred),
                        "gt": " ".join(gt),
                        "dist": distance,
                    }
        with open("errors.json", "w") as f:
            json.dump(errors_dict, f)
        with open("predictions.json", "w") as f:
            json.dump(predictions_dict, f)

    def approximate_joint_search(
        self, img: FloatTensor, mask: LongTensor
    ) -> List[Hypothesis]:
        return self.tamer_model.beam_search(img, mask, **self.hparams)

    def configure_optimizers(self):
        optimizer = optim.Adadelta(
            self.parameters(),
            lr=self.hparams.learning_rate,
            eps=1e-6,
            weight_decay=1e-4,
        )
        # optimizer = optim.AdamW(
        #     self.parameters(), lr=self.hparams.learning_rate, weight_decay=1e-4
        # )

        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=self.hparams.milestones, gamma=0.1
        )
        # reduce_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        #     optimizer,
        #     mode="max",
        #     factor=0.25,
        #     patience=self.hparams.patience // self.trainer.check_val_every_n_epoch,
        # )

        # scheduler = {
        #     "scheduler": reduce_scheduler,
        #     "monitor": "val_ExpRate",
        #     "interval": "epoch",
        #     "frequency": self.trainer.check_val_every_n_epoch,
        #     "strict": True,
        # }

        return {"optimizer": optimizer, "lr_scheduler": scheduler}
