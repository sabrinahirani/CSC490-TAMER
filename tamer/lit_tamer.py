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
        self, img: FloatTensor, img_mask: LongTensor, tgt: LongTensor
    ) -> FloatTensor:
        """run img and bi-tgt

        Parameters
        ----------
        img : FloatTensor
            [b, 1, h, w]
        img_mask: LongTensor
            [b, h, w]
        tgt : LongTensor
            [2b, l]

        Returns
        -------
        FloatTensor
            [2b, l, vocab_size]
        """
        return self.tamer_model(img, img_mask, tgt)
    
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
    def compute_reward(self, pred, target):
        pred_seq = pred.argmax(dim=-1).tolist()
        target_seq = target.argmax(dim=-1).tolist()

        # compute levenshein distance for reward signal (vectorized)
        reward = -torch.tensor(
            [editdistance.eval(p, t) for p, t in zip(pred_seq, target_seq)], 
            device=self.device, dtype=torch.float
        )
        return reward
    
    # helper (sampling)
    @torch.no_grad()
    def sample_output(self, img, mask, sample_size=8):

        # randomly sample images from the batch
        batch_size = img.size(0)
        indices = torch.tensor(random.sample(range(batch_size), min(sample_size, batch_size)), device=self.device)
        
        # subsample images and masks
        sampled_img = img[indices]
        sampled_mask = mask[indices]
        
        # generate predictions using beam search
        hyps = self.approximate_joint_search(sampled_img, sampled_mask)
        sampled_seqs = [torch.tensor(h.seq, dtype=torch.long, device=self.device) for h in hyps]

        padded_seqs = torch.nn.utils.rnn.pad_sequence(
            sampled_seqs, batch_first=True, padding_value=vocab.PAD_IDX
        )
        
        return padded_seqs, indices


    
    # helper (negative log-likelihood los)
    def compute_nll_loss(self, logits, targets):
        # apply log-softmax to logits to get log probabilities
        log_probs = F.log_softmax(logits, dim=-1)  # [batch_size, seq_len, vocab_size]
        # flatten the logits and targets to make them compatible with nll_loss
        loss = F.nll_loss(
            log_probs.view(-1, log_probs.size(-1)),
            targets.view(-1),
            ignore_index=vocab.PAD_IDX
        )
        return loss
    

    def training_step(self, batch: Batch, _):
        tgt, out = to_bi_tgt_out(batch.indices, self.device)
        struct_out, _ = to_struct_output(batch.indices, self.device)
        out_hat, sim = self(batch.imgs, batch.mask, tgt)

        # debugging
        print(f"out_hat shape: {out_hat.shape}, sim shape: {sim.shape}")

        # cross-entropy loss
        loss = ce_loss(out_hat, out)

        # struct loss
        loss += ce_loss(sim, struct_out, ignore_idx=-1)

        # self-critical sequence training:

        # sampled
        with torch.no_grad():
            # randomly sample a small subset of images
            sampled_tgt, sampled_indices = self.sample_output(batch.imgs, batch.mask)

        # extract sampled
        sampled_imgs = batch.imgs[sampled_indices]
        sampled_masks = batch.mask[sampled_indices]
        sampled_out, _ = self(sampled_imgs, sampled_masks, sampled_tgt)
        
        # sampled reward
        sampled_reward = self.compute_reward(sampled_out, out[sampled_indices])

        # baseline reward
        baseline_reward = self.compute_reward(out_hat, out)

        # reinforce loss
        reinforce_loss = (sampled_reward - baseline_reward[sampled_indices].detach()) * self.compute_nll_loss(sampled_out, sampled_tgt)
        reinforce_loss = reinforce_loss.mean()

        # combined loss
        loss += 0.5 * reinforce_loss
        self.log("train_loss", loss, on_step=False, on_epoch=True, sync_dist=True)

        return loss



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
