import zipfile
from typing import List
import editdistance
import json
import pytorch_lightning as pl
import torch
import torch.optim as optim
from torch import FloatTensor, LongTensor

from tamer.datamodule import Batch, vocab
from tamer.model.tamer import TAMER
from tamer.utils.utils import (
    ExpRateRecorder, Hypothesis, LSM, ce_loss, to_bi_tgt_out, to_struct_output, compute_weights, compute_lsm, lsm_score)


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
        self.lsm = LSM()

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

    def training_step(self, batch: Batch, _):
        tgt, out = to_bi_tgt_out(batch.indices, self.device)
        struct_out, _ = to_struct_output(batch.indices, self.device)
        out_hat, sim = self(batch.imgs, batch.mask, tgt)

        # print("out.shape:", out.shape)
        # print("out_hat.shape", out_hat.shape)

        loss = ce_loss(out_hat, out)
        self.log("train_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        struct_loss = ce_loss(sim, struct_out, ignore_idx=-1)
        self.log(
            "train/struct_loss",
            struct_loss,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

        # hyps = self.approximate_joint_search(batch.imgs, batch.mask)
        
        # pred_strings = [vocab.indices2words(h.seq) for h in hyps]
        # print("out length:", out.shape)
        # gt_strings = [vocab.indices2words(ind) for ind in batch.indices]
        # # print("Got gt strings")
        # print(gt_indices.tolist())
        # print()
        # pred_indices = out_hat.argmax(dim=-1)
        # print(batch.indices)
        # print(len(batch.indices))
        # print(pred_indices.tolist())
        # pred_strings = [vocab.indices2words(seq.tolist()) for seq in pred_indices]
        # # print("Got pred strings")
        # for i in range(len(gt_strings)):
        #     print(gt_strings[i])
        #     print(pred_strings[i])
        #     print()
        # exit()
        # weights = compute_weights(gt_strings, pred_strings)
        # weights = compute_weights([vocab.indices2words(ind) for ind in batch.indices], [vocab.indices2words(h.seq) for h in hyps])
        #return weights * loss + struct_loss

        # gradient based adaptive loss:
        loss_grad = torch.autograd.grad(loss, self.parameters(), retain_graph=True, allow_unused=True)
        struct_loss_grad = torch.autograd.grad(struct_loss, self.parameters(), retain_graph=True, allow_unused=True)

        norm_loss_grad = torch.norm(torch.cat([g.view(-1) for g in loss_grad if g is not None]))
        norm_struct_loss_grad = torch.norm(torch.cat([g.view(-1) for g in struct_loss_grad if g is not None]))

        w_loss = norm_loss_grad/ (norm_loss_grad + norm_struct_loss_grad)
        w_struct_loss = norm_struct_loss_grad / (norm_loss_grad + norm_struct_loss_grad)
        
        return w_loss * loss + w_struct_loss * struct_loss


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

        # preds = [vocab.indices2words(h.seq) for h in hyps]
        # gts = [vocab.indices2words(ind) for ind in batch.indices]
        # pred_strings = [' '.join(pred) for pred in preds]
        # gt_strings = [' '.join(gt) for gt in gts]
        self.lsm([h.seq for h in hyps], batch.indices)
        self.log(
            "val_lsm",
            self.lsm,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )

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
        self.lsm([h.seq for h in hyps], batch.indices)
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
        
        # Save the LSM results
        lsm = self.lsm.compute()
        print(f"Validation LSM: {lsm}")
        predictions_lsm_dict = {}
        with zipfile.ZipFile("result_lsm.zip", "w") as zip_f:
            for img_bases, preds, gts in test_outputs:
                for img_base, pred, gt in zip(img_bases, preds, gts):
                    content = f"%{img_base}\n${pred}$".encode()
                    with zip_f.open(f"{img_base}.txt", "w") as f:
                        f.write(content)
                    lsm = lsm_score(pred, gt)

                    predictions_lsm_dict[img_base] = {
                        "pred": " ".join(pred),
                        "gt": " ".join(gt),
                        "lsm": lsm,
                    }
        with open("predictions_lsm.json", "w") as f:
            json.dump(predictions_lsm_dict, f)
        
        

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
