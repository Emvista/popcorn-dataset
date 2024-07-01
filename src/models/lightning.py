from torch import optim
import torch
import pytorch_lightning as pl

from models.base import Base
from models.losses import PRUNERLoss, NLPLoss
from models.pruner import PRUNER
from models.nlp import NLP

torch.use_deterministic_algorithms(False, warn_only=True)


class PL_UNIFIED(pl.LightningModule):
    def __init__(self, config, nb_train_steps=0):
        super().__init__()
        self.loss_span = PRUNERLoss()
        self.loss_nlp = NLPLoss()
        self.base = Base(config, "camembert-base")
        self.pruner = PRUNER(
            config=config["PRUNER"],
            max_span_len=config["max_span_len"],
            batch_size=config["batch_size"],
        )
        self.nlp = NLP(
            config=config["NLP"],
            max_span_len=config["max_span_len"],
            batch_size=config["batch_size"],
            output_len=config["output_len"],
        )
        self.learning_rate = config["learning_rate"]
        self.nb_train_steps = nb_train_steps
        self.validation_step_outputs = []
        self.train_step_outputs = []
        self.output_len = config["output_len"]

    def training_step(self, batch, batch_idx):
        embeddings = self.base(
            batch["ids"],
            batch["maps"],
            batch["len_samples"],
            batch["subtoken_map"],
            batch["mask_ids"].to(self.device),
        )
        span_hat = self.pruner(embeddings)
        loss_span = self.loss_span(span_hat, batch["span_labels"], batch["mask"])
        nlp_hat = self.nlp(embeddings, batch["begin_indexes"], batch["end_indexes"])
        loss_nlp = self.loss_nlp(nlp_hat, batch["nlp_labels"], batch["nlp_mask"])
        self.log("loss_span", loss_span)
        self.log("loss_nlp", loss_nlp)
        self.log("train_loss", loss_nlp + loss_span)
        self.train_step_outputs.append(
            {"loss_span": loss_span, "loss_nlp": loss_nlp, "loss": loss_span + loss_nlp}
        )
        return {
            "loss_span": loss_span,
            "loss_nlp": loss_nlp,
            "loss": loss_span + loss_nlp,
        }

    def on_train_epoch_end(self) -> None:
        loss = sum(output["loss"] for output in self.train_step_outputs) / len(
            self.train_step_outputs
        )
        loss_nlp = sum(output["loss_nlp"] for output in self.train_step_outputs) / len(
            self.train_step_outputs
        )
        loss_span = sum(
            output["loss_span"] for output in self.train_step_outputs
        ) / len(self.train_step_outputs)
        self.log("avg_train_loss", loss)
        self.log("avg_train_loss_nlp", loss_nlp)
        self.log("avg_train_loss_span", loss_span)
        self.train_step_outputs.clear()

    def validation_step(self, batch, batch_idx):
        embeddings = self.base(
            batch["ids"],
            batch["maps"],
            batch["len_samples"],
            batch["subtoken_map"],
            batch["mask_ids"],
        )
        hat_span = self.pruner(embeddings)
        loss_span = self.loss_span(hat_span, batch["span_labels"], batch["mask"])
        hat_nlp = self.nlp(embeddings, batch["begin_indexes"], batch["end_indexes"])
        loss_nlp = self.loss_nlp(hat_nlp, batch["nlp_labels"], batch["nlp_mask"])
        self.log("val_loss_span", loss_span)
        self.log("val_loss_nlp", loss_nlp)
        self.log("val_loss", loss_nlp + loss_span)
        tp_span, fp_span, fn_span = self.loss_span.evaluate(
            hat_span.squeeze() * batch["mask"], batch["span_labels"]
        )
        tp_nlp, fp_nlp, fn_nlp = self.loss_nlp.evaluate(
            hat_nlp * batch["nlp_mask"], batch["nlp_labels"] * batch["nlp_mask"]
        )
        self.validation_step_outputs.append(
            {
                "val_loss_span": loss_span,
                "val_loss_nlp": loss_nlp,
                "val_loss": loss_nlp + loss_span,
                "tp_nlp": tp_nlp,
                "fp_nlp": fp_nlp,
                "fn_nlp": fn_nlp,
                "tp_span": tp_span,
                "fp_span": fp_span,
                "fn_span": fn_span,
            }
        )
        return {
            "val_loss_span": loss_span,
            "val_loss_nlp": loss_nlp,
            "val_loss": loss_nlp + loss_span,
            "tp_nlp": tp_nlp,
            "fp_nlp": fp_nlp,
            "fn_nlp": fn_nlp,
            "tp_span": tp_span,
            "fp_span": fp_span,
            "fn_span": fn_span,
        }

    def on_validation_epoch_end(self):
        loss_nlp = sum(
            output["val_loss_nlp"] for output in self.validation_step_outputs
        ) / len(self.validation_step_outputs)
        loss_span = sum(
            output["val_loss_span"] for output in self.validation_step_outputs
        ) / len(self.validation_step_outputs)
        loss = sum(output["val_loss"] for output in self.validation_step_outputs) / len(
            self.validation_step_outputs
        )
        self.log("avg_val_loss", loss)
        self.log("avg_val_loss_nlp", loss_nlp)
        self.log("avg_val_loss_span", loss_span)
        tp = sum(output["tp_nlp"] for output in self.validation_step_outputs)
        fp = sum(output["fp_nlp"] for output in self.validation_step_outputs)
        fn = sum(output["fn_nlp"] for output in self.validation_step_outputs)
        r = tp / (tp + fn)
        p = tp / (tp + fp)
        nlp = 2 * r * p / (r + p)
        self.log(
            "avg_val_f1_nlp",
            nlp,
        )

        tp = sum(output["tp_span"] for output in self.validation_step_outputs)
        fp = sum(output["fp_span"] for output in self.validation_step_outputs)
        fn = sum(output["fn_span"] for output in self.validation_step_outputs)
        r = tp / (tp + fn)
        p = tp / (tp + fp)
        span = 2 * r * p / (r + p)
        self.log(
            "avg_val_f1_span",
            span,
        )
        self.log("avg_val_f1", ((span + nlp) / 2))
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        opt = optim.Adam(self.parameters(), lr=self.learning_rate)
        sch = optim.lr_scheduler.LinearLR(
            optimizer=opt,
            start_factor=1,
            end_factor=0,
            total_iters=self.nb_train_steps,
            last_epoch=-1,
        )
        return {
            "optimizer": opt,
            "lr_scheduler": sch,
        }

    def predict_relations(self, elems: dict) -> torch.tensor:
        embs = self.base(
            elems["ids"].to(self.device),
            elems["maps"].to(self.device),
            elems["len_samples"],
            elems["subtoken_map"].to(self.device),
            elems["mask_ids"].to(self.device),
        )
        pred_types = self.nlp(embs, elems["begin_indexes"], elems["end_indexes"]).to(
            self.device
        )
        return pred_types * elems["nlp_mask"].to(self.device)

    def predict_all(self, elems: dict) -> tuple[list, list, torch.tensor]:
        embs = self.base(
            elems["ids"].to(self.device),
            elems["maps"].to(self.device),
            elems["len_samples"],
            elems["subtoken_map"].to(self.device),
            elems["mask_ids"].to(self.device),
        )
        pruned_tokens = torch.argwhere(
            self.pruner(embs).squeeze() * elems["mask"].to(self.device) > 0
        )
        begin_indexes, end_indexes = [], []
        for begin, len_mention in pruned_tokens:
            begin_indexes.append(begin)
            end_indexes.append(begin + len_mention)
        pred_interations = self.nlp(embs, begin_indexes, end_indexes)
        return begin_indexes, end_indexes, pred_interations
