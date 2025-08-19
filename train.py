import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from dataloader import BattleSummaryDataModule
from model import ClashRoyaleOutcomeNet


def main():
    num_cards = 120
    deck_size = 8
    lr = 1e-3
    epochs = 5
    batch_size = 64

    dm = BattleSummaryDataModule("battle_summary_list.json", batch_size=batch_size)
    model = ClashRoyaleOutcomeNet(num_cards=num_cards, deck_size=deck_size, lr=lr)

    trainer = Trainer(
        max_epochs=epochs,
        accelerator="auto",
        devices="auto",
        log_every_n_steps=10,
    )
    trainer.fit(model, dm)


if __name__ == "__main__":
    main()
