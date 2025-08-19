import json
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl


class BattleSummaryDataset(Dataset):
    def __init__(self, json_path):
        with open(json_path, "r") as f:
            loaded = json.load(f)

            if isinstance(loaded, dict):
                self.data = list(loaded.values())
            else:
                self.data = loaded

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        team_crowns = item["team_player"]["crowns"]
        opp_crowns = item["opponent_player"]["crowns"]

        mat = torch.zeros(4, 4, dtype=torch.float32)
        mat[team_crowns, opp_crowns] = 1.0

        team_deck = torch.tensor(item["team_player"]["deck"], dtype=torch.long)
        opp_deck = torch.tensor(item["opponent_player"]["deck"], dtype=torch.long)

        return team_deck, opp_deck, mat


class BattleSummaryDataModule(pl.LightningDataModule):
    def __init__(self, json_path, batch_size=32, num_workers=0):
        super().__init__()
        self.json_path = json_path
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        # Idempotent creation so repeated calls don't recreate
        if not hasattr(self, "dataset"):
            self.dataset = BattleSummaryDataset(self.json_path)

    def _ensure_setup(self):
        """Lazily ensure dataset exists if user didn't call setup() manually."""
        if not hasattr(self, "dataset"):
            self.setup()

    def train_dataloader(self):
        self._ensure_setup()
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        self._ensure_setup()
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        self._ensure_setup()
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )


def main() -> None:
    dm = BattleSummaryDataModule("battle_summary_list.json", batch_size=16)
    for i, batch in enumerate(dm.train_dataloader()):
        print(batch)
        if i == 0:
            break


if __name__ == "__main__":
    main()
