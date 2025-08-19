"""
Model Architecture (Updated):

- Input: Two decks of 8 cards each (your deck and opponent's deck)
- Embedding Layer: Each card index is mapped to a learnable embedding of size `embedding_dim`.
- Deck Encoder: For each deck, embeddings are flattened and passed through two fully connected layers with ReLU activations to produce a deck representation.
- Concatenation: The encoded representations of both decks are concatenated.
- Output Layer: A fully connected layer maps the concatenated vector to a 4x4 matrix (flattened to 16 outputs), representing the probabilities of each possible tower outcome.
- Softmax: Softmax activation is applied across the 16 outputs to produce a probability distribution over the 4x4 outcome matrix.
- Output: The output is reshaped to (batch_size, 4, 4), representing the predicted probability for each tower outcome.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim import Adam


class ClashRoyaleOutcomeNet(pl.LightningModule):
    def __init__(
        self, num_cards=120, deck_size=8, embedding_dim=32, hidden_dim=256, lr=1e-3
    ):
        super().__init__()
        self.num_cards = num_cards
        self.deck_size = deck_size
        self.lr = lr

        self.card_embedding = nn.Embedding(num_cards, embedding_dim)

        self.deck_encoder = nn.Sequential(
            nn.Linear(deck_size * embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.fc_out = nn.Linear(hidden_dim * 2, 16)
        self.save_hyperparameters()

    def forward(self, deck1, deck2):
        """
        deck1: LongTensor (batch_size, deck_size)  -> your deck
        deck2: LongTensor (batch_size, deck_size)  -> opponent deck
        """
        x1 = self.card_embedding(deck1)
        x1 = x1.view(x1.size(0), -1)
        x1 = self.deck_encoder(x1)

        x2 = self.card_embedding(deck2)
        x2 = x2.view(x2.size(0), -1)
        x2 = self.deck_encoder(x2)

        x = torch.cat([x1, x2], dim=1)

        x = self.fc_out(x)
        x = F.softmax(x, dim=1)

        x = x.view(-1, 4, 4)
        return x

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        deck1, deck2, targets = batch

        predictions = self(deck1, deck2)
        loss = F.cross_entropy(predictions.view(-1, 16), targets.view(-1, 16))
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        deck1, deck2, targets = batch
        predictions = self(deck1, deck2)
        loss = F.cross_entropy(predictions.view(-1, 16), targets.view(-1, 16))
        self.log("val_loss", loss, prog_bar=True)
        return loss


def main() -> None:
    num_cards = 120
    deck_size = 8
    model = ClashRoyaleOutcomeNet(num_cards=num_cards, deck_size=deck_size)

    # Example of using the model directly for prediction
    my_deck = torch.randint(0, num_cards, (1, deck_size))  # shape (1,8)
    opp_deck = torch.randint(0, num_cards, (1, deck_size))  # shape (1,8)

    outcome_matrix = model(my_deck, opp_deck)
    outcome_matrix = outcome_matrix * 100  # convert probabilities to percentages

    print("Predicted tower outcome probabilities:\n", outcome_matrix)


if __name__ == "__main__":
    main()
