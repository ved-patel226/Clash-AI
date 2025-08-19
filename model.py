"""
Model Architecture:

- Input: Deck of 8 cards, represented as indices (LongTensor of shape [batch_size, deck_size]).
- Embedding Layer: Each card is mapped to a learnable embedding of size `embedding_dim` (captures card features).
- Flatten: Concatenate embeddings of all 8 cards into a single vector of size `deck_size * embedding_dim`.
- Fully Connected Layers:
    - fc1: Hidden layer with `hidden_dim` neurons, ReLU activation.
    - fc2: Hidden layer with `hidden_dim` neurons, ReLU activation.
    - fc3: Output layer with `num_cards` neurons, sigmoid activation.
- Output: Probability vector of length `num_cards` representing likelihood of each card being selected for a counter-deck.
- Training: Use Binary Cross-Entropy Loss treating each card as independent.
- Deck Generation: Sample top-k cards from output probabilities to form a counter-deck of 8 cards.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


class DeckCounterNet(pl.LightningModule):
    def __init__(
        self, num_cards=120, deck_size=8, embedding_dim=32, hidden_dim=256, lr=1e-3
    ):
        super().__init__()
        self.num_cards = num_cards
        self.deck_size = deck_size
        self.lr = lr

        self.card_embedding = nn.Embedding(num_cards, embedding_dim)

        self.fc1 = nn.Linear(deck_size * embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_cards)

        print(f"FC1: {self.fc1}")
        print(f"FC2: {self.fc2}")
        print(f"FC3: {self.fc3}")

    def forward(self, deck_indices):
        """
        deck_indices: LongTensor of shape (batch_size, deck_size)
        """
        # Embed each card and flatten
        x = self.card_embedding(deck_indices)  # (batch, deck_size, embedding_dim)
        x = x.view(x.size(0), -1)  # (batch, deck_size * embedding_dim)

        # Forward through MLP
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))  # probability for each card in counter-deck

        return x

    def training_step(self, batch, batch_idx):
        deck_indices, target_counter_deck = batch
        output = self(deck_indices)
        loss = F.binary_cross_entropy(output, target_counter_deck)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        deck_indices, target_counter_deck = batch
        output = self(deck_indices)
        loss = F.binary_cross_entropy(output, target_counter_deck)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


def main() -> None:
    model = DeckCounterNet()

    cr_deck = torch.tensor([[3, 17, 5, 0, 42, 88, 19, 11]])  # shape: (1, 8)

    output = model(cr_deck)

    # To get a counter-deck, pick the top 8 cards
    top_indices = torch.topk(output, 8, dim=1).indices
    print("Suggested counter-deck indices:", top_indices)


if __name__ == "__main__":
    main()
