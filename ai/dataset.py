import chess
import pandas as pd
import torch
from torch.utils.data import Dataset, Subset

from ai.encoding import board_to_tensor, move_to_index


class ChessDataset(Dataset):

    def __init__(self, csv_file):
        super().__init__()
        self.data = pd.read_csv(csv_file)
        self.features = self.data["fen"].values
        self.labels = self.data["move"].values

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        board = chess.Board(self.features[idx])
        move = chess.Move.from_uci(self.labels[idx])
        x = board_to_tensor(board)
        y = move_to_index(move)
        return x, torch.tensor(y, dtype=torch.long)


def split_by_game(dataset, train_fraction=0.8, val_fraction=0.1, seed=42):
    games = dataset.data["game"].unique()
    generator = torch.Generator().manual_seed(seed)
    games = games[torch.randperm(len(games), generator=generator).numpy()]

    num_train = int(train_fraction * len(games))
    num_val = int(val_fraction * len(games))
    train_games = games[:num_train]
    val_games = games[num_train:num_train + num_val]
    test_games = games[num_train + num_val:]

    subsets = []
    for chosen in (train_games, val_games, test_games):
        indices = dataset.data.index[dataset.data["game"].isin(chosen)].tolist()
        subsets.append(Subset(dataset, indices))
    return subsets
