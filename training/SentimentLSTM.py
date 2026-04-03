import torch
import torch.nn as nn

class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, embedding_matrix):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.embedding.weight.data.copy_(embedding_matrix)
        self.embedding.weight.requires_grad = False   # ← freeze GloVe

        self.embedding_dropout = nn.Dropout(0.4)      # ← new dropout

        self.lstm = nn.LSTM(input_size=embed_dim,
                            hidden_size=hidden_dim,
                            num_layers=2,
                            dropout=0.3,
                            batch_first=True,
                            bidirectional=True)

        self.dropout = nn.Dropout(0.6)                # ← stronger dropout
        self.fc      = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):
        embedded = self.embedding(x)
        embedded = self.embedding_dropout(embedded)   # ← dropout after embed

        output, (hidden, cell) = self.lstm(embedded)

        hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        hidden = self.dropout(hidden)
        return self.fc(hidden).squeeze(1)

if __name__ == "__main__":
    pass