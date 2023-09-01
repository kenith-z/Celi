import torch
from torch import nn


class BiLstmModel(nn.Module):
    def __init__(self, n_vocab, input_size, hidden_size, embedding_dim, num_layers, device=torch.device('cpu'), ):
        super(BiLstmModel, self).__init__()

        self.embedding = nn.Embedding(
            num_embeddings=n_vocab,
            embedding_dim=embedding_dim,
        ).to(device)

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,  # 双向 LSTM
            dropout=0.2,
        ).to(device)

        self.output = nn.Linear(hidden_size * 2, n_vocab).to(device)
        self.softmax = nn.Softmax(dim=1).to(device)  # Softmax 激活函数

    def forward(self, x, prev_state):
        embed = self.embedding(x)
        output, state = self.lstm(embed, prev_state)

        # 对 output 进行整流线性单元操作，并返回处理后的张量
        output = torch.sigmoid(output)

        logits = self.output(output)

        return self.softmax(logits), state

    def init_state(self, num_layers, sequence_length, hidden_size, device=torch.device('cpu')):
        return (torch.zeros(num_layers * 2, sequence_length, hidden_size).to(device),
                torch.zeros(num_layers * 2, sequence_length, hidden_size).to(device))
