import torch
import torch.nn as nn


class TaxiFeatureCreator(nn.Module):
    def __init__(self):
        super(TaxiFeatureCreator, self).__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(6, 10),    # Passenger count - 1
            nn.Embedding(7, 10),    # Year - 2009
            nn.Embedding(12, 10),   # Month
            nn.Embedding(7, 10),    # Weekday
            nn.Embedding(96, 10)    # Quaterhour
        ])

    def forward(self, x, y):
        out = [x]
        for i, v in enumerate(y.t()):
            out.append(self.embeddings[i](v))

        return torch.cat(out, dim=1)


class TaxiNet(nn.Module):
    def __init__(self, p_drop, dim_out=1, **kwargs):
        super(TaxiNet, self).__init__()

        dim_in = 54
        self.feature_creator = TaxiFeatureCreator()

        self.layers = nn.ModuleList([
            nn.Linear(dim_in, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(p_drop),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(p_drop),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(p_drop),
            nn.Linear(64, dim_out)
        ])

        if dim_out > 1:
            self.layers.append(nn.Softmax(dim=1))

    def forward(self, x):
        x = self.feature_creator(x[:, :4], x[:, 4:].long())
        for layer in self.layers:
            x = layer(x)

        return x
