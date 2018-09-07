import torch
import torch.nn as nn


class TaxiFeatureCreator(nn.Module):
    def __init__(self):
        super(TaxiFeatureCreator, self).__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(6, 3),    # Passenger count - 1
            nn.Embedding(7, 4),    # Year - 2009
            nn.Embedding(12, 6),   # Month
            nn.Embedding(7, 4),    # Weekday
            nn.Embedding(96, 50)    # Quaterhour
        ])

    def forward(self, x, y):
        out = [x]
        for i, v in enumerate(y.t()):
            out.append(self.embeddings[i](v))

        return torch.cat(out, dim=1)


class TaxiNet(nn.Module):
    def __init__(self, dim_hidden, n_layers, p_drop, dim_out=1,
                 **kwargs):
        super(TaxiNet, self).__init__()

        dim_in = 71
        dim_hidden = int(dim_hidden)
        n_layers = int(n_layers)
        self.feature_creator = TaxiFeatureCreator()

        self.layers = nn.ModuleList([
            nn.Linear(dim_in, dim_hidden),
            nn.ReLU(),
            nn.BatchNorm1d(dim_hidden),
            nn.Dropout(p_drop)
        ])

        for i in range(n_layers-1):
            self.layers.extend([
                nn.Linear(dim_hidden, dim_hidden),
                nn.ReLU(),
                nn.BatchNorm1d(dim_hidden),
                nn.Dropout(p_drop)
            ])

        self.layers.append(nn.Linear(dim_hidden, dim_out))

        if dim_out > 1:
            self.layers.append(nn.Softmax(dim=1))

    def forward(self, x):
        x = self.feature_creator(x[:, :4], x[:, 4:].long())
        for layer in self.layers:
            x = layer(x)

        return x
