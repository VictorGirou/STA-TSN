from torch import nn
import torch
import pandas as pd

import numpy as np


class PSTA_TCN(nn.Module):

    def __init__(self, window_size, kernel_size, n_hidden_layers, n_hidden_dimensions, n_signals, prediction_horizon):

        super().__init__()

        # Hyperparameters

        self.T = window_size
        self.tau = prediction_horizon
        self.k = kernel_size
        self.L = n_hidden_layers
        self.H = n_hidden_dimensions
        self.n = n_signals

        self.lr = 0.001
        self.metric = nn.MSELoss()

        # Attention mechanisms

        self.spatial_attention = nn.Sequential(
            nn.Linear(in_features=self.T, out_features=self.T, bias=True),
            nn.Softmax(dim=1),
        )

        self.temporal_attention = nn.Sequential(
            nn.Linear(in_features=self.n, out_features=self.n, bias=True),
            nn.Softmax(dim=1),
        )

        # TCN backbone

        self.spatial_conv = nn.utils.weight_norm(
            nn.Conv1d(
                in_channels=self.n,
                out_channels=self.n,
                kernel_size=7,
                dilation=1,
                padding=(self.k - 1) // 2
            )
        )
        self.temporal_conv = nn.utils.weight_norm(
            nn.Conv1d(
                in_channels=self.T,
                out_channels=self.T,
                kernel_size=7,
                dilation=1,
                padding=(self.k - 1) // 2
            )
        )

        self.norm_spatial = nn.BatchNorm1d(num_features=self.T)
        self.norm_temporal = nn.BatchNorm1d(num_features=self.n)

        self.dropout = nn.Dropout(p=0.1)

        # Dense layers

        self.dense_layers_spatial1 = nn.Sequential(
            nn.Linear(in_features=self.n, out_features=self.H),
            nn.ReLU(),
        )

        self.dense_layers_spatial2 = nn.Sequential(
            nn.Linear(in_features=self.H, out_features=1),
            nn.ReLU(),
        )

        self.dense_layers_spatial3 = nn.Sequential(
            nn.Linear(in_features=self.T, out_features=self.tau),
        )

        self.dense_layers_temporal1 = nn.Sequential(
            nn.Linear(in_features=self.n, out_features=self.H),
            nn.ReLU(),
        )

        self.dense_layers_temporal2 = nn.Sequential(
            nn.Linear(in_features=self.H, out_features=1),
            nn.ReLU(),
        )

        self.dense_layers_temporal3 = nn.Sequential(
            nn.Linear(in_features=self.T, out_features=self.tau),
        )

    def forward(self, x):
        x_s = self.spatial_attention(torch.transpose(x, -2, -1))
        x_t = self.temporal_attention(x)

        # Compute the TCN spatial backbone

        y_s = self.spatial_conv(x_s)
        y_s = nn.ReLU()(y_s)
        y_s = self.dropout(y_s)

        y_s += x_s

        y_s = self.dense_layers_spatial1(torch.transpose(y_s, -2, -1))
        y_s = self.dense_layers_spatial2(y_s)
        y_s = self.dense_layers_spatial3(torch.transpose(y_s, -2, -1))

        # Compute the TCN temporal backbone

        y_t = self.temporal_conv(x_t)
        y_t = nn.ReLU()(y_t)
        y_t = self.dropout(y_t)

        y_t += x_t

        y_t = self.dense_layers_temporal1(y_t)
        y_t = self.dense_layers_temporal2(y_t)
        y_t = self.dense_layers_temporal3(torch.transpose(y_t, -2, -1))

        return y_s + y_t

    def predict(self, x):

        self.eval()

        with torch.no_grad():
            x_s = self.spatial_attention(x.T)
            x_t = self.temporal_attention(x)

            # Compute the TCN spatial backbone

            print('outpu attention layer: ', x_s.shape)
            y_s = self.spatial_conv(x_s)
            y_s = nn.ReLU()(y_s)

            y_s += x_s

            print('residual s shape: ', y_s.shape)

            y_s = self.dense_layers_spatial1(y_s.T)
            print('y_s shape: ', y_s.shape)
            y_s = self.dense_layers_spatial2(y_s)
            y_s = self.dense_layers_spatial3(y_s.T)

            # Compute the TCN temporal backbone

            y_t = self.temporal_conv(x_t)
            y_t = nn.ReLU()(y_t)

            y_t += x_t

            print('residual t shape: ', y_t.shape)

            y_t = self.dense_layers_temporal1(y_t)
            print('y_t shape: ', y_t.shape)
            y_t = self.dense_layers_temporal2(y_t)
            y_t = self.dense_layers_temporal3(y_t.T)

        self.train(True)

        return y_s + y_t

    def loss(self, predictions, ground_truth):

        return torch.sqrt(self.metric(predictions, ground_truth))

    def train_(self, trainloader): #valloader

        for epoch in range(self.epochs):
            training_loss=0
            for data,target in trainloader:
                training_loss+=self.training_step(data,target)

            training_loss=training_loss/len(trainloader)

            #val_loss=self.validate(valloader)

    def training_step(self, data,target):

        self.optimizer.zero_grad()
        output = self(data)
        loss = self.loss(output,target)
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def fit(self, train_set):#, val_set):

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        train_set = torch.from_numpy(train_set.astype('float32').values)
        # val_set = torch.from_numpy(val_set.astype('float32').values)
        train_set=torch.utils.data.TensorDataset(train_set[:,:-1],train_set[:,[-1]])
        # val_set=torch.utils.data.TensorDataset(val_set[:,:-1],val_set[:,[-1]])
        trainloader = torch.utils.data.DataLoader(train_set, batch_size=self.batch_size)
        # valloader = torch.utils.data.DataLoader(val_set, batch_size=self.batch_size)
        self.train_(trainloader)#,valloader)



if __name__ == '__main__':
    data_set=pd.read_csv('exchange_rate.csv')
    # Hyperparameters definition

    n_signals = 10
    window_size = 32
    kernel_size = 7
    n_hidden_layers = 12
    n_hidden_dimensions = 1
    prediction_horizon = 2

    model = PSTA_TCN(
        window_size=window_size,
        kernel_size=kernel_size,
        n_hidden_layers=n_hidden_layers,
        n_hidden_dimensions=n_hidden_dimensions,
        n_signals=n_signals,
        prediction_horizon=prediction_horizon,
    )

    model.fit(data_set)

    #Il faudrait split le dataset en train val

