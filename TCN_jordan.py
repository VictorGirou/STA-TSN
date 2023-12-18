from torch import nn
import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
import os

class PSTA_TCN(nn.Module):

    def __init__(self, window_size, kernel_size, n_hidden_layers, n_hidden_dimensions, n_signals, prediction_horizon,lr=0.0001,dropout_rate=0.01,batch_size=64,nb_epochs=100,patience=20,conv_dilatation=1,seed=0,number_TCN=2,name_model=''):

        super().__init__()

        # Hyperparameters
        self.epochs=nb_epochs
        self.batch_size=batch_size
        self.T = window_size
        self.tau = prediction_horizon
        self.k = kernel_size
        self.L = n_hidden_layers
        self.H = n_hidden_dimensions
        self.n = n_signals
        self.lr = lr
        self.metric = nn.MSELoss()
        self.last_loss=np.inf
        self.patience=patience
        self.conv_dilatation=conv_dilatation
        self.seed=seed
        self.number_TCN=number_TCN
        self.dropout_rate = dropout_rate
        # Attention mechanisms

        random.seed(self.seed)
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
                kernel_size=self.k,
                dilation=self.conv_dilatation,
                padding=(self.k - 1) // 2
            )
        )
        self.temporal_conv = nn.utils.weight_norm(
            nn.Conv1d(
                in_channels=self.T,
                out_channels=self.T,
                kernel_size=self.k,
                dilation=self.conv_dilatation,
                padding=(self.k - 1) // 2
            )
        )

        self.norm_spatial = nn.BatchNorm1d(num_features=self.T)
        self.norm_temporal = nn.BatchNorm1d(num_features=self.n)


        self.dropout = nn.Dropout(p=dropout_rate)

        self.spatial_TCN = self.build_TCN('spatial')
        self.temporal_TCN = self.build_TCN('temporal')

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

        self.dense_layers_spatial_jordan = nn.Sequential(
            nn.Linear(in_features=self.n, out_features=1),
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

        self.dense_layers_temporal_jordan = nn.Sequential(
            nn.Linear(in_features=self.T, out_features=1),
        )

        self.model_save_path = r'/Users/jordan/Documents/MVA/ML_TS_2023/Projet/saved_models/model_'+name_model+'_' + str(self.seed) + '.pt'

    def build_TCN(self,type):
        channel=self.n if type=='spatial' else self.T
        TCN_list=[]
        for i in range(self.number_TCN):
            # Dilated Causal Convolution
            dilated_conv = nn.Conv1d(
                in_channels=channel,
                out_channels=channel,
                kernel_size=self.k,
                dilation=self.conv_dilatation,
                padding=((self.k - 1)*self.conv_dilatation // 2),
            )
            weight_norm_conv = nn.utils.weight_norm(dilated_conv)

            # Batch Normalization, ReLU, and Dropout
            block = nn.Sequential(
                weight_norm_conv,
                # nn.BatchNorm1d(num_features=channel),
                nn.ReLU(),
                nn.Dropout(p=self.dropout_rate)
            )

            TCN_list.append(block)

        return nn.Sequential(*TCN_list)
    def forward(self, x):

        x_s = self.spatial_attention(torch.transpose(x, -2, -1))
        x_t = self.temporal_attention(x)

        # Compute the TCN spatial backbone

        # y_s = self.spatial_conv(x_s)
        # y_s = nn.ReLU()(y_s)
        # y_s = self.dropout(y_s)
        y_s=self.spatial_TCN(x_s)

        y_s += x_s

        y_s = self.dense_layers_spatial1(torch.transpose(y_s, -2, -1))
        y_s = self.dense_layers_spatial2(y_s)
        y_s = self.dense_layers_spatial3(torch.transpose(y_s, -2, -1))
        # y_s = self.dense_layers_spatial_jordan(torch.transpose(y_s, -2, -1))
        # Compute the TCN temporal backbone

        # y_t = self.temporal_conv(x_t)
        # y_t = nn.ReLU()(y_t)
        # y_t = self.dropout(y_t)

        y_t=self.temporal_TCN(x_t)

        y_t += x_t

        y_t = self.dense_layers_temporal1(y_t)
        y_t = self.dense_layers_temporal2(y_t)
        y_t = self.dense_layers_temporal3(torch.transpose(y_t, -2, -1))
        # y_t=self.dense_layers_temporal_jordan(y_t)

        return y_s + y_t

    def predict(self, x):

        self.eval()

        with torch.no_grad():

            return self(x)

    def loss(self, predictions, ground_truth):

        return torch.sqrt(self.metric(predictions, ground_truth))

    def train_(self, trainloader, valloader):

        for epoch in range(self.epochs):
            training_loss=0
            for data,target in trainloader:
                training_loss+=self.training_step(data,target)

            training_loss=training_loss/len(trainloader)
            #, validation_loss:{val_loss} \t')
            val_loss=self.validate(valloader)
            if val_loss > self.last_loss:
                trigger_times += 1
                # print('Trigger Times:', trigger_times)

                if trigger_times >= self.patience:
                    # print('Early stopping!\nStart to test process.')

                    break

            else:
                self.last_loss = val_loss
                # print('trigger times: 0')
                trigger_times = 0

        torch.save(self.state_dict(), self.model_save_path)

            # print(f'epochs {epoch}: training_loss: {training_loss}, validation_loss:{val_loss} \t')

    def training_step(self, data,target):
        self.train()
        self.optimizer.zero_grad()
        output = self(data)
        output = output.reshape(output.shape[0],self.tau)

        loss = self.loss(output,target)
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def validate(self,valloader):

        val_loss=0
        for data, target in valloader:
            val_loss += self.validation_step(data, target)


        return val_loss/len(valloader)

    def validation_step(self, data,target):
        self.eval()
        with torch.no_grad():

            output = self(data)
            output = output.reshape(output.shape[0], self.tau)

            val_loss = self.loss(output, target)

        return val_loss.item()

    def set_seed(self) -> None:
        np.random.seed(self.seed)
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        # torch.cuda.manual.seed(self.seed)
        # When running on the CuDNN backend, two further options must be set
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Set a fixed value for the hash self.seed
        os.environ["PYTHONHASHSEED"] = str(self.seed)
        # print(f"Random self.seed set as {self.seed}")

    def fit(self, train_set, val_set):

        self.set_seed()

        self.optimizer = getattr(torch.optim,'Adam')(self.parameters(), lr=self.lr)
        train_set = torch.from_numpy(train_set.astype('float32').values)
        val_set = torch.from_numpy(val_set.astype('float32').values)

        train_data = [[train_set[i: i + self.T], train_set[i+self.T: i + self.T + self.tau][:, 0]] for i in
                      range(0,len(train_set) - self.T - self.tau,self.tau)]
        val_data = [[val_set[i: i + self.T], val_set[i + self.T: i + self.T + self.tau][:, 0]] for i in
                    range(0,len(val_set) - self.T - self.tau,self.tau)]

        train_set=torch.utils.data.TensorDataset(torch.stack([elt[0] for elt in train_data]),torch.stack([elt[1] for elt in train_data]))
        trainloader = torch.utils.data.DataLoader(train_set, batch_size=self.batch_size,shuffle=True)

        val_set = torch.utils.data.TensorDataset(torch.stack([elt[0] for elt in val_data]),torch.stack([elt[1] for elt in val_data]))
        valloader = torch.utils.data.DataLoader(val_set, batch_size=self.batch_size, shuffle=True)

        self.train_(trainloader,valloader)




if __name__ == '__main__':

    # Hyperparameters definition

    n_signals = 8
    window_size = 40
    kernel_size = 7 #7
    n_hidden_layers = 2
    n_hidden_dimensions = 5
    prediction_horizon = 2

    model = PSTA_TCN(
        window_size=window_size,
        kernel_size=kernel_size,
        n_hidden_layers=n_hidden_layers,
        n_hidden_dimensions=n_hidden_dimensions,
        n_signals=n_signals,
        prediction_horizon=prediction_horizon,
        batch_size=100,
        number_TCN=3,
        seed=50,
        dropout_rate=0.1,
        nb_epochs=1,
        conv_dilatation=2,
    )

    data_set = pd.read_csv(r'Dataset/exchange_rate.csv')
    train_val_data = data_set[: int(0.95 * len(data_set))]

    validation_data = train_val_data[int(0.8 * len(train_val_data)):].copy()
    train_data = train_val_data[: int(0.8 * len(train_val_data))]

    test_data = data_set[int(0.95 * len(data_set)):]

    model.fit(train_data,validation_data)


