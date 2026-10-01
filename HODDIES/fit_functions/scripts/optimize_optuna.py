"""
Optuna example that optimizes multi-layer perceptrons using PyTorch.

In this example, we optimize the validation accuracy of fashion product recognition using
PyTorch and FashionMNIST. We optimize the neural network architecture as well as the optimizer
configuration. As it is too time consuming to use the whole FashionMNIST dataset,
we here use a small subset of it.

"""

import os
import optuna
from optuna.trial import TrialState
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import glob
from pycorr.utils import cov_to_corrcoef
from torch.utils.data import Dataset, random_split, DataLoader, TensorDataset
from typing import OrderedDict
from torch.optim.lr_scheduler import ReduceLROnPlateau

import time
from torch.cuda.amp import autocast, GradScaler


class Normalizer:
    def __init__(self, X, y, log_transform=False):
        self.log_transform = log_transform

        # Optionally transform the output space
        if log_transform:
            y = np.arcsinh(y)

        # Compute statistics
        self.mean_X = X.mean(axis=0)
        self.std_X  = X.std(axis=0)

        self.mean_y = y.mean(axis=0)
        self.std_y  = y.std(axis=0)

    def normalize_x(self, X):
        return (X - self.mean_X) / self.std_X

    def normalize_y(self, y):
        if self.log_transform:
            y = np.arcsinh(y)
        return (y - self.mean_y) / self.std_y

    def denormalize_y(self, y_norm, var_norm=None):
        y = y_norm * self.std_y + self.mean_y

        if var_norm is not None:
            var = var_norm * (self.std_y ** 2)
        else:
            var = None

        if self.log_transform:
            y = np.sinh(y)

        return y, var

    def denormalise_x(self, x_pred_norm):
        return x_pred_norm * self.std_X + self.mean_X



class Training_DatasetManager(Dataset):

    def __init__(self, dir_path:str, path_to_test_files=None, stats = ['wp', 'xi'], log_transform=False, seed=None):
        self.dir_path = dir_path
        self.files = [os.path.join(self.dir_path,f) for f in os.listdir(self.dir_path) if f.endswith(".npy")] # Get the name of all the files in the directory
        self.files.sort() # Sort the files to have a reproducible order
        self.data = [] # This will be used for the storage of the data in memory for the training
        self.seed = seed if seed is not None else np.random.randint(0, 2**32 - 1)
        self.generator = torch.Generator().manual_seed(self.seed)
        self.s = None # Value of x (rp and s)
        self.rp = None
        self.stats = stats
        self.path_to_test_files = path_to_test_files
        
        self.idx = []
        self.log_transform = log_transform

        self.X_training, self.y_training = self.load_data(self.files)

        if self.log_transform :
            self.min_y_train_value = self.y_training.min()
            self.y_training = np.log10(self.y_training - self.min_y_train_value)
            

        # Normalisation min-max
        # self.normalise_x()
        # self.normalise_y()
        self.normaliser = Normalizer(self.X_training, self.y_training, log_transform=self.log_transform)
        self.x_norm = self.normaliser.normalize_x(self.X_training)
        self.y_norm = self.normaliser.normalize_y(self.y_training)
        
        self.data_train = [(self.x_norm[ii], self.y_norm[ii]) for ii in range(self.y_norm.shape[0])]
        if self.path_to_test_files is not None:
            self.load_test_set(self.path_to_test_files)
        else:
            self.X_test = None
            self.y_test = None
            self.x_test_norm = None

    def load_data(self, files):
        print(f"Loading data from {os.path.dirname(files[0])} dir_path for {len(files)} files...")
        from tqdm import tqdm
        estimated_total = len(files)
        progress_bar = tqdm(total=estimated_total)
        y_training = []
        X_training = []
        yerr  = []
        for ii,file in enumerate(files):
            res_param =  np.load(file, allow_pickle=True)[()]
            if ii == 0:
                self.name_arr = list(res_param['hod_fit_param'].dtype.names)
                if 'xi' in self.stats:
                    self.s =  0.5*(res_param['smu_bins'][0][:-1] + res_param['smu_bins'][0][1:])
                if 'wp' in self.stats:
                    self.rp = 0.5*(res_param['rppi_bins'][0][:-1] + res_param['rppi_bins'][0][1:])

            comb_trs = res_param[self.stats[0]][0].keys() 
            nreal = len(res_param[self.stats[0]])
            
            res = [np.hstack([np.hstack([np.hstack(res_param[stat][i][comb_tr][1]) for stat in self.stats]) for comb_tr in comb_trs]) for i in range(nreal)]
            hod_param = res_param['hod_fit_param']
            # y_training.append(np.mean(res, axis=0))
            X_training.append(hod_param)
            y_training.append(res[0])
            if nreal >1 :
                yerr.append(np.std(res, axis=0))
            progress_bar.update(1)

        y_training = np.c_[y_training]
        X_training = np.vstack(np.hstack(X_training).tolist())
        if nreal >1 :
            self.yerr_training = np.c_[yerr]
        else:
            self.yerr_training = None
        # ystd_training = np.vstack(ystd_training) if nreal >1 else None
        return X_training, y_training

    def normalise_x(self):
        self.min_X = self.X_training.min(axis=0)
        self.max_X = self.X_training.max(axis=0)
        self.x_norm = (self.X_training - self.min_X) / (self.max_X - self.min_X)

    def normalise_y(self):
        self.min_y = self.y_training.min(axis=0)[0]
        self.max_y = self.y_training.max(axis=0)[0]
        self.y_norm = (self.y_training - self.min_y) / (self.max_y - self.min_y)

    # def denormalise_x(self, x_pred):
    #     return x_pred * (self.max_X - self.min_X) + self.min_X
    
    def denormalise_x(self, x_pred):
        return self.normaliser.denormalise_x(x_pred)

    def denormalise_y(self, y_pred, var=None):
        denorm_y, denorm_var = self.normaliser.denormalize_y(y_pred, var)
        if var is None:
            return denorm_y
        else:
            return denorm_y, denorm_var
    
    # def denormalise_y(self, y_pred):
    #     denorm_y = y_pred * (self.max_y - self.min_y) + self.min_y
    #     if self.log_transform :
    #         denorm_y = np.sinh(denorm_y)
    #     return denorm_y

    def __len__(self):
        return len(self.data_train)

    def __getitem__(self, idx):
        
        values, y = self.data_train[idx]
    
        return torch.tensor(values, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

    def extract_data(self):
        X_t, y_t = [], []
        for i in range(len(self)):
            xi, yi = self[i]
    
            X_t.append(xi)
            y_t.append(yi)

        return torch.stack(X_t), torch.stack(y_t)

    def get_train_val_sets(self, train_frac=0.8):
        """
        Return the training and validation dataset for the training of the neural net.
        
        PARAMETERS:
        -----------
        train_frac : float 
            
    
        RETURN:
        ------
            (train_dataset, val_dataset): deux sous-ensembles TensorDataset
        """
        X, y = self.extract_data()

        dataset = TensorDataset(X, y)
    
        # Split train/val
        train_size = int(train_frac * len(dataset))
        val_size = len(dataset) - train_size
            
        return random_split(dataset, [train_size, val_size], generator=self.generator)

    def load_test_set(self, testset_path):
        """
        Return the test dataset for the training of the neural net.
        """

        files = [os.path.join(testset_path,f) for f in os.listdir(testset_path) if f.endswith(".npy")] # Get the name of all the files in the directory
        files.sort()
        self.X_test, self.y_test = self.load_data(files)

        # self.x_test_norm = (self.X_test - self.min_X) / (self.max_X - self.min_X)
        # self.y_test_norm = (self.y_test - self.min_y) / (self.max_y - self.min_y)

        self.x_test_norm = self.normaliser.normalize_x(self.X_test)
        self.y_test_norm = self.normaliser.normalize_y(self.y_test)
        # print('Warning remove 2.5% of the edges for the test set')

        # self.mask_test = np.all([(self.x_test_norm > 0.025), (self.x_test_norm < 0.975)], axis=0).all(axis=1)
        # self.mask_test = np.ones_like(self.X_test).astype(bool)
        # self.X_test = self.X_test[self.mask_test]
        # self.y_test = self.y_test[self.mask_test]

        # self.x_test_norm = self.x_test_norm[self.mask_test]
        # self.y_test_norm = self.y_test_norm[self.mask_test]

    def plot_training_data_distribution(self):
        import matplotlib.pyplot as plt
        import seaborn as sns
        import pandas as pd

        df = pd.DataFrame(self.X_training, columns=self.name_arr)
        sns.pairplot(df)
        plt.show()


class FCNN(nn.Module):
    """
    Fully Connected Neural Network
    """
    def __init__(self,
                 n_input : int,
                 n_output : int,
                 n_hidden: list[int] =[512, 512, 512, 512],
                 activation_fn = 'ReLU',
                 loss: str = 'rmse',
                 learning_rate: float = 1.e-3,
                 dropout_rate: float = 0.0,
                 weight_decay=2.5e-6,
                 device: str = 'cpu',
                 var_loss_weight: float = 1.0,
                 verbose=True,
                 use_std=False
                ):
        """
        Initialize the FCNN model.

        PARAMETERS:
        -----------
        n_input : int
            Number of input features (HOD parameters).
        n_output : int
            Number of output values (e.g. number of wp bins).
        n_hidden : List[int]
            List of hidden layer sizes.
        activation_fn : str
            Activation function name (e.g., 'ReLU', 'SiLU').
        loss : str
            Type of loss function ('mse', 'rmse', 'mae').
        learning_rate : float
            Learning rate for the optimizer.
        dropout_rate : float
            Dropout rate between layers (0.0 disables it).
        device : str
            'cpu' or 'cuda' (for GPU support).
        """
        super().__init__()
        self.n_input = n_input
        self.n_output = n_output
        self.n_hidden = n_hidden
        self.learning_rate = learning_rate
        self.activation_fn = activation_fn
        self.loss_type = loss
        self.device = torch.device(device)
        self.dropout_rate = dropout_rate
        self.var_loss_weight = var_loss_weight
        self.weight_decay = weight_decay
        self.use_std = use_std
        
        if self.loss_type == "learned_gaussian":
            self.n_output *= 2 # Prediction of the mean and prediction variance of each bin
        else:
            self.n_output *= 1 # Prediction of the mean of each bin

        self.model = self._build_mlp()
        self.to(self.device) 
        
        self.loss_fn = self._get_loss_fn()
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        # *** AMP ***
        self.use_amp = (self.device.type == "cuda")
        self.scaler  = GradScaler(enabled=self.use_amp)

        


    def _get_activation(self, layer_index: int):
        """ Returns the activation function for the given layer index. """
        return getattr(nn, self.activation_fn)()
        

    def _build_mlp(self):
        """
        Build a multi-layer perceptron (MLP) dynamically with optional dropout.
    
        PARAMETERS:
        -----------
        n_input : int
            Number of input features.
        n_hidden : list[int]
            List of hidden layer sizes.
        n_output : int
            Number of output features (e.g. length of wp).
        dropout_rate : float
            Dropout rate to apply after each activation layer (0.0 to disable).
    
        RETURNS:
        --------
        nn.Sequential
            A complete PyTorch MLP model.
        """
        model = nn.Sequential(OrderedDict())  # Create an ordered container for the layers

        last_dim = self.n_input  # Start with input size

        for i, hidden_dim in enumerate(self.n_hidden):
            # Create unique names for layers
            layer_name = f"mlp{i}"
            act_name = f"act{i}"
            dropout_name = f"dropout{i}"
    
            # Linear layer: input → hidden_dim
            linear_layer = nn.Linear(last_dim, hidden_dim)
            # Activation function: e.g., ReLU, SiLU, or LearnedSigmoid
            activation = self._get_activation(i)
    
            # Add layers to the model
            model.add_module(layer_name, linear_layer)
            model.add_module(act_name, activation)
            if self.dropout_rate > 0.0:
                model.add_module(dropout_name, nn.Dropout(self.dropout_rate))
    
            # Update input size for the next layer
            last_dim = hidden_dim
    
        # Final layer: last hidden size → output
        final_layer_name = f"mlp{len(self.n_hidden)}"
        model.add_module(final_layer_name, nn.Linear(last_dim, self.n_output))
    
        return model


    def forward(self, x):
        """
        Forward pass of the neural network.
    
        Returns:
        -------
        If loss is 'learned_gaussian': tuple of (prediction, variance)
        Else: prediction, zeros_like(prediction)
        """
        out = self.model(x)

        if self.loss_type == "learned_gaussian":
            # On sépare le vecteur de sortie en moyenne et variance
            mean, log_var = torch.chunk(out, 2, dim=-1)
            var = nn.functional.softplus(log_var) 
            return mean, var
        else:
            mean = out
            var = torch.zeros_like(mean, device=mean.device)
            return mean, var


    def compute_loss(self, X, y_true):
        """
        Compute the total loss:
        - prediction loss (wp) + supervised std loss (from mocks)
    
        Parameters
         ----------
        X : torch.Tensor
                Input features (batch_size, n_input)
        y_true : torch.Tensor
            Ground truth for wp (batch_size, n_output)
    
        Returns
        -------
        torch.Tensor
            Scalar loss value
        """     
        if self.loss_type == "learned_gaussian":
            preds, var_pred = self.forward(X)
            # loss = nn.GaussianNLLLoss(full=True)(preds, y_true, var_pred)
            # print(var_pred[0].shape, var_pred.shape, preds.shape, y_true[:,0].shape, y_true[:,1].shape)
            # loss = nn.GaussianNLLLoss(full=True)(torch.rand(var_pred.shape, device=self.device)*var_pred+preds, y_true[:,0], y_true[:,1]) # Sample from mean+pred to reduce both
            if self.use_std:
                preds = torch.rand(preds.shape, device=self.device)*var_pred+preds
                loss = nn.GaussianNLLLoss(full=True)(preds, y_true, var_pred)
            else:
                loss = nn.GaussianNLLLoss(full=True)(preds, y_true, var_pred) # Sample from mean+pred to reduce both

        else:
            preds, _ = self.forward(X)
            loss = self.loss_fn(preds, y_true)               
        return loss


    def _get_loss_fn(self):
        """
        Return the appropriate loss function based on self.loss_type.
        """
        if self.loss_type == "mse":
            return nn.MSELoss()
        elif self.loss_type == "rmse":
            return lambda y, y_pred: torch.sqrt(nn.MSELoss()(y, y_pred))
        elif self.loss_type == "mae":
            return nn.L1Loss()
        elif self.loss_type == "learned_gaussian":
            return nn.GaussianNLLLoss(full=True)
        else:   
            raise NotImplementedError(f"Loss '{self.loss_type}' is not implemented.")

    def predict(self, X, no_grad: bool = True):
        """
        Predict output values from input HOD parameters.
    
        PARAMETERS:
        -----------
        X : Tensor
            Input tensor of shape (batch_size, n_input)
        no_grad : bool
            Whether to disable gradient tracking (default: True)
    
        RETURNS:
        --------
        Tensor: Predicted output of shape (batch_size, n_output)
        """
        self.eval()
        X = X.to(self.device)
    
        if no_grad:
            with torch.no_grad():
                preds, var = self.forward(X)
        else:
            preds, var = self.forward(X)
    
        return preds, var



    def train_epoch(self, dataloader):
        self.train()
        total_loss = 0.0
    
        for X_batch, y_batch in dataloader:
            # 1) on déplace les données une seule fois
            X_batch = X_batch.to(self.device, non_blocking=True)
            y_batch = y_batch.to(self.device, non_blocking=True)
    
            self.optimizer.zero_grad(set_to_none=True)
    
            # 2) forward + backward en FP16/BF16 si GPU
            with autocast(enabled=self.use_amp):
                loss = self.compute_loss(X_batch, y_batch)
    
            # 3) mise à jour AMP
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
    
            total_loss += loss.item()
    
        return total_loss / len(dataloader)


    def fit(self,
        train_loader,
        val_loader,
        min_epochs: int = 100,
        max_epochs: int = 5000,
        patience: int = 30,          # plateau patience (early stopping)
        verbose: bool = True,
        optimizer=None):
        """
        Train the model with early stopping based on validation loss.
        The model will train at least `min_epochs`, and at most `max_epochs`.

        Parameters
        ----------
        train_loader : DataLoader
        val_loader   : DataLoader
        min_epochs   : int
            Minimum number of epochs before early stopping is allowed.
        max_epochs   : int
            Maximum number of epochs to train.
        patience     : int
            Number of epochs with no improvement before stopping.
        scheduler    : PyTorch scheduler or None
            If using ReduceLROnPlateau, pass e.g.:
                scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=10)
        """

        if optimizer is not None:
            self.optimizer = optimizer

        # IMPORTANT: create scheduler once
        scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=10, verbose=True)
        # self.scheduler = scheduler

        train_losses = []
        val_losses   = []

        best_val_loss = float("inf")
        best_state = None
        epochs_no_improve = 0  # for plateau tracking

        start_time = time.time()

        for epoch in range(max_epochs):

            train_loss = self.train_epoch(train_loader)
            val_loss = self.evaluate(val_loader)

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            if verbose and epoch % 10 == 0:
                print(f"Epoch {epoch+1}/{max_epochs} - "
                    f"Train: {train_loss:.5f} | Val: {val_loss:.5f}")

            # ---- LR Scheduler step ----
            scheduler.step(val_loss)
    
            # ---- Track best model ----
            if val_loss < best_val_loss - 1e-7:  # small tolerance
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in self.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            # ---- EARLY STOPPING ----
            if epoch + 1 >= min_epochs and epochs_no_improve >= patience:
                if verbose:
                    print(f"\n⛔ Early stopping triggered at {epoch}: no improvement for {patience} epochs.")
                break

        # Restore best weights
        if best_state is not None:
            self.load_state_dict(best_state)

        if verbose:
            print(f"\nTraining completed in {time.time() - start_time:.1f}s "
                f"| Best Val Loss: {best_val_loss:.5f}")

        return train_losses, val_losses


    # def fit(self,
    #         train_loader: torch.utils.data.DataLoader,
    #         val_loader: torch.utils.data.DataLoader,
    #         num_epochs: int = 300,
    #         verbose: bool = True, # verbose is used to control the printing, if verbose = False the print doesn't appears
    #         optimizer: bool = None,
    #         scheduler: bool = None
            
    #     ):
    #     """
    #     Train the model for multiple epochs.
    
    #     PARAMETERS:
    #     -----------
    #     train_loader : DataLoader
    #         Dataloader for training set
    #     val_loader : DataLoader
    #         Dataloader for validation set
    #     num_epochs : int
    #         Number of training epochs
    #     verbose : bool
    #         Whether to print loss at each epoch
    
    #     RETURNS:
    #     --------
    #     (train_losses, val_losses): tuple of lists of float
    #     """
    #     tt = time.time()
    #     train_losses = []
    #     val_losses = []

    #     if optimizer is not None:
    #         self.optimizer = optimizer

    #     for epoch in range(num_epochs):
    #         train_loss = self.train_epoch(train_loader)
    #         val_loss = self.evaluate(val_loader)
    
    #         train_losses.append(train_loss)
    #         val_losses.append(val_loss)
    
    #         if verbose:
    #             if epoch%100 ==0:
    #                 print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")

    #         if scheduler is not None:
    #             scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=10, verbose=True)
    #     print("Training complete in {} s.".format(tt-time.time()))
    #     return train_losses, val_losses


    @torch.no_grad()
    def evaluate(self, dataloader):
        self.eval()
        total_loss = 0.0
    
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(self.device, non_blocking=True)
            y_batch = y_batch.to(self.device, non_blocking=True)
    
            with autocast(enabled=self.use_amp):
                loss = self.compute_loss(X_batch, y_batch)
            total_loss += loss.item()
    
        return total_loss / len(dataloader)

    
    def save_model(self, path: str = "model.pth"):
        """
        Save the model weights to a file.
    
        PARAMETERS:
        -----------
        path : str
            Path to the output file (default: 'model.pth')
        """
        torch.save(self.state_dict(), path)


    def load_model(self, path: str):
        """
        Load model weights from a file.
    
        PARAMETERS:
        -----------
        path : str
            Path to the file where weights were saved
        """
        self.load_state_dict(torch.load(path, map_location=self.device))
        self.to(self.device)



def make_training_dataset(dir_path:str, stats = ['wp', 'xi'], log_transform=False, seed=None, path_to_test_files=None, batch_size=256):
    """
    This function Will creat the training dataset for the training of the neural network.

    PARAMETER:
    ---------
    dataset_path : str
        The path of the training dataset.
    normalization_cst_name : str
        The name of the normalization constants.

    RETURNS:
    --------
    train_loader : 
        Training loader for the training of the neural network.
    val_loader : 
        Validation loader for the training of the neural network.
    """

    train_Dataset = Training_DatasetManager(dir_path, stats=stats, log_transform=log_transform, seed=seed, path_to_test_files=path_to_test_files)

    X, y = train_Dataset.extract_data()
    train_dataset, val_dataset = train_Dataset.get_train_val_sets()
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,                
        shuffle=True,
        num_workers=0,                
        pin_memory=True,                
        persistent_workers=False,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        persistent_workers=False,
    )
    

    # No normalisation cst

    return train_Dataset, train_loader, val_loader

def train_model(
    train_loader,
    val_loader,
    n_hidden:list[int] = [128,128,128],
    Activation_fn:str = "SiLU", 
    Learning_rate:float = 3e-4, 
    Dropout_rate:float = 0.05, 
    weight_decay=2.5e-6,
    min_epochs:int = 100,
    max_epochs:int = 5000,
    loss="learned_gaussian",
    Model_saving_path:str = None, 
    use_std=True):
    """
    This function train the neural network.
    
    PARAMETERS:
    -----------
    Model_saving_path : str
        The saving path of the model.
    train_loader :
        The training loader.
    val_loader : 
        The validation loader.
    n_hidden_layers : int
        The number of hidden layers in the model.
    Activation_fn : str
        The activation function of the model.
    Learning_rate : float
        The learning rate of the model.
    Dropout_rate : float
        The dopout rate of the model.
    Train_epoch : int
        The number of epoch of training.
    

    RETURNS:
    -------
    model : 
        The trained model ready to be used
    Train_losses : 
        The values of the training losses over the epoch of training.
    val_losses : 
        The values of the validation losses over the epoch of training.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    nb_outpout = train_loader.dataset[0][1].shape[-1]
    nb_input = train_loader.dataset[0][0].shape[-1]
    
    # Hidden_layers = []

    # for i in range(n_hidden_layers):
    #     Hidden_layers.append(nb_nerons)
    
    model = FCNN(
        n_input=nb_input,
        n_output=nb_outpout,
        # n_hidden=[512, 512, 512],
        # n_hidden=[1024, 1024, 1024, 1024],
        #n_hidden=[64, 64, 64],
        #n_hidden=[128, 128, 128, 128],
        # n_hidden=[128, 128, 128, 128, 128],
        n_hidden=n_hidden,
        # n_hidden=[128, 128], # model 10
        # n_hidden=[64, 64], # model 12
        activation_fn=Activation_fn,
        #activation_fn="ReLU",
        loss=loss,
        #learning_rate=0.009332352540651494,
        # learning_rate=0.0002249937260017888,
        learning_rate=Learning_rate,
        #dropout_rate=0.010011267028423554,
        #dropout_rate=0.027582214112809256,
        # dropout_rate=0.01, # model 4
        # dropout_rate=0.02,
        # dropout_rate=0.0, # model 10
        dropout_rate=Dropout_rate,
        weight_decay=weight_decay,
        device=device,
        use_std=use_std
    ).to(device)
    
    # optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=2.5e-6)
    
    # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
    
    model = torch.compile(model)    

    
    train_losses, val_losses = model.fit(
        train_loader, val_loader, min_epochs=min_epochs, max_epochs=max_epochs)
    
    if Model_saving_path is not None:
        model.save_model(Model_saving_path)

    return model, train_losses, val_losses



def objective(trial, train_loader=None, val_loader=None, loss="learned_gaussian"):

    # Get the dataset.
    # dataset_path='/pscratch/sd/a/arocher/Y3_hod_fits/training_Hammersley/LRG/AbacusSummit_highbase_c000_ph100/z0.500/LRG_SHOD_6p'
    # testset_path='/pscratch/sd/a/arocher/Y3_hod_fits/test_set/LRG/AbacusSummit_highbase_c000_ph100/z0.500/LRG_SHOD_6p'
    # train_Dataset, train_loader, val_loader = make_training_dataset(dataset_path, stats=['wp', 'xi'], log_transform=False, path_to_test_files=testset_path, seed=42)
    
    # Generate the model.
    st = time.time()
    same_n_hidden = False
    weight_decay = trial.suggest_float("weight_decay", 1.0e-5, 0.001)
    n_layers = trial.suggest_int("n_layers", 1, 10)
    if same_n_hidden:
        n_hidden = [trial.suggest_int("n_hidden", 128, 1024)] * n_layers
    else:
        n_hidden = [
            trial.suggest_int(f"n_hidden_{layer}", 128, 1024)
            for layer in range(n_layers)
        ]
    dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.15)
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)


    model, train_losses, val_losses = train_model(
    train_loader,
    val_loader,
    n_hidden = n_hidden, 
    Activation_fn = "SiLU", 
    Learning_rate=lr, 
    Dropout_rate=dropout_rate, 
    weight_decay=weight_decay,
    loss=loss, use_std=True)
    
    # Z_score, Chi_square = Z_score_and_chi_square_calculation(model, train_Dataset)
    # Z_score_std = np.std(Z_score, axis=1)
    # Z_score_mean = np.mean(Z_score, axis=1)
    # res = np.sqrt(Z_score_mean.mean()**2 + (Z_score_std.std()-1)**2)
    # print(f'Trial {trial.number}: Z_score mean {np.mean(Z_score_mean)}, Z_score std {np.mean(Z_score_std)}, took {time.time() - st} sec')
    return np.min(val_losses)


if __name__ == "__main__":
    from functools import partial
    dataset_path='/pscratch/sd/a/arocher/Y3_hod_fits/training_Hammersley/LRG/AbacusSummit_highbase_c000_ph100/z0.500/LRG_SHOD_6p'
    testset_path='/pscratch/sd/a/arocher/Y3_hod_fits/test_set/LRG/AbacusSummit_highbase_c000_ph100/z0.500/LRG_SHOD_6p'
    train_Dataset, train_loader, val_loader = make_training_dataset(dataset_path, stats=['wp', 'xi'], log_transform=False, path_to_test_files=testset_path, seed=42)
    
    study = optuna.create_study(direction="minimize")
    study.optimize(partial(objective, train_loader=train_loader, val_loader=val_loader, loss="learned_gaussian"), n_trials=500)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    keys, values = [], []
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
        keys += [key]
        values += [value]
    np.save('/global/homes/a/arocher/Code/postdoc/HOD/Cosmological_emulator_ELGs/best_train_model.npy', dict(zip(keys, values)))