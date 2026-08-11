# import numpy as np
# import os
# import torch
# from torch import nn
# import glob
# from pycorr.utils import cov_to_corrcoef
# from torch.utils.data import Dataset, random_split, DataLoader, TensorDataset
# from typing import OrderedDict
# from torch.optim.lr_scheduler import ReduceLROnPlateau



# class Normalizer:
#     def __init__(self, X, y, log_transform=False):
#         self.log_transform = log_transform

#         # Optionally transform the output space
#         if log_transform:
#             y = np.arcsinh(y)

#         # Compute statistics
#         self.mean_X = X.mean(axis=0)
#         self.std_X  = X.std(axis=0)

#         self.mean_y = y.mean(axis=0)
#         self.std_y  = y.std(axis=0)

#     def normalize_x(self, X):
#         return (X - self.mean_X) / self.std_X

#     def normalize_y(self, y):
#         if self.log_transform:
#             y = np.arcsinh(y)
#         return (y - self.mean_y) / self.std_y

#     def denormalize_y(self, y_norm, var_norm=None):
#         y = y_norm * self.std_y + self.mean_y

#         if var_norm is not None:
#             var = var_norm * (self.std_y ** 2)
#         else:
#             var = None

#         if self.log_transform:
#             y = np.sinh(y)

#         return y, var

#     def denormalise_x(self, x_pred_norm):
#         return x_pred_norm * self.std_X + self.mean_X



# class Training_DatasetManager(Dataset):

#     def __init__(self, dir_path:str, path_to_test_files=None, stats = ['wp', 'xi'], log_transform=False, seed=None):
#         self.dir_path = dir_path
#         self.files = [os.path.join(self.dir_path,f) for f in os.listdir(self.dir_path) if f.endswith(".npy")] # Get the name of all the files in the directory
#         self.files.sort() # Sort the files to have a reproducible order
#         self.data = [] # This will be used for the storage of the data in memory for the training
#         self.seed = seed if seed is not None else np.random.randint(0, 2**32 - 1)
#         self.generator = torch.Generator().manual_seed(self.seed)
#         self.sep = {} # Value of x (rp and s)
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.device = torch.device(device)
#         self.stats = stats
#         self.path_to_test_files = path_to_test_files
        
#         self.idx = []
#         self.log_transform = log_transform

#         self.X_training, self.y_training, self.yerr_training = self.load_data(self.files)

#         if self.log_transform :
#             self.min_y_train_value = self.y_training.min()
#             self.y_training = np.log10(self.y_training - self.min_y_train_value)
            

#         # Normalisation min-max
#         # self.normalise_x()
#         # self.normalise_y()
#         self.normaliser = Normalizer(self.X_training, self.y_training, log_transform=self.log_transform)
#         self.x_norm = self.normaliser.normalize_x(self.X_training)
#         self.y_norm = self.normaliser.normalize_y(self.y_training)
        
#         self.data_train = [(self.x_norm[ii], self.y_norm[ii]) for ii in range(self.y_norm.shape[0])]
#         if self.path_to_test_files is not None:
#             self.load_test_set(self.path_to_test_files)
#         else:
#             self.X_test = None
#             self.y_test = None
#             self.x_test_norm = None

#     def check_stat(stat):
#         stat = list(stat)

#         # Vérifier que tout est autorisé
#         unknown = set(stat) - set(AVAIL_STAT)
#         if unknown:
#             raise ValueError(f"Stat(s) {unknown} not available. Available stats are {AVAIL_STAT}")

#         # Familles xi
#         xi_ell  = [s for s in stat if re.match(r'xi_\d+$', s)]
#         xi_smu  = [s for s in stat if s == 'xi_smu']
#         xi_rppi = [s for s in stat if s == 'xi_rppi']

#         n_families = sum(bool(x) for x in [xi_ell, xi_smu, xi_rppi])

#         if n_families > 1:
#             raise ValueError(
#                 "Incompatible xi statistics: "
#                 "choose either xi_ell (xi_0, xi_2, ...), "
#                 "xi_smu, or xi_rppi — not a mix."
#             )

#         return True

#     def load_data(self, files):
#         print(f"Loading data from {os.path.dirname(files[0])} dir_path for {len(files)} files...")
#         from tqdm import tqdm
#         estimated_total = len(files)
#         progress_bar = tqdm(total=estimated_total)
#         y_training = []
#         X_training = []
#         yerr  = []
#         self.bin_num = {}
#         for ii,file in enumerate(files):
#             res_param = np.load(file, allow_pickle=True)[()]
#             if ii == 0:
#                 self.name_arr = list(res_param['hod_fit_param'].dtype.names)
#                 for stat in self.stats:
#                     if stat == 'xi':
#                         for ell in np.arange(0, len(list(res_param[stat][0].values())[0][1])+1, 2):
#                             self.sep[f'xi_{ell}'] = list(res_param[stat][0].values())[0][0]
#                             if len(self.bin_num) ==0:
#                                 last_val = 0
#                             else:
#                                 last_val = list(self.bin_num.values())[-1][-1]
#                             self.bin_num[f'xi_{ell}'] = [last_val, last_val+len(self.sep[f'xi_{ell}'])]
#                     else:
#                         self.sep[stat] = list(res_param[stat][0].values())[0][0]
#                         if len(self.bin_num) ==0:
#                             last_val = 0
#                         else:
#                             last_val = list(self.bin_num.values())[-1][-1]
#                         self.bin_num[stat] = [last_val, last_val+len(self.sep[stat])]
            
                
#             comb_trs = res_param[self.stats[0]][0].keys() 
#             nreal = len(res_param[self.stats[0]])            
#             res = [np.hstack([np.hstack([np.hstack(res_param[stat][i][comb_tr][1]) for stat in self.stats]) for comb_tr in comb_trs]) for i in range(nreal)]
#             hod_param = res_param['hod_fit_param']
#             # y_training.append(np.mean(res, axis=0))
#             X_training.append(hod_param)
#             y_training.append(res[0])
#             if nreal >1 :
#                 yerr.append(np.std(res, axis=0))
#             progress_bar.update(1)

#         y_training = np.c_[y_training]
#         X_training = np.vstack(np.hstack(X_training).tolist())
#         if nreal > 1:
#             yerr = np.c_[yerr]
#         else:
#             yerr = None
#         # ystd_training = np.vstack(ystd_training) if nreal >1 else None
#         return X_training, y_training, yerr

#     def normalise_x(self):
#         self.min_X = self.X_training.min(axis=0)
#         self.max_X = self.X_training.max(axis=0)
#         self.x_norm = (self.X_training - self.min_X) / (self.max_X - self.min_X)

#     def normalise_y(self):
#         self.min_y = self.y_training.min(axis=0)[0]
#         self.max_y = self.y_training.max(axis=0)[0]
#         self.y_norm = (self.y_training - self.min_y) / (self.max_y - self.min_y)

#     # def denormalise_x(self, x_pred):
#     #     return x_pred * (self.max_X - self.min_X) + self.min_X
    
#     def denormalise_x(self, x_pred):
#         return self.normaliser.denormalise_x(x_pred)

#     def denormalise_y(self, y_pred, var=None):
#         denorm_y, denorm_var = self.normaliser.denormalize_y(y_pred, var)
#         if var is None:
#             return denorm_y
#         else:
#             return denorm_y, denorm_var
        
#     def split_Y_into_stats(self ,Y):
#         """
#         Split a flat Y array into a dictionary of statistics.

#         Parameters
#         ----------
#         Y : ndarray, shape (n_samples, n_features)
#             The full output array (concatenate of all stats).
#         stats : list of str
#             Names of statistics in the order they appear in Y.
#         sep_dict : dict
#             sep_dict[stat] = array of separation values → defines the length.

#         Returns
#         -------
#         Y_dict : dict
#             Y_dict[stat] = sub-array for that statistic.
#         """

#         # Compute lengths of each statistic
#         stat_sizes = [len(self.sep[st]) for st in self.sep.keys()]
        
#         # Compute cumulative cut positions
#         cum = np.cumsum([0] + stat_sizes)

#         Y_dict = {}

#         for i, st in enumerate(self.sep.keys()):
#             a, b = cum[i], cum[i+1]
#             Y_dict[st] = Y[:, a:b]

#         return Y_dict

    
#     # def denormalise_y(self, y_pred):
#     #     denorm_y = y_pred * (self.max_y - self.min_y) + self.min_y
#     #     if self.log_transform :
#     #         denorm_y = np.sinh(denorm_y)
#     #     return denorm_y

#     def __len__(self):
#         return len(self.data_train)

#     def __getitem__(self, idx):
        
#         values, y = self.data_train[idx]
    
#         return torch.tensor(values, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

#     def extract_data(self):
#         X_t, y_t = [], []
#         for i in range(len(self)):
#             xi, yi = self[i]
    
#             X_t.append(xi)
#             y_t.append(yi)

#         return torch.stack(X_t).to(self.device), torch.stack(y_t).to(self.device)

#     def get_train_val_sets(self, train_frac=0.8):
#         """
#         Return the training and validation dataset for the training of the neural net.
        
#         PARAMETERS:
#         -----------
#         train_frac : float 
            
    
#         RETURN:
#         ------
#             (train_dataset, val_dataset): deux sous-ensembles TensorDataset
#         """
#         X, y = self.extract_data()

#         dataset = TensorDataset(X, y)
    
#         # Split train/val
#         train_size = int(train_frac * len(dataset))
#         val_size = len(dataset) - train_size
            
#         return random_split(dataset, [train_size, val_size], generator=self.generator)

#     def load_test_set(self, testset_path):
#         """
#         Return the test dataset for the training of the neural net.
#         """

#         # files = [os.path.join(testset_path,f) for f in os.listdir(testset_path) if f.endswith(".npy")] # Get the name of all the files in the directory
#         files = glob.glob(testset_path+'/lhs*')
#         files.sort()
#         self.X_test, self.y_test,self.yerr_test = self.load_data(files)

#         # self.x_test_norm = (self.X_test - self.min_X) / (self.max_X - self.min_X)
#         # self.y_test_norm = (self.y_test - self.min_y) / (self.max_y - self.min_y)

#         self.x_test_norm = self.normaliser.normalize_x(self.X_test)
#         self.y_test_norm = self.normaliser.normalize_y(self.y_test)
#         # print('Warning remove 2.5% of the edges for the test set')

#         # self.mask_test = np.all([(self.x_test_norm > 0.025), (self.x_test_norm < 0.975)], axis=0).all(axis=1)
#         # self.mask_test = np.ones_like(self.X_test).astype(bool)
#         # self.X_test = self.X_test[self.mask_test]
#         # self.y_test = self.y_test[self.mask_test]

#         # self.x_test_norm = self.x_test_norm[self.mask_test]
#         # self.y_test_norm = self.y_test_norm[self.mask_test]

#     def plot_training_data_distribution(self):
#         import matplotlib.pyplot as plt
#         import seaborn as sns
#         import pandas as pd

#         df = pd.DataFrame(self.X_training, columns=self.name_arr)
#         sns.pairplot(df)
#         plt.show()


# import numpy as np
# import os
# import torch
# from torch import nn
# from torch.utils.data import Dataset, random_split, DataLoader, TensorDataset
# from typing import OrderedDict
# from torch.optim.lr_scheduler import ReduceLROnPlateau
# import time
# from torch.cuda.amp import autocast, GradScaler


# class FCNN(nn.Module):
#     """
#     Fully Connected Neural Network
#     """
#     def __init__(self,
#                  n_input : int,
#                  n_output : int,
#                  n_hidden: list[int] =[512, 512, 512, 512],
#                  activation_fn = 'ReLU',
#                  loss: str = 'rmse',
#                  learning_rate: float = 1.e-3,
#                  dropout_rate: float = 0.0,
#                  weight_decay=2.5e-6,
#                  device: str = 'cpu',
#                  var_loss_weight: float = 1.0,
#                  verbose=True,
#                  use_std=False
#                 ):
#         """
#         Initialize the FCNN model.

#         PARAMETERS:
#         -----------
#         n_input : int
#             Number of input features (HOD parameters).
#         n_output : int
#             Number of output values (e.g. number of wp bins).
#         n_hidden : List[int]
#             List of hidden layer sizes.
#         activation_fn : str
#             Activation function name (e.g., 'ReLU', 'SiLU').
#         loss : str
#             Type of loss function ('mse', 'rmse', 'mae').
#         learning_rate : float
#             Learning rate for the optimizer.
#         dropout_rate : float
#             Dropout rate between layers (0.0 disables it).
#         device : str
#             'cpu' or 'cuda' (for GPU support).
#         """
#         super().__init__()
#         self.n_input = n_input
#         self.n_output = n_output
#         self.n_hidden = n_hidden
#         self.learning_rate = learning_rate
#         self.activation_fn = activation_fn
#         self.loss_type = loss
#         self.device = torch.device(device)
#         self.dropout_rate = dropout_rate
#         self.var_loss_weight = var_loss_weight
#         self.weight_decay = weight_decay
#         self.use_std = use_std
        
#         if self.loss_type == "learned_gaussian":
#             self.n_output *= 2 # Prediction of the mean and prediction variance of each bin
#         else:
#             self.n_output *= 1 # Prediction of the mean of each bin

#         self.model = self._build_mlp()
#         self.to(self.device) 
        
#         self.loss_fn = self._get_loss_fn()
#         self.optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

#         # *** AMP ***
#         self.use_amp = (self.device.type == "cuda")
#         self.scaler  = GradScaler(enabled=self.use_amp)

        


#     def _get_activation(self, layer_index: int):
#         """ Returns the activation function for the given layer index. """
#         return getattr(nn, self.activation_fn)()
        

#     def _build_mlp(self):
#         """
#         Build a multi-layer perceptron (MLP) dynamically with optional dropout.
    
#         PARAMETERS:
#         -----------
#         n_input : int
#             Number of input features.
#         n_hidden : list[int]
#             List of hidden layer sizes.
#         n_output : int
#             Number of output features (e.g. length of wp).
#         dropout_rate : float
#             Dropout rate to apply after each activation layer (0.0 to disable).
    
#         RETURNS:
#         --------
#         nn.Sequential
#             A complete PyTorch MLP model.
#         """
#         model = nn.Sequential(OrderedDict())  # Create an ordered container for the layers

#         last_dim = self.n_input  # Start with input size

#         for i, hidden_dim in enumerate(self.n_hidden):
#             # Create unique names for layers
#             layer_name = f"mlp{i}"
#             act_name = f"act{i}"
#             dropout_name = f"dropout{i}"
    
#             # Linear layer: input → hidden_dim
#             linear_layer = nn.Linear(last_dim, hidden_dim)
#             # Activation function: e.g., ReLU, SiLU, or LearnedSigmoid
#             activation = self._get_activation(i)
    
#             # Add layers to the model
#             model.add_module(layer_name, linear_layer)
#             model.add_module(act_name, activation)
#             if self.dropout_rate > 0.0:
#                 model.add_module(dropout_name, nn.Dropout(self.dropout_rate))
    
#             # Update input size for the next layer
#             last_dim = hidden_dim
    
#         # Final layer: last hidden size → output
#         final_layer_name = f"mlp{len(self.n_hidden)}"
#         model.add_module(final_layer_name, nn.Linear(last_dim, self.n_output))
    
#         return model


#     def forward(self, x):
#         """
#         Forward pass of the neural network.
    
#         Returns:
#         -------
#         If loss is 'learned_gaussian': tuple of (prediction, variance)
#         Else: prediction, zeros_like(prediction)
#         """
#         out = self.model(x)

#         if self.loss_type == "learned_gaussian":
#             # On sépare le vecteur de sortie en moyenne et variance
#             mean, log_var = torch.chunk(out, 2, dim=-1)
#             var = nn.functional.softplus(log_var) 
#             return mean, var
#         else:
#             mean = out
#             var = torch.zeros_like(mean, device=mean.device)
#             return mean, var


#     def compute_loss(self, X, y_true):
#         """
#         Compute the total loss:
#         - prediction loss (wp) + supervised std loss (from mocks)
    
#         Parameters
#          ----------
#         X : torch.Tensor
#                 Input features (batch_size, n_input)
#         y_true : torch.Tensor
#             Ground truth for wp (batch_size, n_output)
    
#         Returns
#         -------
#         torch.Tensor
#             Scalar loss value
#         """     
#         if self.loss_type == "learned_gaussian":
#             preds, var_pred = self.forward(X)
#             # loss = nn.GaussianNLLLoss(full=True)(preds, y_true, var_pred)
#             # print(var_pred[0].shape, var_pred.shape, preds.shape, y_true[:,0].shape, y_true[:,1].shape)
#             # loss = nn.GaussianNLLLoss(full=True)(torch.rand(var_pred.shape, device=self.device)*var_pred+preds, y_true[:,0], y_true[:,1]) # Sample from mean+pred to reduce both
#             if self.use_std:
#                 preds = torch.rand(preds.shape, device=self.device)*var_pred+preds
#                 loss = nn.GaussianNLLLoss(full=True)(preds, y_true, var_pred)
#             else:
#                 loss = nn.GaussianNLLLoss(full=True)(preds, y_true, var_pred) # Sample from mean+pred to reduce both

#         else:
#             preds, _ = self.forward(X)
#             loss = self.loss_fn(preds, y_true)               
#         return loss


#     def _get_loss_fn(self):
#         """
#         Return the appropriate loss function based on self.loss_type.
#         """
#         if self.loss_type == "mse":
#             return nn.MSELoss()
#         elif self.loss_type == "rmse":
#             return lambda y, y_pred: torch.sqrt(nn.MSELoss()(y, y_pred))
#         elif self.loss_type == "mae":
#             return nn.L1Loss()
#         elif self.loss_type == "learned_gaussian":
#             return nn.GaussianNLLLoss(full=True)
#         else:   
#             raise NotImplementedError(f"Loss '{self.loss_type}' is not implemented.")

#     def predict(self, X, no_grad: bool = True):
#         """
#         Predict output values from input HOD parameters.
    
#         PARAMETERS:
#         -----------
#         X : Tensor
#             Input tensor of shape (batch_size, n_input)
#         no_grad : bool
#             Whether to disable gradient tracking (default: True)
    
#         RETURNS:
#         --------
#         Tensor: Predicted output of shape (batch_size, n_output)
#         """
#         self.eval()
#         X = X.to(self.device)
    
#         if no_grad:
#             with torch.no_grad():
#                 preds, var = self.forward(X)
#         else:
#             preds, var = self.forward(X)
    
#         return preds, var



#     def train_epoch(self, dataloader):
#         self.train()
#         total_loss = 0.0
    
#         for X_batch, y_batch in dataloader:
#             # 1) on déplace les données une seule fois
#             X_batch = X_batch.to(self.device, non_blocking=True)
#             y_batch = y_batch.to(self.device, non_blocking=True)
    
#             self.optimizer.zero_grad(set_to_none=True)
    
#             # 2) forward + backward en FP16/BF16 si GPU
#             with autocast(enabled=self.use_amp):
#                 loss = self.compute_loss(X_batch, y_batch)
    
#             # 3) mise à jour AMP
#             self.scaler.scale(loss).backward()
#             self.scaler.step(self.optimizer)
#             self.scaler.update()
    
#             total_loss += loss.item()
    
#         return total_loss / len(dataloader)


#     def fit(self,
#         train_loader,
#         val_loader,
#         min_epochs: int = 100,
#         max_epochs: int = 5000,
#         patience: int = 30,          # plateau patience (early stopping)
#         verbose: bool = True,
#         optimizer=None):
#         """
#         Train the model with early stopping based on validation loss.
#         The model will train at least `min_epochs`, and at most `max_epochs`.

#         Parameters
#         ----------
#         train_loader : DataLoader
#         val_loader   : DataLoader
#         min_epochs   : int
#             Minimum number of epochs before early stopping is allowed.
#         max_epochs   : int
#             Maximum number of epochs to train.
#         patience     : int
#             Number of epochs with no improvement before stopping.
#         scheduler    : PyTorch scheduler or None
#             If using ReduceLROnPlateau, pass e.g.:
#                 scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=10)
#         """

#         if optimizer is not None:
#             self.optimizer = optimizer

#         # IMPORTANT: create scheduler once
#         scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=10, verbose=True)
#         # self.scheduler = scheduler

#         train_losses = []
#         val_losses   = []

#         best_val_loss = float("inf")
#         best_state = None
#         epochs_no_improve = 0  # for plateau tracking

#         start_time = time.time()

#         for epoch in range(max_epochs):

#             train_loss = self.train_epoch(train_loader)
#             val_loss = self.evaluate(val_loader)

#             train_losses.append(train_loss)
#             val_losses.append(val_loss)

#             if verbose and epoch % 10 == 0:
#                 print(f"Epoch {epoch+1}/{max_epochs} - "
#                     f"Train: {train_loss:.5f} | Val: {val_loss:.5f}")

#             # ---- LR Scheduler step ----
#             scheduler.step(val_loss)
    
#             # ---- Track best model ----
#             if val_loss < best_val_loss - 1e-7:  # small tolerance
#                 best_val_loss = val_loss
#                 best_state = {k: v.cpu().clone() for k, v in self.state_dict().items()}
#                 epochs_no_improve = 0
#             else:
#                 epochs_no_improve += 1

#             # ---- EARLY STOPPING ----
#             if epoch + 1 >= min_epochs and epochs_no_improve >= patience:
#                 if verbose:
#                     print(f"\n⛔ Early stopping triggered at {epoch}: no improvement for {patience} epochs.")
#                 break

#         # Restore best weights
#         if best_state is not None:
#             self.load_state_dict(best_state)

#         if verbose:
#             print(f"\nTraining completed in {time.time() - start_time:.1f}s "
#                 f"| Best Val Loss: {best_val_loss:.5f}")

#         return train_losses, val_losses


#     # def fit(self,
#     #         train_loader: torch.utils.data.DataLoader,
#     #         val_loader: torch.utils.data.DataLoader,
#     #         num_epochs: int = 300,
#     #         verbose: bool = True, # verbose is used to control the printing, if verbose = False the print doesn't appears
#     #         optimizer: bool = None,
#     #         scheduler: bool = None
            
#     #     ):
#     #     """
#     #     Train the model for multiple epochs.
    
#     #     PARAMETERS:
#     #     -----------
#     #     train_loader : DataLoader
#     #         Dataloader for training set
#     #     val_loader : DataLoader
#     #         Dataloader for validation set
#     #     num_epochs : int
#     #         Number of training epochs
#     #     verbose : bool
#     #         Whether to print loss at each epoch
    
#     #     RETURNS:
#     #     --------
#     #     (train_losses, val_losses): tuple of lists of float
#     #     """
#     #     tt = time.time()
#     #     train_losses = []
#     #     val_losses = []

#     #     if optimizer is not None:
#     #         self.optimizer = optimizer

#     #     for epoch in range(num_epochs):
#     #         train_loss = self.train_epoch(train_loader)
#     #         val_loss = self.evaluate(val_loader)
    
#     #         train_losses.append(train_loss)
#     #         val_losses.append(val_loss)
    
#     #         if verbose:
#     #             if epoch%100 ==0:
#     #                 print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")

#     #         if scheduler is not None:
#     #             scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=10, verbose=True)
#     #     print("Training complete in {} s.".format(tt-time.time()))
#     #     return train_losses, val_losses


#     @torch.no_grad()
#     def evaluate(self, dataloader):
#         self.eval()
#         total_loss = 0.0
    
#         for X_batch, y_batch in dataloader:
#             X_batch = X_batch.to(self.device, non_blocking=True)
#             y_batch = y_batch.to(self.device, non_blocking=True)
    
#             with autocast(enabled=self.use_amp):
#                 loss = self.compute_loss(X_batch, y_batch)
#             total_loss += loss.item()
    
#         return total_loss / len(dataloader)

    
#     def save_model(self, path: str = "model.pth"):
#         """
#         Save the model weights to a file.
    
#         PARAMETERS:
#         -----------
#         path : str
#             Path to the output file (default: 'model.pth')
#         """
#         torch.save(self.state_dict(), path)


#     def load_model(self, path: str):
#         """
#         Load model weights from a file.
    
#         PARAMETERS:
#         ----------- 
#         path : str
#             Path to the file where weights were saved
#         """
#         self.load_state_dict(torch.load(path, map_location=self.device))
#         self.to(self.device)




# from torch.utils.data import Dataset, random_split, DataLoader, TensorDataset
# from matplotlib import pyplot as plt

# def make_training_dataset(dir_path:str, stats = ['wp', 'xi'], log_transform=False, seed=None, path_to_test_files=None, batch_size=256):
#     """
#     This function Will creat the training dataset for the training of the neural network.

#     PARAMETER:
#     ---------
#     dataset_path : str
#         The path of the training dataset.
#     normalization_cst_name : str
#         The name of the normalization constants.

#     RETURNS:
#     --------
#     train_loader : 
#         Training loader for the training of the neural network.
#     val_loader : 
#         Validation loader for the training of the neural network.
#     """

#     train_Dataset = Training_DatasetManager(dir_path, stats=stats, log_transform=log_transform, seed=seed, path_to_test_files=path_to_test_files)

#     X, y = train_Dataset.extract_data()
#     train_dataset, val_dataset = train_Dataset.get_train_val_sets()
    
#     train_loader = DataLoader(
#         train_dataset,
#         batch_size=batch_size,                
#         shuffle=True,
#         num_workers=0,                
#         pin_memory=True,                
#         persistent_workers=False,
#     )
    
#     val_loader = DataLoader(
#         val_dataset,
#         batch_size=batch_size,
#         shuffle=False,
#         num_workers=0,
#         pin_memory=True,
#         persistent_workers=False,
#     )
    

#     # No normalisation cst

#     return train_Dataset, train_loader, val_loader

# def train_model(
#     train_loader,
#     val_loader,
#     n_hidden:list[int] = [128,128,128],
#     Activation_fn:str = "SiLU", 
#     Learning_rate:float = 3e-4, 
#     Dropout_rate:float = 0.05, 
#     weight_decay=2.5e-6,
#     min_epochs:int = 100,
#     max_epochs:int = 5000,
#     loss="learned_gaussian",
#     path_to_model:str = None,
#     Model_saving_path:str = None,
#     use_std=False):
#     """
#     This function train the neural network.
    
#     PARAMETERS:
#     -----------
#     Model_saving_path : str
#         The saving path of the model.
#     train_loader :
#         The training loader.
#     val_loader : 
#         The validation loader.
#     n_hidden_layers : int
#         The number of hidden layers in the model.
#     Activation_fn : str
#         The activation function of the model.
#     Learning_rate : float
#         The learning rate of the model.
#     Dropout_rate : float
#         The dopout rate of the model.
#     Train_epoch : int
#         The number of epoch of training.
    

#     RETURNS:
#     -------
#     model : 
#         The trained model ready to be used
#     Train_losses : 
#         The values of the training losses over the epoch of training.
#     val_losses : 
#         The values of the validation losses over the epoch of training.
#     """
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     nb_outpout = train_loader.dataset[0][1].shape[-1]
#     nb_input = train_loader.dataset[0][0].shape[-1]
    
#     # Hidden_layers = []

#     # for i in range(n_hidden_layers):
#     #     Hidden_layers.append(nb_nerons)
    
#     model = FCNN(
#         n_input=nb_input,
#         n_output=nb_outpout,
#         # n_hidden=[512, 512, 512],
#         # n_hidden=[1024, 1024, 1024, 1024],
#         #n_hidden=[64, 64, 64],
#         #n_hidden=[128, 128, 128, 128],
#         # n_hidden=[128, 128, 128, 128, 128],
#         n_hidden=n_hidden,
#         # n_hidden=[128, 128], # model 10
#         # n_hidden=[64, 64], # model 12
#         activation_fn=Activation_fn,
#         #activation_fn="ReLU",
#         loss=loss,
#         #learning_rate=0.009332352540651494,
#         # learning_rate=0.0002249937260017888,
#         learning_rate=Learning_rate,
#         #dropout_rate=0.010011267028423554,
#         #dropout_rate=0.027582214112809256,
#         # dropout_rate=0.01, # model 4
#         # dropout_rate=0.02,
#         # dropout_rate=0.0, # model 10
#         dropout_rate=Dropout_rate,
#         weight_decay=weight_decay,
#         device=device,
#         use_std=use_std
#     ).to(device)

#     if path_to_model is not None and os.path.isfile(path_to_model):
#         model.load_model(path_to_model)
#         print(f"Loaded model weights from {path_to_model}")
#         return model, [], []
        
    
#     # optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=2.5e-6)
    
#     # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
    
#     model = torch.compile(model)    

    
#     train_losses, val_losses = model.fit(
#         train_loader, val_loader, min_epochs=min_epochs, max_epochs=max_epochs)
    
#     if Model_saving_path is not None:
#         model.save_model(Model_saving_path)

#     return model, train_losses, val_losses


# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.optim.lr_scheduler import ReduceLROnPlateau
# from tqdm import tqdm


# # ======================================================
# # Define the Feed-Forward Neural Network (FFNN)
# # ======================================================
# class FFNN(nn.Module):
#     def __init__(self, n_input, n_output, n_hidden_layers=2, n_neurons=200, activation_fn="ReLU", dropout_rate=0.0):
#         super(FFNN, self).__init__()

#         # Choose activation function
#         if activation_fn.lower() == "relu":
#             act_fn = nn.ReLU()
#         elif activation_fn.lower() == "silu":
#             act_fn = nn.SiLU()
#         elif activation_fn.lower() == "tanh":
#             act_fn = nn.Tanh()
#         else:
#             raise ValueError(f"Unsupported activation function: {activation_fn}")

#         layers = []
#         in_dim = n_input

#         for _ in range(n_hidden_layers):
#             layers.append(nn.Linear(in_dim, n_neurons))
#             layers.append(act_fn)
#             if dropout_rate > 0:
#                 layers.append(nn.Dropout(dropout_rate))
#             in_dim = n_neurons

#         # Output layer (no activation for regression)
#         layers.append(nn.Linear(in_dim, n_output))

#         self.model = nn.Sequential(*layers)

#     def forward(self, x):
#         return self.model(x)

#     def predict(self, X):
#         """
#         Keras-style predict() wrapper for convenience.
#         Accepts numpy arrays or torch tensors.
#         Returns numpy array.
#         """
#         self.eval()
#         device = next(self.parameters()).device

#         if isinstance(X, np.ndarray):
#             X = torch.tensor(X, dtype=torch.float32, device=device)
#         elif isinstance(X, torch.Tensor):
#             X = X.to(device)
#         else:
#             raise TypeError("Input must be a numpy array or torch tensor.")

#         with torch.no_grad():
#             y_pred = self.forward(X).cpu().numpy()
#         return y_pred


# # ======================================================
# # Training Function
# # ======================================================
# def train_model_FFNN(
# train_loader,
# val_loader,
# n_hidden_layers: int = 3,
# nb_neurons: int = 200,
# Activation_fn: str = "ReLU",
# Learning_rate: float = 0.001,
# Dropout_rate: float = 0.0,
# Train_epoch: int = 100,
# Model_saving_path: str = None
# ):
#     """
#     Train a Feed-Forward Neural Network (FFNN) for regression.

#     PARAMETERS
#     ----------
#     train_loader : DataLoader
#         Training dataset loader
#     val_loader : DataLoader
#         Validation dataset loader
#     n_hidden_layers : int
#         Number of fully connected hidden layers
#     nb_neurons : int
#         Number of neurons per hidden layer
#     Activation_fn : str
#         Activation function name ("ReLU", "SiLU", etc.)
#     Learning_rate : float
#         Learning rate for Adam optimizer
#     Dropout_rate : float
#         Dropout rate (0.0 disables dropout)
#     Train_epoch : int
#         Number of epochs
#     Model_saving_path : str
#         Optional path to save trained model

#     RETURNS
#     -------
#     model : torch.nn.Module
#         The trained model ready for inference
#     train_losses : list
#         Training loss per epoch
#     val_losses : list
#         Validation loss per epoch
#     """
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # Get input/output sizes from the first batch
#     sample_x, sample_y = next(iter(train_loader))
#     n_input = sample_x.shape[1]
#     n_output = sample_y.shape[1] if sample_y.ndim > 1 else 1

#     # Initialize model
#     model = FFNN(
#         n_input=n_input,
#         n_output=n_output,
#         n_hidden_layers=n_hidden_layers,
#         n_neurons=nb_neurons,
#         activation_fn=Activation_fn,
#         dropout_rate=Dropout_rate
#     ).to(device)

#     # Optimizer and loss
#     optimizer = optim.Adam(model.parameters(), lr=Learning_rate)
#     criterion = nn.MSELoss()
#     scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

#     train_losses = []
#     val_losses = []

#     # ======================================================
#     # Training Loop
#     # ======================================================
#     for epoch in range(Train_epoch):
#         model.train()
#         running_loss = 0.0
#         for X_batch, y_batch in train_loader:
#             X_batch, y_batch = X_batch.to(device), y_batch.to(device)

#             optimizer.zero_grad()
#             outputs = model(X_batch)
#             loss = criterion(outputs, y_batch)
#             loss.backward()
#             optimizer.step()
#             running_loss += loss.item() * X_batch.size(0)

#         epoch_train_loss = running_loss / len(train_loader.dataset)
#         train_losses.append(epoch_train_loss)

#         # ======================================================
#         # Validation
#         # ======================================================
#         model.eval()
#         val_loss = 0.0
#         with torch.no_grad():
#             for X_val, y_val in val_loader:
#                 X_val, y_val = X_val.to(device), y_val.to(device)
#                 preds = model(X_val)
#                 vloss = criterion(preds, y_val)
#                 val_loss += vloss.item() * X_val.size(0)

#         epoch_val_loss = val_loss / len(val_loader.dataset)
#         val_losses.append(epoch_val_loss)

#         scheduler.step(epoch_val_loss)

#         tqdm.write(f"Epoch [{epoch+1}/{Train_epoch}] "
#                    f"Train Loss: {epoch_train_loss:.6f} | "
#                    f"Val Loss: {epoch_val_loss:.6f}")

#     # ======================================================
#     # Save model if path provided
#     # ======================================================
#     if Model_saving_path is not None:
#         torch.save(model.state_dict(), Model_saving_path)
#         print(f"✅ Model saved at: {Model_saving_path}")

#     return model, train_losses, val_losses


# def Uncertanities_computation(model, train_Dataset):
#     """
#     This function compute the Emulator's predictiv uncertanities.

#     PARAMETERS:
#     ----------
#     saving_path : str
#         The saving path of the model uncertanities.
#     covariance_data_path : str
#         The path of the directory containing the dataset that will be used for comute the uncertanity of the emulator.
#     normalization_cst_cosmo_param_path : str
#         The normalization constants for the inputs parameters.
#     normalization_cst_data_vector_path : str
#         The normalization constants for the output data vector.
#     model_path : str
#         The path of the model.
#     nb_input : int
#         The number of input parameters. Initally set to 19, 13 cosmological parameters and 6 HOD parameters.
#     nb_outpout : int
#         The number of output point. Initially set to 73 but this is for the observables in logarithmic bin only, for linear bin nb_output is 325.
#     n_hidden_layers : int
#         The number of hidden layers in the model.
#     nb_nerons : int
#         The number of neurons per hidden layers.
#     Activation_fn : str
#         The activation function of the model.
#     Learning_rate : float
#         The learning rate of the model.
#     Dropout_rate : float
#         The dopout rate of the model.
        

#     RETURNS:
#     -------
#     emulator_cov_matrix : np.2Darray
#         The covariance matrix of the emulator uncertanities
#     """

#     # Loading the covariance dataset
#     q05, q95 = np.quantile(train_Dataset.X_training, q=[0.05,0.95], axis=0)
#     mask_test_prior = ((train_Dataset.X_test > q05) & (train_Dataset.X_test < q95)).all(axis=1)
#     X_test, Y_test = train_Dataset.X_test[mask_test_prior], train_Dataset.y_test[mask_test_prior]


#     # Do the predictions for all of the parameters set in the covariance dataset
#     # Getting rp and s
#     rp = np.array([ 0.04641589,  0.06812921,  0.1       ,  0.14677993,  0.21544347,
#         0.31622777,  0.46415888,  0.68129207,  1.        ,  1.46779927,
#         2.15443469,  3.16227766,  4.64158883,  6.81292069, 10.        ,
#     14.67799268, 21.5443469 , 31.6227766 ])

#     rp = (rp[:-1] + rp[1:]) / 2
#     s = np.array([  0.21544347,  0.26101572,  0.31622777,  0.38311868,  0.46415888,
#             0.56234133,  0.68129207,  0.82540419,  1.        ,  1.21152766,
#             1.46779927,  1.77827941,  2.15443469,  2.61015722,  3.16227766,
#             3.83118685,  4.64158883,  5.62341325,  6.81292069,  8.25404185,
#             10.        , 12.11527659, 14.67799268, 17.7827941 , 21.5443469 ,
#             26.10157216, 31.6227766 ])
        
#     X_test_norm = torch.tensor(train_Dataset.x_test_norm[mask_test_prior], dtype=torch.float32)
#     y_pred_norm  = model.predict(X_test_norm, no_grad=True)[0].squeeze().cpu().numpy()
#     Y_pred = train_Dataset.denormalise_y(y_pred_norm)


    
#     deltas = Y_pred - Y_test      
#     # deltas = y_pred_norm-train_Dataset.y_test_norm
#     delta_mean = np.mean(deltas, axis=0) 
    
#     cov = np.zeros((deltas.shape[1], deltas.shape[1]))  
    
#     for i in range(len(deltas)):
#         delta = deltas[i] - delta_mean              
#         cov += np.outer(delta, delta)                   
    
#     cov_emu = cov / (len(deltas) - 1)
#     cov_emu = np.cov(deltas, rowvar=False,  ddof=0) / (len(deltas) - 1)
    
#     return cov_emu
    
    
# param_labels = {
#     "Ac": r"$A_c$",
#     "As": r"$A_s$",
#     "M_0": r"$M_0$",
#     "M_1": r"$M_1$",
#     "Q": r"$Q$",
#     "alpha": r"$\alpha$",
#     # --- assembly bias parameters ---
#     "ab_c_cen": r"$A_{B,\,c}^{\mathrm{cen}}$",
#     "ab_c_sat": r"$A_{B,\,c}^{\mathrm{sat}}$",
#     "ab_env_cen": r"$A_{B,\,\mathrm{env}}^{\mathrm{cen}}$",
#     "ab_env_sat": r"$A_{B,\,\mathrm{env}}^{\mathrm{sat}}$",
#     # --- other model parameters ---
#     "f_sigv": r"$f_{\sigma_v}$",
#     "gamma": r"$\gamma$",
#     "log_Mcent": r"$\log M_{\mathrm{cent}}$",
#     "pmax": r"$p_{\max}$",
#     "sigma_M": r"$\sigma_M$",
#     "exp_frac": r"$f_{\exp}$",
#     "exp_scale": r"$s_{\exp}$",
#     "nfw_rescale": r"$\lambda_{\mathrm{NFW}}$",
#     "v_infall": r"$v_{\mathrm{infall}}$",
#     "v_smear": r"$v_{\mathrm{smear}}$"
# }



# def plot_verif(train_Dataset, model, nb_plots=5, stats=['wp', 'xi'], add_emu_err=False):

#     X_test, Y_test = train_Dataset.X_test, train_Dataset.y_test
    
#     X_test_norm = torch.tensor(train_Dataset.x_test_norm, dtype=torch.float32)
#     y_pred_norm, varypred = model.predict(X_test_norm, no_grad=True)
#     Y_pred, varypred = train_Dataset.denormalise_y(y_pred_norm.squeeze().cpu().numpy(), varypred.squeeze().cpu().numpy())
#     # varypred = varypred.squeeze().cpu().numpy() * (train_Dataset.max_y - train_Dataset.min_y)**2
#     cov_emu = Uncertanities_computation(model, train_Dataset)
#     std_emu = np.sqrt(cov_emu.diagonal())
#     residuals = (Y_test - Y_pred)/std_emu

#     rp = np.array([ 0.04641589,  0.06812921,  0.1       ,  0.14677993,  0.21544347,
#         0.31622777,  0.46415888,  0.68129207,  1.        ,  1.46779927,
#         2.15443469,  3.16227766,  4.64158883,  6.81292069, 10.        ,
#     14.67799268, 21.5443469 , 31.6227766 ])

#     rp = (rp[:-1] + rp[1:]) / 2
#     s = np.array([  0.21544347,  0.26101572,  0.31622777,  0.38311868,  0.46415888,
#             0.56234133,  0.68129207,  0.82540419,  1.        ,  1.21152766,
#             1.46779927,  1.77827941,  2.15443469,  2.61015722,  3.16227766,
#             3.83118685,  4.64158883,  5.62341325,  6.81292069,  8.25404185,
#             10.        , 12.11527659, 14.67799268, 17.7827941 , 21.5443469 ,
#             26.10157216, 31.6227766 ])
    
#     s = (s[:-1] + s[1:]) / 2
#     x_vals = [rp, s, s]
#     indx_rp = len(rp) 
#     indx_s0 = len(s)
    
#     y_true = []
#     y_pred = []
#     std_emulator = []
#     err_residuals = []
#     err_preds = []
#     if 'wp' in stats:
#         y_true += [Y_test[:,:indx_rp]]
#         y_pred += [Y_pred[:,:indx_rp]]
#         std_emulator += [std_emu[:indx_rp]]
#         err_residuals += [residuals[:,:indx_rp]] 
#         err_preds += [np.sqrt(varypred)[:,:indx_rp]]  
#     else:
#         indx_rp = 0
#         y_true += [[]]
#         y_pred += [[]]
#         std_emulator += [[]]
#         err_residuals += [[]]
#         err_preds += [[]]
#     if 'xi' in stats:
#         y_true += [Y_test[:,indx_rp:-indx_s0], Y_test[:,-indx_s0:]]
#         y_pred += [Y_pred[:,indx_rp:-indx_s0], Y_pred[:,-indx_s0:]]        
#         std_emulator += [std_emu[indx_rp:-indx_s0], std_emu[-indx_s0:]]
#         err_residuals += [residuals[:,indx_rp:-indx_s0], residuals[:,-indx_s0:]]
#         y_pred += [Y_pred[:,indx_rp:-indx_s0], Y_pred[:,-indx_s0:]]        
#         err_preds += [np.sqrt(varypred)[:,indx_rp:-indx_s0], np.sqrt(varypred)[:,-indx_s0:]]
    
#     curve_names = [r"$w_p$", r"$\xi_0$", r"$\xi_2$"]
#     subplot_labels = [r"$\delta w_p / \sigma_{w_p}$", r"$\delta \xi_0 / \sigma_{\xi_0}$", r"$\delta \xi_2 / \sigma_{\xi_2}$"]


#     for i in np.random.choice(np.arange(Y_pred.shape[0]), nb_plots, replace=False):
#         fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(28, 6), sharex='col', gridspec_kw={'height_ratios': [1, 0.2]})
#         fig.subplots_adjust(hspace=0.1, wspace=0.3)
#         for col in range(3):
#             if ('xi' not in stats) & (col >0):
#                 continue
#             if ('wp' not in stats) & (col==0):
#                 continue

#             x = x_vals[col]
#             y = (x * y_true[col][i])
#             y_predict = (x * y_pred[col][i])
#             std_emul = (x * std_emulator[col])
#             err_resid = err_residuals[col][i]
#             err_pred = (x * err_preds[col][i])
#             # err_resid = [1]*len(y_predict)
            
#             sub_label = subplot_labels[col]
    
#             axs[0, col].plot(x, y, label='Real', linewidth=2)
#             axs[0, col].plot(x, y_predict, label='Prediction', linestyle="--", color='orange')
#             axs[0, col].fill_between(x, (y_predict - std_emul), (y_predict + std_emul), alpha=0.3, label='prediction error', color='green')    
#             if add_emu_err:        
#                 axs[0, col].fill_between(x, (y_predict - err_pred), (y_predict + err_pred), alpha=0.3, label='emu pred error', color='red')
            
#             xlabel = r"$r_p$" if col == 0 else r"$s$"
#             axs[0, col].set_ylabel(fr"{curve_names[col]} $\cdot$ {xlabel}", fontsize=15)
#             axs[0, col].set_title(fr"{curve_names[col]}", fontsize=17)
#             axs[0, col].legend(fontsize=15)
#             axs[0, col].grid(True)
#             axs[0, col].tick_params(axis='both', labelsize=15)
#             axs[0, col].set_xscale("log")
    
#             axs[1, col].plot(x, err_resid, color='black')
#             # axs[1, col].axhline(0, color='grey', linestyle='--')
#             axs[1, col].set_xlabel(xlabel, fontsize=17)
#             axs[1, col].set_ylim([-5, 5])
#             axs[1, col].set_ylabel(sub_label, fontsize=17)
#             axs[1, col].grid(True)
#             axs[1, col].tick_params(axis='both', labelsize=15)
#             axs[1, col].set_xscale("log")

#             name_params = ', '.join(([f'{param_labels[par[:-4]]} = {val:.2f}' for par, val in zip(train_Dataset.name_arr, X_test[i])]))
#             fig.suptitle(name_params, fontsize=16)
        
#         # fig.suptitle(fr"Mock nb.{indicies}, HOD: {X}, chi² = {chi_square}", fontsize=16)
#         # fig.suptitle(fr"Mock nb.{indicies}, chi² = {chi_square}", fontsize=16)
#         # fig.suptitle(fr"Prediction made for mock nb.{indicies}, with a chi² = {chi_square}", fontsize=16)
#         plt.show()

        
# def Z_score_and_chi_square_calculation(model, train_Dataset, use_var_pred=False, use_red_prior=True):
#     """
#     This function is used to do a Z-score test of the Emulator and the chi square distribution of the pedictions.

#     PARAMETERS:
#     -----------
#     Testing_dataset_path : str
#         The path to the testind dataset folder.
#     model_path : str
#         The path of the model.
#     normalization_cst_cosmo_param_path : str
#         The normalization constants for the inputs parameters.
#     normalization_cst_data_vector_path : str
#         The normalization constants for the output data vector.
#     emulator_covariance_path : str
#         The covariance matrix of the emulator uncertanities.
#     nb_input : int
#         The number of input parameters. Initally set to 19, 13 cosmological parameters and 6 HOD parameters.
#     nb_outpout : int
#         The number of output point. Initially set to 73 but this is for the observables in logarithmic bin only, for linear bin nb_output is 325.
#     n_hidden_layers : int
#         The number of hidden layers in the model.
#     nb_nerons : int
#         The number of neurons per hidden layers.
#     Activation_fn : str
#         The activation function of the model.
#     Learning_rate : float
#         The learning rate of the model.
#     Dropout_rate : float
#         The dopout rate of the model.

#     RETURN:
#     ------
#     Z_score : np.array
#         Z-score dsitribution of the predictions.
#     Chi_square : np.array
#         Chi square distribution of the predictions.
#     """


#     cov_emu = Uncertanities_computation(model, train_Dataset)
#     std_emu = np.sqrt(np.diag(cov_emu))


#     # Loading of the verification dataset
#     # X_test, Y_test = train_Dataset.X_test, train_Dataset.y_test


#     # Do the predictions for all of the parameters set in the covariance dataset
#     # Getting rp and s
#     rp = np.array([ 0.04641589,  0.06812921,  0.1       ,  0.14677993,  0.21544347,
#         0.31622777,  0.46415888,  0.68129207,  1.        ,  1.46779927,
#         2.15443469,  3.16227766,  4.64158883,  6.81292069, 10.        ,
#     14.67799268, 21.5443469 , 31.6227766 ])

#     rp = (rp[:-1] + rp[1:]) / 2
#     s = np.array([  0.21544347,  0.26101572,  0.31622777,  0.38311868,  0.46415888,
#             0.56234133,  0.68129207,  0.82540419,  1.        ,  1.21152766,
#             1.46779927,  1.77827941,  2.15443469,  2.61015722,  3.16227766,
#             3.83118685,  4.64158883,  5.62341325,  6.81292069,  8.25404185,
#             10.        , 12.11527659, 14.67799268, 17.7827941 , 21.5443469 ,
#             26.10157216, 31.6227766 ])

#     q05, q95 = np.quantile(train_Dataset.X_training, q=[0.05,0.95], axis=0)
#     mask_test_prior = ((train_Dataset.X_test > q05) & (train_Dataset.X_test < q95)).all(axis=1)
#     if not use_red_prior:
#         mask_test_prior = np.ones_like(train_Dataset.x_test_norm[:,0], dtype=bool)
    
#     Y_test = train_Dataset.y_test[mask_test_prior]
    
#     # X_test_norm = torch.tensor(train_Dataset.x_test_norm[mask_test_prior], dtype=torch.float32)
#     # y_pred_norm = model.predict(X_test_norm, no_grad=True)[0].squeeze().cpu().numpy()
#     # Y_pred = train_Dataset.denormalise_y(y_pred_norm)
#     # Y_test = train_Dataset.y_test[mask_test_prior]

#     X_test_norm = torch.tensor(train_Dataset.x_test_norm[mask_test_prior], dtype=torch.float32)
#     y_pred_norm, var_pred_norm = model.predict(X_test_norm, no_grad=True)
#     y_pred_norm = y_pred_norm.squeeze().cpu().numpy()
#     var_pred_norm = var_pred_norm.squeeze().cpu().numpy()
#     Y_pred, var_pred = train_Dataset.denormalise_y(y_pred_norm, var_pred_norm)
#     Y_test = train_Dataset.y_test[mask_test_prior]

#     if use_var_pred:
#         std_emu = np.sqrt(var_pred)
#     Z_score = (Y_pred - Y_test) / std_emu    

#     # Z_score = (y_pred_norm - train_Dataset.y_test_norm[mask_test_prior]) / std_emu    

#     # Calculation of the chi²
#     chi_square = (Y_pred - Y_test) ** 2 / std_emu ** 2

#     chi_square = chi_square / (Y_test.shape[1] - X_test_norm.shape[1])
#     Chi_square = np.sum(chi_square, axis=1)
    
#     return Z_score, Chi_square




from matplotlib import pyplot as plt
import numpy as np
import os
import torch
from torch import nn
from typing import OrderedDict
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, random_split, DataLoader, TensorDataset

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA']   = '1'   # optional, for device-side asserts
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = "cpu"


class Normalizer:
    def __init__(self, X, y, log_transform=False):
        self.log_transform = log_transform

        # Optionally transform the output space
        if log_transform:
            y = np.arcsinh(y)

        # Compute statistics
        self.mean_X = np.mean(X.tolist(), axis=0)
        self.std_X  = np.std(X .tolist(), axis=0)

        self.mean_y = y.mean(axis=0)
        self.std_y  = y.std(axis=0)

    def normalize_x(self, X):
        return (X - self.mean_X) / self.std_X

    def normalize_y(self, y):
        if self.log_transform:
            y = np.arcsinh(y)
        return np.nan_to_num((y - self.mean_y) / self.std_y)

    def denormalize_y(self, y_norm, var_norm=None):
        y = y_norm * self.std_y + self.mean_y

        if var_norm is not None:
            var = var_norm * (self.std_y ** 2)
        else:
            var = None

        if self.log_transform:
            y = np.sinh(y)

        return y, var

    def denormalize_x(self, x_pred_norm):
        return x_pred_norm * self.std_X + self.mean_X


class Training_DatasetManager(Dataset):

    def __init__(self, dir_path:str, path_to_test_files=None, stats = ['wp', 'xi'], log_transform=False, seed=None):
        self.dir_path = dir_path
        self.files = [os.path.join(self.dir_path,f) for f in os.listdir(self.dir_path) if f.endswith(".npy")] # Get the name of all the files in the directory
        self.files.sort() # Sort the files to have a reproducible order
        self.data = [] # This will be used for the storage of the data in memory for the training
        self.seed = seed if seed is not None else np.random.randint(0, 2**32 - 1)
        self.generator = torch.Generator().manual_seed(self.seed)
        self.sep = {} # Value of x (rp and s)
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(DEVICE)

        self.stats = stats
        if path_to_test_files is not None:
            print(f"Loading test files from {path_to_test_files}")
            self.path_to_test_files = [os.path.join(path_to_test_files,f) for f in os.listdir(path_to_test_files) if f.endswith(".npy")] # Get the name of all the files in the directory
            self.path_to_test_files.sort() # Sort the files to have a reproducible order
        
        self.idx = []
        self.log_transform = log_transform

        self.X_training, self.y_training, self.data_dict = self.load_data()

        if self.log_transform :
            self.min_y_train_value = self.y_training.min()
            self.y_training = np.log10(self.y_training - self.min_y_train_value)
            

        # Normalisation min-max
        # self.normalise_x()
        # self.normalise_y()
        self.normalizer = Normalizer(self.X_training, self.y_training, log_transform=self.log_transform)
        self.x_norm = self.normalizer.normalize_x(self.X_training)
        self.y_norm = self.normalizer.normalize_y(self.y_training)
        
        self.data_train = [(self.x_norm[ii], self.y_norm[ii]) for ii in range(self.y_norm.shape[0])]
        if path_to_test_files is not None:
            self.X_test, self.y_test, self.data_dict_test = self.load_data(files=self.path_to_test_files)
            self.x_test_norm = self.normalizer.normalize_x(self.X_test)
            self.y_test_norm = self.normalizer.normalize_y(self.y_test)
        else:
            self.X_test = None
            self.y_test = None
            self.x_test_norm = None


    def load_data(self, files=None):
        """
        Load and merge HOD training data from multiple .npy files.

        Parameters
        ----------
        files : list of str
            List of file paths to .npy files containing HOD training data.

        Returns
        -------
        merged : dict
            Dictionary containing merged HOD parameters and statistics.
            Structure:
                {
                    'hod_fit_param': np.ndarray of shape (n_samples, n_params),
                    'wp': {tracer: [coord_array, xi_array]},
                    'xi_rppi': {tracer: [coord_array1, coord_array2, xi_array]},
                    'xi_smu': {tracer: [coord_array1, coord_array2, xi_array]},
                    'xi_ells': {tracer: [coord_array, xi_array]}
                }
        """

        # ── Load ──────────────────────────────────────────────────────────────────
        raw    = {stat: {} for stat in self.stats}
        params = []

        files = files or self.files
        self.name_arr = None
        for f in files:
            d = np.load(f, allow_pickle=True).item()
            # d.pop('LRG')
            # d.pop('comb_trs')
            # d.pop('param_file')
            missing = [s for s in self.stats if s not in d]
            if missing:
                raise KeyError(f'{f} is missing requested statistics: {missing}')

            params.append(d['hod_fit_param'].tolist())
            if self.name_arr is None:
                self.name_arr = list(d['hod_fit_param'].dtype.names)

            for stat in self.stats:
                for tracer, arrays in d[stat].items():
                    if tracer not in raw[stat]:
                        raw[stat][tracer] = [[] for _ in arrays]
                    for k, arr in enumerate(arrays):
                        raw[stat][tracer][k].append(np.asarray(arr))

        params = np.stack(params, axis=0)
        n      = params.shape[0]

        # ── Stack: last entry is the data, everything before it is a coordinate ───
        merged = {'hod_fit_param': params}

        for stat in self.stats:
            merged[stat] = {}
            for tracer, entries in raw[stat].items():
                merged[stat][tracer] = [stack[0] for stack in entries[:-1]]      # coords
                merged[stat][tracer].append(np.stack(entries[-1], axis=0))       # xi

        # ── Data vector ───────────────────────────────────────────────────────────
        blocks, labels = [], []

        for stat in self.stats:
            for tracer in sorted(merged[stat]):
                flat = merged[stat][tracer][-1].reshape(n, -1)
                blocks.append(flat)
                labels.append((stat, tracer, flat.shape[1]))

        data_vector = np.concatenate(blocks, axis=1)

        edges  = np.cumsum([0] + [lab[2] for lab in labels])
        self.slices = {f'{s}_{t}': slice(edges[j], edges[j+1])
                for j, (s, t, _) in enumerate(labels)}

        # ── keep the coordinate arrays so plotting can split blocks ───
        self.coords = {stat: {tr: merged[stat][tr][:-1]
                              for tr in merged[stat]}
                       for stat in self.stats}
        self.block_shapes = {f'{stat}_{tr}': merged[stat][tr][-1].shape[1:]
                             for stat in self.stats for tr in merged[stat]}

        return merged['hod_fit_param'], data_vector, merged
    
    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.X_training)
    
    def __getitem__(self, idx):
        """Return a sample from the dataset."""
        values, y = self.x_norm[idx], self.y_norm[idx]
    
        return torch.tensor(values, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

    def extract_data(self):
        return torch.tensor(self.x_norm, dtype=torch.float32).to(self.device), torch.tensor(self.y_norm, dtype=torch.float32).to(self.device)

    def get_train_val_sets(self, train_frac=0.8):
        """
        Return the training and validation dataset for the training of the neural net.
        
        PARAMETERS:
        -----------
        train_frac : float 
            Fraction of the dataset to use for training.
    
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
    
    def plot_training_data_distribution(self, show_test=False):
        """
        Plot the distribution of the training data using pair plots.
        This function uses seaborn's pairplot to visualize the relationships between the features in the training dataset. It creates a grid of scatter plots for each pair of features, along with histograms for the individual features on the diagonal. This is useful for understanding the distribution and correlations of the training data.
        """

        import matplotlib.pyplot as plt
        import seaborn as sns
        import pandas as pd

        df = pd.DataFrame(self.X_training, columns=self.name_arr)
        df['set'] = 'train'

        if show_test and self.X_test is not None:
            df_test = pd.DataFrame(self.X_test, columns=self.name_arr)
            df_test['set'] = 'test'
            df = pd.concat([df, df_test], ignore_index=True)

        # sns.pairplot(df, hue='set',
        #      corner=True,                          # drop the redundant upper triangle
        #      diag_kind='hist',                     # 'kde' if the sets overlap heavily
        #      palette={'train': 'C0', 'test': 'C3'},
        #      plot_kws={'s': 12, 'alpha': 0.5, 'edgecolor': 'none'})
        g = sns.PairGrid(df, hue='set', corner=True,
                 palette={'train': 'lightgray', 'test': 'C3'})
        g.map_lower(sns.scatterplot, s=10, alpha=0.4, edgecolor='none')
        g.map_diag(sns.kdeplot, fill=False)
        g.add_legend()
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
        self.device = torch.device(DEVICE)
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
        scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=10)
        # self.scheduler = scheduler

        self.train_losses = []
        self.val_losses   = []

        best_val_loss = float("inf")
        best_state = None
        epochs_no_improve = 0  # for plateau tracking

        start_time = time.time()

        for epoch in range(max_epochs):

            train_loss = self.train_epoch(train_loader)
            val_loss = self.evaluate(val_loader)

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

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

    def plot_loss(self):
        """
        Plot the training and validation loss curves.
        """

        if getattr(self, 'train_losses', None) is None or getattr(self, 'val_losses', None) is None:
            raise ValueError("Training and validation losses are not available from a preloaded model.")
        
        fig,ax = plt.subplots(1,1,figsize=(6, 4))
        ax.plot(self.train_losses, label="Training loss")
        ax.plot(self.val_losses, label="Validation loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")

        ax.set_title(f"Training and Validation Loss \n Loss value at last epoch train {self.train_losses[-1]:.2f}, validation {self.val_losses[-1]:.2f}")
        ax.legend()
        ax.grid(True)
        fig.tight_layout()
        fig.show()
        

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
    path_to_model:str = None,
    Model_saving_path:str = None,
    use_std=False):
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
    device = torch.device(DEVICE)
    # device = 'cpu'
    nb_outpout = train_loader.dataset[0][1].shape[-1]
    nb_input = train_loader.dataset[0][0].shape[-1]
    
    # Hidden_layers = []

    # for i in range(n_hidden_layers):
    #     Hidden_layers.append(nb_nerons)
    
    model = FCNN(
        n_input=nb_input,
        n_output=nb_outpout,
        n_hidden=n_hidden,
        activation_fn=Activation_fn,
        loss=loss,
        learning_rate=Learning_rate,
        dropout_rate=Dropout_rate,
        weight_decay=weight_decay,
        device=device,
        use_std=use_std
    ).to(device)

    if path_to_model is not None and os.path.isfile(path_to_model):
        model.load_model(path_to_model)
        print(f"Loaded model weights from {path_to_model}")
        return model
        
    
    # optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=2.5e-6)
    
    # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
    
    model = torch.compile(model)    

    
    model.fit(
        train_loader, val_loader, min_epochs=min_epochs, max_epochs=max_epochs)
    
    if Model_saving_path is not None:
        model.save_model(Model_saving_path)

    return model



def Uncertanities_computation(model, train_Dataset):
    """
    Compute the covariance matrix of the prediction errors on the test set.

    Parameters
    ----------
    model : FCNN
        The trained neural network model.
    train_Dataset : Training_DatasetManager
        The dataset manager containing the test data.

    Returns
    -------
    cov : np.ndarray
        Covariance matrix of the prediction errors.
    std_test : np.ndarray
        Standard deviation of the prediction errors.
    """
    Y_test = train_Dataset.y_test
    Y_pred = model.predict(torch.tensor(train_Dataset.x_test_norm, dtype=torch.float32).to(model.device), no_grad=True)[0]
    Y_pred = train_Dataset.normalizer.denormalize_y(Y_pred)[0].squeeze().cpu().numpy()
    deltas = Y_pred - Y_test      
    delta_mean = np.mean(deltas, axis=0) 


    deltas = Y_pred - Y_test      
    # deltas = y_pred_norm-train_Dataset.y_test_norm
    delta_mean = np.mean(deltas, axis=0) 
    deltas -= delta_mean 
    cov = np.cov(deltas, rowvar=False, ddof=1) # Use ddof=1 for sample covariance

    return cov