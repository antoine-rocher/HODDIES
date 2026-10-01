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
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class Normalizer():
    def __init__(self, train_dataset):

        self.log_transform = train_dataset.log_transform
        self.log_mask = self._build_log_mask(train_dataset)

        self.mean_X = np.nanmean(train_dataset.X_training.tolist(), axis=0)
        self.std_X  = np.nanstd(train_dataset.X_training.tolist(), axis=0)

        # Transform data
        self.y_transform = np.asarray(train_dataset.y_training, dtype=float).copy()      
        if self.log_mask.any():
            self.y_transform[..., self.log_mask] = np.arcsinh(self.y_transform[..., self.log_mask])

        self.mean_y = np.nanmean(self.y_transform, axis=0)
        self.std_y  = np.nanstd(self.y_transform, axis=0)
        
        
    def _build_log_mask(self, train_dataset):
        """
        Return a boolean array (n_output,) marking which bins get arcsinh.
        log_transform can be:
        - False / None      -> nothing transformed
        - True              -> everything transformed
        - dict {stat: bool} -> per-statistic, using self.slices[stat]
        - list/set of stats -> those stats transformed, rest not
        """
        mask = np.zeros(train_dataset.n_output, dtype=bool)
        if not self.log_transform:
            return mask
        if self.log_transform is True:
            mask[:] = True
            return mask
        if isinstance(self.log_transform, dict):
            for st, flag in self.log_transform.items():
                if flag:
                    print('Applying arcsinh transform to', st)
                    for kk in train_dataset.slices.keys():
                        if st in kk:
                            mask[train_dataset.slices[kk]] = True
        else:  # iterable of stat names
            for st in self.log_transform:
                print('Applying arcsinh transform to', st)
                for kk in train_dataset.slices.keys():
                    if st in kk:
                        mask[train_dataset.slices[kk]] = True
        return mask

    def normalize_x(self, X):
        """
        Normalize the input features X using mean and std from training data.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input features to normalize.
        
        Returns
        -------
        X_norm : array-like, shape (n_samples, n_features)
            Normalized input features.
        """

        return (X - self.mean_X) / self.std_X


    def normalize_y(self, y):
        """
        Normalize the output values y using mean and std from training data.
        Apply arcsinh transformation to specified bins if log_mask is set.

        Parameters
        ----------
        y : array-like, shape (n_samples, n_output)
            Output values to normalize.

        Returns
        -------
        y_norm : array-like, shape (n_samples, n_output)
            Normalized output values.
        """

        y_norm = np.asarray(y, dtype=float).copy()
        if self.log_mask.any():
            y_norm[..., self.log_mask] = np.arcsinh(y_norm[..., self.log_mask])
        return np.nan_to_num((y_norm - self.mean_y) / self.std_y)


    def denormalize_y(self, y_norm, var_norm=None, method="sample", n_samp=1000, seed=None):
        """
        Undo the y-normalization (standardization, then per-bin arcsinh -> sinh
        controlled by self.log_mask), propagating the variance if given.

        Parameters
        ----------
        y_norm   : (N, B) or (B,) normalized predictions (mean).
        var_norm : normalized VARIANCE, same shape as y_norm, or None.
        method   : variance propagation through the sinh nonlinearity (log bins only):
                "delta"  -> first-order Jacobian, var_phys = cosh(u)^2 * var (fast, small-sigma).
                "sample" -> Monte-Carlo draws through sinh (exact, handles skew; default).
                Linear bins are exact either way.
        n_samp   : number of Monte-Carlo draws when method="sample".

        Returns
        -------
        y   : physical-space mean, same shape as y_norm.
        var : physical-space VARIANCE, same shape, or None if var_norm is None.
        """
        y_norm = np.asarray(y_norm, dtype=float)

        # ---- undo standardization (all bins) ----
        u = y_norm * self.std_y + self.mean_y

        # ---- mean: sinh on log bins, identity on linear bins ----
        y = u.copy()
        y[..., self.log_mask] = np.sinh(u[..., self.log_mask])

        if var_norm is None:
            return y, None

        var_norm = np.asarray(var_norm, dtype=float)
        var = var_norm * self.std_y**2                      # exact on linear bins

        if not self.log_mask.any():
            return y, var

        # ---- start from the linear result, then fix up the log bins ----

        if method == "delta":
            # first-order Jacobian only on log bins: (d sinh/du)^2 = cosh(u)^2
            var = var.copy()
            var[..., self.log_mask] = np.cosh(u[..., self.log_mask])**2 * var[..., self.log_mask]
            return y, var

        elif method == "sample":
            # exact propagation on the LOG bins only; linear bins keep var_norm * std_y^2
            rng = np.random.default_rng(seed)
            sigma_norm = np.sqrt(var_norm)                  # std, not variance

            # draw only for the log-transformed columns
            y_norm_log   = y_norm[..., self.log_mask]                   # (..., n_log)
            sigma_log    = sigma_norm[..., self.log_mask]
            std_log, mean_log = self.std_y[self.log_mask], self.mean_y[self.log_mask]

            # add a sample axis: shape (..., n_samp, n_log)
            draws = (y_norm_log[..., None, :]
                    + sigma_log[..., None, :]
                    * rng.standard_normal(y_norm_log.shape[:-1] + (n_samp, y_norm_log.shape[-1])))
            u_draws = draws * std_log + mean_log
            draws_phys = np.sinh(u_draws)                   # (..., n_samp, n_log)

            var = var.copy()
            var[..., self.log_mask] = draws_phys.var(axis=-2)           # variance over the sample axis
            return y, var

        else:
            raise ValueError(f"unknown method {method!r}; use 'delta' or 'sample'")

    def denormalize_x(self, x_pred_norm):
        """
        Undo the x-normalization (standardization) to return to physical space.

        Parameters
        ----------
        x_pred_norm : array-like, shape (n_samples, n_input)
            Normalized input features.

        Returns
        -------
        x_pred_phys : array-like, shape (n_samples, n_input)
            Input features in physical space.
        """
        return x_pred_norm * self.std_X + self.mean_X


class Training_DatasetManager(Dataset):

    def __init__(self, dir_path:str, path_to_test_files=None, stats = ['wp'], log_transform=False, seed=None, num_testset=None, device=None):
        """
        Initialize the Training_DatasetManager.

        PARAMETERS:
        -----------
        dir_path : str
            Path to the directory containing training .npy files.
        path_to_test_files : str, optional
            Path to the directory containing test .npy files. If None, no test set is loaded.
        stats : list of str
            List of statistics to load (e.g., ['wp', 'xi_ells']). 
        log_transform : bool or dict
            If True, apply arcsinh transformation to all statistics. If False, no transformation.
            If a dict, specify which statistics to transform (e.g., {'wp': True, 'xi_ells': False}).
        seed : int, optional
            Random seed for reproducibility. If None, a random seed is generated.
        num_testset : int, optional
            Number of test files to load. If None, all available test files are loaded.
        """

        self.dir_path = dir_path
        self.files = [os.path.join(self.dir_path,f) for f in os.listdir(self.dir_path) if f.endswith(".npy")] # Get the name of all the files in the directory
        self.files.sort() # Sort the files to have a reproducible order
        self.data = [] # This will be used for the storage of the data in memory for the training
        self.seed = seed if seed is not None else np.random.randint(0, 2**32 - 1)
        self.generator = torch.Generator().manual_seed(self.seed)
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(DEVICE if device is None else device)

        self.stats = stats
        from HODDIES.estimators import get_list_stat
        missing_stats = [s for s in stats if s not in get_list_stat()]
        if missing_stats:
            raise ValueError(f"Requested statistics {missing_stats} not available. Available statistics: {get_list_stat()}")
        if path_to_test_files is not None:
            self.path_to_test_files = [os.path.join(path_to_test_files,f) for f in os.listdir(path_to_test_files) if f.endswith(".npy")]
            print(f"Loading test files from {path_to_test_files}")
            if num_testset is not None:
                if num_testset > len(self.path_to_test_files):
                    print(f"Requested number of test files ({num_testset}) exceeds available files ({len(self.path_to_test_files)}). Continue with the maximum available files ({len(self.path_to_test_files)}).")
                else:
                    self.path_to_test_files = np.random.choice(self.path_to_test_files, size=num_testset, replace=False)
            
            self.path_to_test_files.sort() # Sort the files to have a reproducible order
        
        self.idx = []
        self.log_transform = log_transform

        print(f"Loading training files from {self.dir_path}")
        self.X_training, self.y_training, self.data_dict = self.load_data()
        self.n_input = self.X_training.shape[1]
        self.n_output = self.y_training.shape[1]
        self.len_trainning = self.X_training.shape[0]

        # Normalisation min-max

        self.normalizer = Normalizer(self)
        self.x_norm = self.normalizer.normalize_x(self.X_training)
        self.y_norm = self.normalizer.normalize_y(self.y_training)
        
        self.data_train = [(self.x_norm[ii], self.y_norm[ii]) for ii in range(self.len_trainning)]
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
                    'CIC': {tracer: [xi_array]},
                    'wp': {tracer: [coord_array, xi_array]},
                    'xi_rppi': {tracer: [coord_array1, coord_array2, xi_array]},
                    'xi_smu': {tracer: [coord_array1, coord_array2, xi_array]},
                    'xi_ells': {tracer: [coord_array, xi_array]}
                }
        """

        # ── Load ──────────────────────────────────────────────────────────────────
        raw    = {stat: {} for stat in self.stats}
        params = []

        files = files if files is not None else self.files
        self.name_arr = None
        for f in files:
            d = np.load(f, allow_pickle=True).item()
            missing = [s for s in self.stats if s not in d]
            if missing:
                raise KeyError(f'{f} is missing requested statistics: {missing}')
            
            params.append(d['hod_fit_param'].tolist())
            if self.name_arr is None:
                self.name_arr = list(d['hod_fit_param'].dtype.names)

            for stat in self.stats:
                for tracer, arrays in d[stat].items():
                    # Coordinate-free statistics may be a bare array, not a list.
                    if not isinstance(arrays, (list, tuple)):
                        arrays = [arrays]
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
        This function uses seaborn's pairplot to visualize the relationships between the features in the training dataset. 
        It creates a grid of scatter plots for each pair of features, along with histograms for the individual features on the diagonal. 
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

    def plot_training_stats(self, stats=None, data=None, data_err=None,
            max_cols=4, fontsize=11, block_hspace=0.55,
            wspace=0.30, colors=None, show=True):
        
        """Compare emulator predictions with the test set.

        One figure per test sample, one row of panels per tracer, wrapped at
        ``max_cols`` columns, with a ``(truth - prediction)/sigma`` sub-panel
        flush beneath each panel.

        Parameters
        ----------
        self : Training_DatasetManager
            Must expose ``slices``; ``coords`` (see note in the module
            docstring) is needed to split multipole blocks into panels.
        data : array-like
            The data to plot.
        model : object with ``predict``
        nb_plots : int
            Number of random test samples to show, ignored when ``indices``
            is given.
        stats : sequence of str or None
            Restrict to these statistics. ``None`` plots everything in
            ``slices``.
        indices : sequence of int or None
            Explicit test-set indices to plot.
        """    

        # ---- panels ------------------------------------------------------
        from HODDIES.fit_functions.plotting_emulator_func import _panels, _ylabel
        from matplotlib.gridspec import GridSpec

        panels = _panels(self, stats)
        if not panels:
            raise ValueError(
                f'no panels for stats={stats}; available slices: '
                f'{sorted(self.slices)}')

        tracers = list(dict.fromkeys(p['tracer'] for p in panels))
        per_tracer = {t: [p for p in panels if p['tracer'] == t] for t in tracers}
        npanel = max(len(v) for v in per_tracer.values())
        ncol = max(1, min(max_cols, npanel))
        nsub = int(np.ceil(npanel / ncol))
        nblock = len(tracers)

        default_colors = {'ELG': 'deepskyblue', 'QSO': 'seagreen',
                        'LRG': 'red', 'BGS': 'goldenrod'}
        colors = {**default_colors, **(colors or {})}

        # ---- which samples ----------------------------------------------
        
        fig = plt.figure(figsize=(4.6 * ncol,
                                4.4 * nsub * nblock + 0.9 * (nsub * nblock - 1)))
        outer = GridSpec(nsub * nblock, ncol, figure=fig,
                        hspace=block_hspace, wspace=wspace,
                        left=0.08, right=0.98, top=0.92, bottom=0.08)

        for b, tracer in enumerate(tracers):
            plist = per_tracer[tracer]
            color = colors.get(tracer, f'C{b}')

            for j, p in enumerate(plist):
                r, c = b * nsub + j // ncol, j % ncol
                sl, x, spec = p['sl'], p['x'], p['spec']

                if spec.ndim == 2:      # maps get the full cell
                    continue
                ax = fig.add_subplot(outer[r, c])
                ax.tick_params(labelbottom=False)

                for i in range(self.len_trainning): 
                    f = (lambda v: spec.scale(x, v)) if spec.scale else (lambda v: v)
                    y_train = f(self.y_training[i][sl])

                    ax.semilogy(x, y_train, lw=0.1, color=color)

                ax.set_ylabel(r'$\delta/\sigma$', fontsize=fontsize)
                ax.set_ylabel(_ylabel(p), fontsize=fontsize)
                ax.set_xscale(spec.xscale)
                ax.set_yscale(spec.yscale)
                # ax.set_ylim((0,1000))
                ax.grid(alpha=0.25)
                if j == 0:
                    ax.set_title(tracer, fontsize=fontsize + 1, loc='left')
                if data is not None:
                    yerr = f(data_err[sl]) if data_err is not None else None
                    ax.errorbar(x, f(data[sl]), yerr=yerr, fmt=':o', color='firebrick', label='data')

        if show:
            fig.tight_layout()
            plt.show()
        return fig

        
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
                 device: str = None,
                 var_loss_weight: float = 1.0,
                 file_best_fit_train: str = None,
                 verbose: bool = True,
                 use_std: bool = False
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
        var_loss_weight : float
            Weight for the variance loss term.
        file_best_fit_train : str, optional
            Path to a pre-trained model file to load.
        verbose : bool
            If True, print detailed information during training.
        use_std : bool
            If True, use standard deviation in the loss function.
        """
        super().__init__()
        self.n_input = n_input
        self.n_output = n_output
        if file_best_fit_train is not None:
            self.read_model_fit(file_best_fit_train)
        else:
            self.n_hidden = n_hidden
            self.learning_rate = learning_rate
            self.dropout_rate = dropout_rate
            self.weight_decay = weight_decay
    
        self.activation_fn = activation_fn
        self.loss_type = loss
        self.device = torch.device(DEVICE) if device is None else torch.device(device)
        self.var_loss_weight = var_loss_weight
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
                loss = nn.GaussianNLLLoss(full=True)(preds, y_true, var_pred)

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
        """
        Train the model for one epoch.
        """

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
        """
        Evaluate the model on a validation or test set.
        """
        self.eval()
        total_loss = 0.0
    
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(self.device, non_blocking=True)
            y_batch = y_batch.to(self.device, non_blocking=True)
    
            with autocast(enabled=self.use_amp):
                loss = self.compute_loss(X_batch, y_batch)
            total_loss += loss.item()
    
        return total_loss / len(dataloader)

    
    def read_model_fit(self, file_best_fit_train):
        """
        Read the best model fit saved after the training.

        PARAMETERS:
        -----------
        file_name : str
            The path of the file containing the best model fit.

        RETURNS:
        -------
        model_params : dict
            The parameters of the best model fit.
        """

        model_params = np.load(file_best_fit_train, allow_pickle=True)[()]
        
        n_hidden = [model_params[f'n_hidden_{n}'] for n in range(model_params['n_layers'])]
        self.n_hidden = n_hidden
        self.learning_rate = model_params['lr']
        self.dropout_rate = model_params['dropout_rate']
        self.weight_decay = model_params['weight_decay']


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
        

def make_training_dataset(dir_path:str, stats = ['wp', 'xi'], log_transform=False, seed=None, path_to_test_files=None, batch_size=256, num_testset=None, device=None):
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

    train_Dataset = Training_DatasetManager(dir_path, stats=stats, log_transform=log_transform, seed=seed, path_to_test_files=path_to_test_files, num_testset=num_testset, device=device)

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
    file_best_fit_train:str = None,
    use_std=False, 
    device:str = None
    ):
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
    device = torch.device(DEVICE) if device is None else torch.device(device)
    print(f"Using device: {device}")
    # device = 'cpu'
    nb_outpout = train_loader.dataset[0][1].shape[-1]
    nb_input = train_loader.dataset[0][0].shape[-1]

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
        use_std=use_std,
        file_best_fit_train=file_best_fit_train,
    )
    print(f"Model device {model.device}")
    if path_to_model is not None and os.path.isfile(path_to_model):
        model.load_model(path_to_model)
        print(f"Loaded model weights from {path_to_model}")
        return model

    
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


def Z_score(
    model, train_Dataset,
    use_var_pred=True,
    use_red_prior=False,
    use_normalize=False,      # True: calibrate in normalized space (recommended)
    rescale_sigma=False,     # optionally fit a global scale so Z.std() -> 1
    show=True
):
    """
    Z-score and reduced chi-square diagnostics for the emulator.

    Returns
    -------
    Z_score   : (N, B)  per-sample, per-bin standardized residuals
    chi2_red  : (N,)    reduced chi-square per test sample
    info      : dict    summary diagnostics (std, coverage, scale)
    """

    # ---- Remove edges of the training prior (optional) ----
    if use_red_prior:
        q_lo, q_hi = np.quantile(train_Dataset.X_training, q=[0.025, 0.985], axis=0)
        mask = ((train_Dataset.X_test > q_lo) & (train_Dataset.X_test < q_hi)).all(axis=1)
    else:
        mask = np.ones(train_Dataset.x_test_norm.shape[0], dtype=bool)


    X_test_norm = torch.tensor(train_Dataset.x_test_norm[mask], dtype=torch.float32)
    y_pred_norm, var_pred_norm = model.predict(X_test_norm, no_grad=True)

    y_pred_norm   = y_pred_norm.cpu().numpy()
    var_pred_norm = var_pred_norm.cpu().numpy()

    if use_normalize:
        Y_pred = y_pred_norm
        Y_test = train_Dataset.y_test_norm[mask]
        var    = var_pred_norm
    else:
        Y_pred, var = train_Dataset.normalizer.denormalize_y(y_pred_norm, var_pred_norm)
        Y_test = train_Dataset.y_test[mask]
    if use_var_pred:
        sigma = np.sqrt(var)
    else:
        
        # empirical per-bin error from the WHOLE test set (independent of the prior mask,
        # so it's a stable estimate), computed in the SAME space we're evaluating in.
        X_all = torch.tensor(train_Dataset.x_test_norm, dtype=torch.float32).to(model.device)
        y_pred_all_norm = model.predict(X_all, no_grad=True)[0].cpu().numpy()

        if use_normalize:
            Y_pred_all = y_pred_all_norm
            Y_test_all = train_Dataset.y_test_norm
        else:
            Y_pred_all, _ = train_Dataset.normalizer.denormalize_y(y_pred_all_norm, None)
            Y_pred_all = Y_pred_all
            Y_test_all = train_Dataset.y_test

        resid_all = Y_test_all - Y_pred_all
        sigma_bin = np.sqrt(np.mean(resid_all**2, axis=0))    # (B,) per-bin RMS error
        sigma = np.broadcast_to(sigma_bin, Y_pred.shape).copy()  # (N, B)
        
    # ---- optional global recalibration (fit ONE scalar so Z.std() -> 1) ----
    scale = 1.0
    if rescale_sigma:
        z_raw = (Y_test - Y_pred) / sigma
        scale = z_raw.std()          # if >1, sigma is too small; if <1, too large
        sigma = sigma * scale
    
    Z_score = (Y_test - Y_pred) / sigma
    cov68 = np.mean(np.abs(Z_score) < 1)
    cov95 = np.mean(np.abs(Z_score) < 2)

    from scipy.stats import skew, kurtosis
    skewness, kurt = skew(Z_score.ravel()), kurtosis(Z_score.ravel())
    # ---- summary diagnostics ----
    info = {
        "z_std": Z_score.std(),     # want ~1.0
        "coverage_68": cov68,     # want ~0.68
        "coverage_95": cov95,     # want ~0.95
        "skew": skewness,
        "kurtosis": kurt,
        "sigma_scale_applied": scale,
    }
    if show:

        z = Z_score.ravel()

        fig, ax = plt.subplots(figsize=(6.5, 4.5))

        ax.hist(z, bins=60, density=True, alpha=0.55,
                color="steelblue", edgecolor="none",
                label=f"emulator  (std={info['z_std']:.2f})")

        # reference unit Gaussian N(0,1)
        x_vals = np.linspace(-5, 5, 1000)
        gauss = np.exp(-(x_vals**2) / 2) / np.sqrt(2 * np.pi)
        ax.plot(x_vals, gauss,
                "k--", lw=1.5, label="N(0, 1)")

        # 1-sigma guides
        for s in (-1, 1):
            ax.axvline(s, color="grey", ls=":", lw=1)
        ax.axvline(0, color="k", lw=0.8)

        
        
        ax.set_xlim(-10, 10)
        ax.set_xlabel(f"Z-score")
        ax.set_ylabel("PDF")
        ax.set_title("Z-score mean {:.2f} std {:.2f} \n Coverage 68%: {:.2f}, 95%: {:.2f} \n Skew: {:.2f}, Kurtosis: {:.2f}".format(z.mean(),  z.std(), cov68, cov95, skewness, kurt), fontsize=15)
        ax.legend(frameon=False)
        fig.tight_layout()
    
    return Z_score, info