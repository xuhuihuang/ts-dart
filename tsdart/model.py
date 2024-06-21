import numpy as np
import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import *
from .loss import VAMPLoss, DisLoss, Prototypes
from .utils import map_data

class TSDARTLayer(nn.Module):
    """ Create TS-DART lobe.

    Parameters
    ----------
    layer_sizes : list
        The size of each layer of the encoder.
        The last component should represent the dimension of the euclidean space where the latent hypersphere is embedded.

    n_states : int
        Number of metastable states to be specified. 

    scale : int, default = 1
        The radius of the hypersphere.
    """

    def __init__(self, layer_sizes:list, n_states:int, scale=1):
        super().__init__()
        
        self.hypersphere_embs = None
        self.logits = None
        self.probs = None

        self.scale = scale
        self.n_states = n_states

        encoder = [nn.BatchNorm1d(layer_sizes[0])]
        for i in range(len(layer_sizes)-1):
            encoder.append(nn.Linear(layer_sizes[i],layer_sizes[i+1]))
            encoder.append(nn.ReLU()) if i<(len(layer_sizes)-2) else None
        # note that the last element in layer_sizes represent the latent dimension (feat_dim).

        self.encoder = nn.Sequential(*encoder)
        self.lt = nn.Sequential(nn.Linear(layer_sizes[-1],n_states,bias=True))
        self.sf = nn.Softmax(dim=-1)

    def forward(self, x):
        
        self.hypersphere_embs = self.encoder(x)
        self.hypersphere_embs = self.scale*self.hypersphere_embs/(torch.sqrt(torch.sum(self.hypersphere_embs**2,dim=-1,keepdim=True)))
        self.logits = self.lt(self.hypersphere_embs)
        self.probs = self.sf(self.logits)

        return self.probs

class TSDARTModel:
    """ The TS-DART model from TS-DART.

    Parameters
    ----------
    lobe : torch.nn.Module
        TS-DART lobe.

    device : torch device, default = None
        The device on which the torch modules are executed.

    dtype : dtype, default = np.float32
        The data type of the input data and the parameters of the model.
    """

    def __init__(self, lobe, device=None, dtype=np.float32):

        self._lobe = lobe
        if dtype == np.float32:
            self._lobe = self._lobe.float()
        elif dtype == np.float64:
            self._lobe = self._lobe.double()

        self._dtype = dtype
        self._device = device

    @property
    def lobe(self):
        return self._lobe

    def transform(self, data, return_type='probs'):
        """ Transform the original trajectores to different outputs after training.

        Parameters
        ----------
        data : list or ndarray
            The original trajectories.
        
        return_type : string
            'probs': the softmax probabilties to assign each conformation to a metastable state.
            'states': the metastable state assignments of each conformation.
            'hypersphere_embs': the hyperspherical embeddings of each conformation. 
        """

        ### return_type: 'probs' or 'states' 'hypersphere_embs'
        self._lobe.eval()
        net = self._lobe

        probs = []
        states = []
        hypersphere_embs = []
        for data_tensor in map_data(data, device=self._device, dtype=self._dtype):
            probs.append(net(data_tensor).cpu().numpy())
            states.append(np.argmax(probs[-1],axis=-1))
            hypersphere_embs.append(net.hypersphere_embs.cpu().numpy())

        if return_type == 'probs':
            return probs if len(probs) > 1 else probs[0]
        elif return_type == 'states':
            return states if len(states) > 1 else states[0]
        elif return_type == 'hypersphere_embs':
            return hypersphere_embs if len(hypersphere_embs) > 1 else hypersphere_embs[0]
        else:
            raise ValueError('Valid return types: probs, states, hypersphere_embs')

class TSDART:
    """ The method used to train TS-DART.

    Parameters
    ----------
    data : list or ndarray
        The original trajectories.
    
    optimizer : str, default = 'Adam'
        The type of optimizer used for training.

    device : torch.device, default = None
        The device on which the torch modules are executed.

    learning_rate : float, default = 1e-3
        The learning rate of the optimizer.

    epsilon : float, default = 1e-6
        The strength of the regularization/truncation under which matrices are inverted.

    mode : str, default = 'regularize'
        'regularize': regularize the eigenvalues by adding epsilon.
        'trunc': truncate the eigenvalues by filtering out the eigenvalues below epsilon.

    symmetrized : boolean, default = False
        Whether to symmetrize time-correlation matrices or not. 

    dtype : dtype, default = np.float32
        The data type of the input data and the parameters of the model.
    
    feat_dim : int, default = 2
        The dimension of the euclidean space where the latent hypersphere is embedded.
        The dimension of latent hypersphere is (feat_dim-1).

    n_states : int, default = 4
        Number of metastable states to be specified. 

    proto_update_factor : float, default = 0.5
        The state center update factor.

    scaling_temperature : float, default = 0.1
        The scaling hyperparameter to compute dispersion loss. 

    beta : float, default = 0.01
        The weight of dispersion loss.

    save_model_interval : int, default = None
        Saving the model every 'save_model_interval' epochs.

    pretrain : int, default = 0
        The number of epochs of the pretraining with pure VAMP2 loss.

    print : boolean, default = False
        Whether to print the validation loss every epoch during the training. 
    """

    def __init__(self, lobe, optimizer='Adam', device=None, learning_rate=1e-3,
                 epsilon=1e-6, mode='regularize', symmetrized=False, dtype=np.float32, 
                 feat_dim=2, n_states=4, proto_update_factor=0.5, scaling_temperature=0.1, beta=0.01, 
                 save_model_interval=None, pretrain=0, print=False):
        
        self._lobe = lobe
        self._device = device
        self._learning_rate = learning_rate
        self._epsilon = epsilon
        self._mode = mode
        self._symmetrized = symmetrized
        self._dtype = dtype
        self._feat_dim = feat_dim
        self._n_states = n_states
        self._proto_update_factor = proto_update_factor
        self._scaling_temperature = scaling_temperature
        self._beta = beta
        self._save_model_interval = save_model_interval
        self._pretrain = pretrain
        if self._dtype == np.float32:
            self._lobe = self._lobe.float()
        elif self._dtype == np.float64:
            self._lobe = self._lobe.double()
        self._step = 0
        self._epoch = 0
        self._save_models = []
        self._print = print
        self.optimizer_types = {'Adam': torch.optim.Adam, 'SGD': torch.optim.SGD, 'RMSprop': torch.optim.RMSprop}
        if optimizer not in self.optimizer_types.keys():
            raise ValueError(f"Unknown optimizer type, supported types are {self.optimizer_types.keys()}")
        else:
            self._optimizer = self.optimizer_types[optimizer](self._lobe.parameters(), lr=learning_rate)
        self._training_vamp = []
        self._training_dis = []
        self._validation_vamp = []
        self._validation_dis = []
        self._validation_prototypes = []
        self._vamploss = VAMPLoss(epsilon=self._epsilon, mode=self._mode, symmetrized=self._symmetrized)
        self._disloss = DisLoss(feat_dim=feat_dim,n_states=n_states,device=device,proto_update_factor=proto_update_factor,scaling_temperature=scaling_temperature)
        self._proto = Prototypes(n_states=n_states,device=device,scaling_temperature=scaling_temperature)

    @property
    def training_vamp(self):
        return np.array(self._training_vamp)

    @property
    def training_dis(self):
        return np.array(self._training_dis)
    
    @property
    def validation_vamp(self):
        return np.array(self._validation_vamp)

    @property
    def validation_dis(self):
        return np.array(self._validation_dis)
    
    @property
    def validation_prototypes(self):
        return np.array(self._validation_prototypes)
    
    def partial_fit(self, data):

        batch_0, batch_1 = data[0], data[1]
        self._lobe.train()
        self._optimizer.zero_grad()
        x_0 = self._lobe(batch_0)
        z_0 = self._lobe.hypersphere_embs
        x_1 = self._lobe(batch_1)
        labels_0 = torch.argmax(x_0,dim=-1).detach()
        lv = self._vamploss([x_0,x_1])
        if self._epoch >= self._pretrain:
            ld = self._disloss(z_0,labels_0)
            loss = lv + self._beta*ld
        else:
            ld = torch.tensor(0.)
            loss = lv
        loss.backward()
        self._optimizer.step()
        self._training_vamp.append((-lv).item())
        self._training_dis.append((ld).item())
        self._step += 1

        return self
    
    def validate(self, val_data):

        batch_0, batch_1 = val_data[0], val_data[1]
        self._lobe.eval()
        with torch.no_grad():
            x_0 = self._lobe(batch_0)
            z_0 = self._lobe.hypersphere_embs
            x_1 = self._lobe(batch_1)
            labels_0 = torch.argmax(x_0,dim=-1).detach()
            _ = self._proto(z_0,labels_0)
            _ = self._vamploss([x_0,x_1])
            self._vamploss.save()
            
        return None
    
    def fit(self, train_loader, n_epochs=1, validation_loader=None, progress=tqdm):
        """ Performs fit on data.

        Parameters
        ----------
        train_loader : torch.utils.data.DataLoader
            Yield a tuple of batches representing instantaneous and time-lagged samples for training.

        n_epochs : int, default=1
            The number of epochs to use for training.
            Note that n_epochs should be larger than pretrain. 

        validation_loader : torch.utils.data.DataLoader, optional, default=None
             Yield a tuple of batches representing instantaneous and time-lagged samples for validation.

        progress : context manager, default=tqdm

        Returns
        -------
        self : TSDART
        """

        for epoch in progress(range(n_epochs), desc="epoch", total=n_epochs, leave=False):
            for batch_0, batch_1 in train_loader:
                self.partial_fit((batch_0.to(device=self._device), batch_1.to(device=self._device)))
            if validation_loader is not None:
                with torch.no_grad():
                    for val_batch_0, val_batch_1 in validation_loader:
                        self.validate((val_batch_0.to(device=self._device), val_batch_1.to(device=self._device)))
                    mean_vamp = self._vamploss.output_mean_score()
                    mean_dis = self._proto.output_mean_disloss()
                    mean_prototypes = self._proto.output_mean_prototypes()
                    self._validation_vamp.append(mean_vamp.item())
                    self._validation_dis.append(mean_dis.item())
                    self._validation_prototypes.append(mean_prototypes.cpu().numpy())
                    self._vamploss.clear()
                    self._proto.clear()
                    if self._print:
                        print(epoch, mean_vamp.item(), mean_dis.item())
                    if self._save_model_interval is not None:
                        if (epoch + 1) % self._save_model_interval == 0:
                            self._save_models.append(self.fetch_model())

            self._epoch = self._epoch+1
            
            if self._epoch == self._pretrain and validation_loader is not None:
                self._disloss.prototypes = mean_prototypes
                    
        return self
    
    def fetch_model(self):

        from copy import deepcopy
        lobe = deepcopy(self._lobe)

        return TSDARTModel(lobe, device=self._device, dtype=self._dtype)
    
class TSDARTEstimator:
    """ The TS-DART estimator the generate the state center vectors and ood scores of original trajectories.

    Parameters
    ----------
    tsdart_model : TSDARTModel
        The trained TS-DART model.
    """

    def __init__(self, tsdart_model: TSDARTModel):

        self._model = tsdart_model
        self._state_centers = None
        self._ood_scores = None
        self._is_fitted = False

    @property
    def state_centers(self):
        if not self._is_fitted:
            raise ValueError('Please fit the model first')
        else:
            return self._state_centers
    
    @property
    def ood_scores(self):
        if not self._is_fitted:
            raise ValueError('Please fit the model first')
        else:
            return self._ood_scores

    def fit(self, data):
        """ Fit the TS-DART model with original trajectories to compute OOD scores and state center vectors.

        Parameters
        ----------
        data : list or ndarray
            The original trajectories.

        Returns
        -------
        TSDARTEstimator
        """

        states = self._model.transform(data, return_type='states')
        hypersphere_embs = self._model.transform(data, return_type='hypersphere_embs')

        if isinstance(data, (list, tuple)) and len(data) >= 2:
            states_cat = np.concatenate(states)
            hypersphere_embs_cat = np.concatenate(hypersphere_embs)
        else:
            states_cat = states
            hypersphere_embs_cat = hypersphere_embs

        p = Prototypes(n_states=self._model.lobe.n_states,device='cpu')
        self._state_centers = p(torch.from_numpy(hypersphere_embs_cat),torch.from_numpy(states_cat)).numpy()

        if isinstance(data, (list, tuple)) and len(data) >= 2:
            self._ood_scores = []
            for i in range(len(data)):
                self._ood_scores.append(-np.max(np.dot(hypersphere_embs[i],self._state_centers.T),axis=1)+1)
        else:
            self._ood_scores = -np.max(np.dot(hypersphere_embs,self._state_centers.T),axis=1)+1

        self._is_fitted = True

        return self
    