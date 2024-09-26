import numpy as np
import torch

def set_random_seed(seed):
    """ Set a random seed. """

    import random
    import os
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def eig_decomposition(matrix, epsilon=1e-6, mode='regularize'):
    """ This method can be applied to do the eig-decomposition for a rank deficient hermetian matrix,
    this method will be further used to estimate koopman matrix.

    Parameters
    ----------
    matrix : torch.Tensor
        The hermitian matrix: specifically, the covariance matrix.

    epsilon : float, default = 1e-6
        The regularization/trunction parameters for eigenvalues.

    mode : str, default = 'regularize'
        'regularize': regularize the eigenvalues by adding epsilon.
        'trunc': truncate the eigenvalues by filtering out the eigenvalues below epsilon.

    Returns
    -------
    (eigval, eigvec) : Tuple[torch.Tensor, torch.Tensor]
        Eigenvalues and eigenvectors.
    """

    if mode == 'regularize':
        identity = torch.eye(matrix.shape[0], dtype=matrix.dtype, device=matrix.device)
        matrix = matrix + epsilon * identity
        eigval, eigvec = torch.linalg.eigh(matrix)
        eigval = torch.abs(eigval)
        eigvec = eigvec.transpose(0, 1)  # row -> column

    elif mode == 'trunc':
        eigval, eigvec = torch.linalg.eigh(matrix)
        eigvec = eigvec.transpose(0, 1)
        mask = eigval > epsilon
        eigval = eigval[mask]
        eigvec = eigvec[mask]

    else:
        raise ValueError('Mode is not included')

    return eigval, eigvec

def calculate_inverse(matrix, epsilon=1e-6, return_sqrt=False, mode='regularize'):
    """ This method can be applied to compute the inverse or the square-root of the inverse of the matrix,
    this method will be further used to estimate koopman matrix.

    Parameters
    ----------
    matrix : torch.Tensor
        The matrix to be inverted.

    epsilon : float, default = 1e-6
        The regularization/trunction parameters for eigenvalues.

    return_sqrt : boolean, optional, default = False
        If True, the square root of the inverse matrix is returned instead.

    mode : str, default = 'regularize'
        'regularize': regularize the eigenvalues by adding epsilon.
        'trunc': truncate the eigenvalues by filtering out the eigenvalues below epsilon.

    Returns
    -------
    inverse : torch.Tensor
        Inverse of the matrix.
    """

    eigval, eigvec = eig_decomposition(matrix, epsilon, mode)

    if return_sqrt:
        diag = torch.diag(torch.sqrt(1. / eigval))
    else:
        diag = torch.diag(1. / eigval)

    try:
    # inverse = torch.chain_matmul(eigvec.t(), diag, eigvec)
        inverse = torch.linalg.multi_dot((eigvec.t(), diag, eigvec))
    except:
        inverse = torch.chain_matmul(eigvec.t(), diag, eigvec)

    return inverse

def compute_covariance_matrix(x: torch.Tensor, y: torch.Tensor, remove_mean=True):
    """ This method can be applied to compute the covariance matrix from two batches of data.

    Parameters
    ----------
    x : torch.Tensor
        The first batch of data of shape [batch_size, num_basis].

    y : torch.Tensor
        The second batch of data of shape [batch_size, num_basis].

    remove_mean : boolean, optional, default = True
        Whether to remove mean of the data.

    Returns
    -------
    (cov_00, cov_01, cov11) : Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        Instantaneous covariance matrix of x, time-lagged covariance matrix of x and y,
        and instantaneous covariance matrix of y.
    """

    batch_size = x.shape[0]

    if remove_mean:
        x = x - x.mean(dim=0, keepdim=True)
        y = y - y.mean(dim=0, keepdim=True)

    y_t = y.transpose(0, 1)
    x_t = x.transpose(0, 1)

    cov_01 = 1 / (batch_size - 1) * torch.matmul(x_t, y)
    cov_00 = 1 / (batch_size - 1) * torch.matmul(x_t, x)
    cov_11 = 1 / (batch_size - 1) * torch.matmul(y_t, y)

    return cov_00, cov_01, cov_11

def estimate_koopman_matrix(data: torch.Tensor, data_lagged: torch.Tensor, epsilon=1e-6, mode='regularize', symmetrized=False):
    """ This method can be applied to compute the koopman matrix from time-instant and time-lagged data.

    Parameters
    ----------
    data : torch.Tensor
        The time-instant data of shape [batch_size, num_basis].

    data_lagged : torch.Tensor
        The time-lagged data of shape [batch_size, num_basis].

    epsilon : float, default = 1e-6
        The regularization/trunction parameters for eigenvalues.

    mode : str, default = 'regularize'
        'regularize': regularize the eigenvalues by adding epsilon.
        'trunc': truncate the eigenvalues by filtering out the eigenvalues below epsilon.

    symmetrized : boolean, default = False
        Whether to symmetrize time-correlation matrices or not. 

    Returns
    -------
    koopman_matrix : torch.Tensor
        The koopman matrix of shape [num_basis, num_basis].
    """

    cov_00, cov_01, cov_11 = compute_covariance_matrix(data, data_lagged)

    if not symmetrized:
        cov_00_sqrt_inverse = calculate_inverse(cov_00, epsilon=epsilon, return_sqrt=True, mode=mode)
        cov_11_sqrt_inverse = calculate_inverse(cov_11, epsilon=epsilon, return_sqrt=True, mode=mode)
        try:
            koopman_matrix = torch.linalg.multi_dot((cov_00_sqrt_inverse, cov_01, cov_11_sqrt_inverse)).t()
        except:
            koopman_matrix = torch.chain_matmul(cov_00_sqrt_inverse, cov_01, cov_11_sqrt_inverse).t()
    else:
        cov_0 = 0.5*(cov_00+cov_11)
        cov_1 = 0.5*(cov_01+cov_01.t())
        cov_0_sqrt_inverse = calculate_inverse(cov_0, epsilon=epsilon, return_sqrt=True, mode=mode)
        try:
            koopman_matrix = torch.linalg.multi_dot((cov_0_sqrt_inverse, cov_1, cov_0_sqrt_inverse)).t()
        except:
            koopman_matrix = torch.chain_matmul((cov_0_sqrt_inverse, cov_1, cov_0_sqrt_inverse)).t()

    return koopman_matrix

def map_data(data, device=None, dtype=np.float32):
    """ This function is used to yield the torch.Tensor type data from multiple trajectories.

    Parameters
    ----------
    data : list or tuple or ndarray
        The trajectories of data.

    device : torch device, default = None
        The device on which the torch modules are executed.
        
    dtype : dtype, default = np.float32
        The data type of the input data and the parameters of the model.

    Yields
    ------
    x : torch.Tensor
        The mapped data.
    """

    with torch.no_grad():
        if not isinstance(data, (list, tuple)):
            data = [data]
        for x in data:
            if isinstance(x, torch.Tensor):
                x = x.to(device=device)
            else:
                x = torch.from_numpy(np.asarray(x, dtype=dtype).copy()).to(device=device)
            yield x
            