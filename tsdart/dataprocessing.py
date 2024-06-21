import numpy as np

class Preprocessing:

    def __init__(self, dtype=np.float32):
        """ Preprocess the original trajectories to create datasets for training.

        Parameters
        ----------
        dtype : dtype, default = np.float32
        """

        self._dtype = dtype

    def _seq_trajs(self, data):

        data = data.copy()
        if not isinstance(data, list):
            data = [data]
        for i in range(len(data)):
            data[i] = data[i].astype(self._dtype)
        
        return data

    def transform2pw(self, data):
        """ Transform xyz coordinates data to pairwise distances data.

        Parameters
        ----------
        data : list or ndarray
            xyz coordinates data, shape of each traj [num_frames,num_atoms,3].

        Returns
        -------
        pw_data : list or ndarray
            Pairwise distances data.
        """

        data = self._seq_trajs(data)

        if not (len(data[0].shape) == 3 and data[0].shape[-1] == 3):
            raise ValueError('Please make sure the shape of each traj is [num_frames,num_atoms,3]')
        
        num_trajs = len(data)
        num_atoms = data[0].shape[1]

        pw_data = []
        for traj in range(num_trajs):
            tmp = []
            for i in range(num_atoms-1):
                for j in range(i+1, num_atoms):
                    dist = np.sqrt(np.sum((data[traj][:,i,:] - data[traj][:,j,:])**2, axis=-1))
                    tmp.append(dist)
            pw_data.append(np.stack(tmp,axis=1))
        
        return pw_data if num_trajs > 1 else pw_data[0]
    
    def create_dataset(self, data, lag_time):
        """ Create the dataset as the input to TS-DART.

        Parameters
        ----------
        data : list or ndarray
            The original trajectories.

        lag_time : int
            The lag_time used to create the dataset consisting of time-instant and time-lagged data.

        Returns
        -------
        dataset : list
            List of tuples: the length of the list represents the number of data.
            Each tuple has two elements: one is the instantaneous data frame, the other is the corresponding time-lagged data frame.
        """

        data = self._seq_trajs(data)

        num_trajs = len(data)
        dataset = []
        for k in range(num_trajs):
            L_all = data[k].shape[0]
            L_re = L_all - lag_time
            for i in range(L_re):
                dataset.append((data[k][i,:], data[k][i+lag_time,:]))

        return dataset    
    