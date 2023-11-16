from typing import Tuple, Union
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def get_drp_coverage(
    samples: np.ndarray,
    theta: np.ndarray,
    normalize: bool,
    references: Union[str, np.ndarray, list] = "random",
    metric: str = "euclidean",
    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimates coverage with the distance to random point method.

    Reference: `Lemos, Coogan et al 2023 <https://arxiv.org/abs/2302.03026>`_

    Args:
        samples: the samples to compute the coverage of, with shape ``(n_samples, n_sims, n_dims)``.
        theta: the true parameter values for each samples, with shape ``(n_sims, n_dims)``.
        references: the reference points to use for the DRP regions, with shape
            ``(n_references, n_sims)``, or the string ``"random"``. If the later, then
            the reference points are chosen randomly from the unit hypercube over
            the parameter space.
        metric: the metric to use when computing the distance. Can be ``"euclidean"`` or
            ``"manhattan"``.

    Returns:
        Credibility values (``alpha``) and expected coverage probability (``ecp``).
    """

    # Check that shapes are correct
    if samples.ndim != 3:
        raise ValueError("samples must be a 3D array")

    if theta.ndim != 2:
        raise ValueError("theta must be a 2D array")

    num_samples = samples.shape[0]
    num_sims = samples.shape[1]
    num_dims = samples.shape[2]

    if theta.shape[0] != num_sims:
        raise ValueError("theta must have the same number of rows as samples")

    if theta.shape[1] != num_dims:
        raise ValueError("theta must have the same number of columns as samples")

    # Reshape theta
    theta = theta[np.newaxis, :, :]

    # Normalize
    if normalize==True:
        low = np.min(theta, axis=1, keepdims=True)
        high = np.max(theta, axis=1, keepdims=True)
        samples = (samples - low) / (high - low + 1e-10)
        theta = (theta - low) / (high - low + 1e-10)

    # Generate reference points
    if isinstance(references, str) and references == "random":
            references = np.random.uniform(low=0, high=1, size=(num_sims, num_dims))

    if isinstance(references, list):
        if len(references) != 2:
            raise ValueError('Your list should specify max and min of uniform distribution.' 
                             'Should be a list containing two values [min, max] in that order.')
        
        references = np.random.uniform(low=references[0], high=references[1], size=(num_sims, num_dims))

    else:
        assert isinstance(references, np.ndarray)  # to quiet pyright
        if references.ndim != 2:
            raise ValueError("references must be a 2D array")

        if references.shape[0] != num_sims:
            raise ValueError("references must have the same number of rows as samples")

        if references.shape[1] != num_dims:
            raise ValueError(
                "references must have the same number of columns as samples"
            )

    # Compute distances
    if metric == "euclidean":
        samples_distances = np.sqrt(
            np.sum((references[np.newaxis] - samples) ** 2, axis=-1)
        )
        theta_distances = np.sqrt(np.sum((references - theta) ** 2, axis=-1))
    elif metric == "manhattan":
        samples_distances = np.sum(np.abs(references[np.newaxis] - samples), axis=-1)
        theta_distances = np.sum(np.abs(references - theta), axis=-1)
    else:
        raise ValueError("metric must be either 'euclidean' or 'manhattan'")

    # Compute coverage
    f = np.sum((samples_distances < theta_distances), axis=0) / num_samples

    # Compute expected coverage
    h, alpha = np.histogram(f, density=True, bins=num_sims // 10, range=(0,1) )
    dx = alpha[1] - alpha[0]
    ecp = np.cumsum(h) * dx
    return np.insert(ecp,0,0), alpha


def bootstrap( true, samples, num_boot, boot_type, reference='random', 
               norm=False, data_val=False, plot=False, filename=None, maxs=False ):
    #Tarp Coverage with Bootsraping
    ecp = []
    alpha = []
    
    theta_maxs = []
    theta_mins = []
    
    for i in range( num_boot ):

        if boot_type == 'truths':
            #Resample ground truths with substitution
            rd_ints = np.random.randint(0, true.shape[0], size=true.shape[0])
            rd_truths = true[rd_ints, :]
            rd_samples = samples[:, rd_ints, :]

        if boot_type == 'samples':
            #Resample generated samples with substitution 
            rd_ints = np.random.randint(0, samples.shape[0], size=samples.shape[0])
            rd_truths = true
            rd_samples = samples[rd_ints, :, :]

        if boot_type == 'both':
            #Resample both ground truths and generated samples with substitution 
            rd_ints_true = np.random.randint(0, true.shape[0], size=true.shape[0])
            rd_truths = true[rd_ints_true, :]
            
            rd_ints_samp = np.random.randint(0, samples.shape[0], size=samples.shape[0])
            rd_samples = samples[:, rd_ints_true, :]
            rd_samples = samples[rd_ints_samp, :, :]

        if maxs is True:
            theta_maxs.append( np.max(rd_truths, axis=0) )
            theta_mins.append( np.min(rd_truths, axis=0) )

        #Do coverage test over those resampled ground truths 
        cov_test = get_drp_coverage(rd_samples, rd_truths, normalize=norm, references=reference, metric='euclidean')
        ecp.append(cov_test[0])
        alpha.append(cov_test[1])


    #Turn the lists into numpy arrays
    np_ecp = np.array(ecp)
    np_alpha = np.array(alpha)

    #Calculate the mean and std
    ecp_mean = np.mean( np_ecp, axis=0)
    ecp_std = np.std( np_ecp, axis=0, ddof=1 )
    alpha_mean = np.mean( np_alpha, axis=0)

    if plot is True:
        #plot them with error bars
        plt.fill_between( x=alpha_mean, y1=ecp_mean-ecp_std, y2=ecp_mean+ecp_std, alpha=0.9, color='royalblue', label=r'$\sigma$')
        plt.fill_between( x=alpha_mean, y1=ecp_mean-3*ecp_std, y2=ecp_mean+3*ecp_std, alpha=0.4, color='lightsalmon',  label=r'$3\sigma$') 
        plt.plot([0,1], [0,1], "--", label='Calibrated', color='red')

        plt.xlabel('Credibility')
        plt.ylabel('Expected Coverage Probability')
        plt.legend()       
        
        if filename is not None:
            plt.savefig( filename, dpi=300)
        
        plt.show()
        plt.close()
    
    if data_val is True :
        print(f"ecp_mean:{ecp_mean} \necp_std:{ecp_std}")

    if  maxs is True:
        mins_std = np.std(np.array(theta_mins), axis=0, ddof=1)
        maxs_std = np.std(np.array(theta_maxs), axis=0, ddof=1)
    
        print(f"\nnorm min std: {mins_std[0]} \nnorm max std: {maxs_std[0]}")