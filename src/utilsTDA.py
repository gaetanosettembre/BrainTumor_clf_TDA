"""

@author: Serena Grazia De Benedictis, Grazia Gargano, Gaetano Settembre
"""

import time
import numpy as np
import numpy as np # handling arrays and general math
from scipy import sparse # working with sparse matrices
from ripser import lower_star_img # computing topological persistence of images
from scipy.sparse.csgraph import connected_components # compute connected components from sparse adjacency matrix
import cv2 # image processing library
#import random # setting seeds
from scipy import ndimage # image smoothening
#import PIL # imaging library
from scipy.ndimage.morphology import distance_transform_edt # compute closest background pixel
from skimage.measure import find_contours # find iso-valued contours in an image


def img_to_sparseDM(img):
    """
    Compute a sparse distance matrix from the pixel entries of a single channel image for persistent homology
    
    Parameters
    ----------
    img: ndarray (M, N)
        An array of single channel image data
        Infinite entries correspond to empty pixels
        
    Returns
    -------
    sparseDM: scipy.sparse (M * N, M * N)
        A sparse distance matrix representation of img
    """
    m, n = img.shape

    idxs = np.arange(m * n).reshape((m, n))

    I = idxs.flatten()
    J = idxs.flatten()
    
    # Make sure non-finite pixel entries get added at the end of the filtration
    img[img==-np.inf] = np.inf
    V = img.flatten()

    # Connect 8 spatial neighbors
    tidxs = np.ones((m + 2, n + 2), dtype=np.int64) * np.nan
    tidxs[1:-1, 1:-1] = idxs

    tD = np.ones_like(tidxs) * np.nan
    tD[1:-1, 1:-1] = img

    for di in [-1, 0, 1]:
        for dj in [-1, 0, 1]:

            if di == 0 and dj == 0:
                continue

            thisJ = np.roll(np.roll(tidxs, di, axis=0), dj, axis=1)
            thisD = np.roll(np.roll(tD, di, axis=0), dj, axis=1)
            thisD = np.maximum(thisD, tD)

            # Deal with boundaries
            boundary = ~np.isnan(thisD)
            thisI = tidxs[boundary]
            thisJ = thisJ[boundary]
            thisD = thisD[boundary]

            I = np.concatenate((I, thisI.flatten()))
            J = np.concatenate((J, thisJ.flatten()))
            V = np.concatenate((V, thisD.flatten()))
            
    return sparse.coo_matrix((V, (I, J)), shape=(idxs.size, idxs.size))


def connected_components_img(img):
    """
    Identify the connected components of an image
    
    Parameters
    ----------
    img: ndarray (M, N)
        An array of single channel image data
        Infinite entries correspond to empty pixels
        
    Returns
    -------
    img: ndarray (M, N)
        An array of single channel image data where each pixel value equals its modified lifespan
    """
    
    m, n = img.shape
    
    component = connected_components(img_to_sparseDM(img), directed=False)[1].reshape((m, n))
            
    return component


def smoothen(img, window_size):
    
    return(ndimage.uniform_filter(img.astype("float"), size=window_size))


def add_border(img, border_width):
    
    border_value = np.min(img) - 1 # make sure the pixels near the border reach te minimal value
    
    img[0:border_width,:] = border_value
    img[(img.shape[0] - border_width):img.shape[0],:] = border_value
    img[:,0:border_width] = border_value
    img[:,(img.shape[1] - border_width):img.shape[1]] = border_value

    return(img)


def lifetimes_from_dgm(dgm, tau=False):
    """
    Rotate a persistence diagram by 45 degrees, to indicate lifetimes by the y-coordinate
    
    Parameters
    ----------
    dgm: ndarray (K, 2)
        The persistence diagram to rotate
    tau: boolean
        Whether to return a threshold for indentifying connected components
        
    Returns
    -------
    dgm_lifetimes: ndarray (K, 2)
        The rotated diagram
        
    tau: float
        A threshold for identifying connected components 
        as those with finite oordinate above tau in the rotated diagram
    """ 
    
    dgm_lifetimes = np.vstack([dgm[:,0], dgm[:,1] - dgm[:,0]]).T
        
    if(tau):
        dgm_for_tau = np.delete(dgm_lifetimes.copy(), np.where(dgm_lifetimes[:,1] == np.inf)[0], axis=0)
        sorted_points = dgm_for_tau[:,1]
        sorted_points[::-1].sort()
        dist_to_next = np.delete(sorted_points, len(sorted_points) - 1) - np.delete(sorted_points, 0)
        most_distant_to_next = np.argmax(dist_to_next)
        tau = (sorted_points[most_distant_to_next] + sorted_points[most_distant_to_next + 1]) / 2
        
        return dgm_lifetimes, tau
    
    return dgm_lifetimes


def topological_process_img(img, dgm=None, tau=None, window_size=None, border_width=None):
    
    return_modified = False
    if dgm is None:
        if window_size is not None:
            img = smoothen(img, window_size=window_size)
            return_modified = True
            
        if border_width is not None:
            img = add_border(img, border_width=border_width)
            return_modified = True
            
        dgm = lower_star_img(img)
    
    if tau is None:
        dgm_lifetimes, tau = lifetimes_from_dgm(dgm, tau=True)
        
    else:
        dgm_lifetimes = lifetimes_from_dgm(dgm)
        
    idxs = np.where(np.logical_and(tau < dgm_lifetimes[:,1], dgm_lifetimes[:,1] < np.inf))[0]
    idxs = np.flip(idxs[np.argsort(dgm[idxs, 0])])
    didxs = np.zeros(0).astype("int")
    
    img_components = np.zeros_like(img)

    dist = np.zeros([len(idxs), img.shape[0], img.shape[1]])
    nearest_value = np.zeros([len(idxs), img.shape[0], img.shape[1]])

    for i, idx in enumerate(idxs):
        bidx = np.argmin(np.abs(img - dgm[idx, 0]))
        didxs = np.append(didxs, np.argmin(np.abs(img - dgm[idx, 1])))

        img_temp = np.ones_like(img)
        img_temp[np.logical_or(img < dgm[idx, 0] - 0.01, dgm[idx, 1] - 0.01 < img)] = np.nan
        component_at_idx = connected_components_img(img_temp)
        del(img_temp)

        component_at_idx = component_at_idx == component_at_idx[bidx // img.shape[1], bidx % img.shape[1]]
        if i > 0:
            didxs_in_component = idxs[np.where([component_at_idx[didx // img.shape[1], didx % img.shape[1]] 
                                                for didx in didxs])[0]]
            if len(didxs_in_component) > 0:
                component_at_idx[img > np.min(dgm[didxs_in_component, 1]) - 0.1] = False

        img_components[component_at_idx] = 1

        img_temp = np.ones_like(img)
        img_temp[component_at_idx] = 0
        dist[i,:,:], nearest_neighbor_temp = distance_transform_edt(img_temp, return_indices=True)
        nearest_value[i,:,:] = img[nearest_neighbor_temp[0], nearest_neighbor_temp[1]]
        del(img_temp, nearest_neighbor_temp)
        
    img_processed = np.zeros_like(img)
    all_components = img_components > 0
    img_processed[all_components] = img[all_components]
    with np.errstate(divide="ignore"):
        img_processed[~all_components] = np.sum(nearest_value / dist, axis=0)[~all_components] / \
                                            np.sum(1 / dist, axis=0)[~all_components]
     
    if return_modified:
        return {"modified": img, "components": img_components, "processed": img_processed}
    
    return {"components": img_components, "processed": img_processed}



def tda_and_mask(x_data, window_size, border_width, param_mask):
    x_data_tda = []
    cc = []

    start_time = time.time()
    for i in range(len(x_data)):
        image = x_data[i]

        TIP_img = topological_process_img(image, window_size=window_size, border_width=border_width)
        img_components = TIP_img["components"]
        img_processed = TIP_img["processed"]

        cc.append(img_components)
        # ip.append(img_processed)

    for i in range(len(x_data)):
        image = 1 - x_data[i]
        # TPI_img= ip[i]
        mask = cc[i]
        mask = np.where(mask == 0, param_mask, mask)
        weighted_image = mask * image
        x_data_tda.append(weighted_image)

    elapsed_time = time.time() - start_time
    print("Time for Topological Image Processing: " + time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
    return x_data_tda

def extractROI(x_train, window_size=10, border_width=70):
    cc = []
    for i in range(len(x_data)):
        image = x_data[i]

        TIP_img = topological_process_img(image, window_size=window_size, border_width=border_width)
        img_components = TIP_img["components"]
        img_processed = TIP_img["processed"]

        cc.append(img_components)
    return cc

def image_with_ROI(x_data_wp,cc,param_mask):
    x_data_tda = []
    for i in range(len(x_data_wp)):
        image = x_data_wp[i]
        mask = cc[i]
        mask = np.where(mask == 0, param_mask, mask)
        weighted_image = mask * image
        x_data_tda.append(weighted_image)
    return x_data_tda