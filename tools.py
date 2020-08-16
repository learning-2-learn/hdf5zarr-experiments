from dask.distributed import Client
from dask_kubernetes import KubeCluster
from dask_gateway import Gateway
from dask import bag as db
from scipy.io import loadmat
from tqdm import tqdm_notebook as tqdm
import pandas as pd
import numpy as np
import h5py
import os



def init_cluster(n_workers=10):
    """ 
    Initialize a dask cluster
    
    Parameters
    ----------
    n_workers : int.
        Number of workers in the cluster. Default: 10.
        
    Returns
    -------
    cluster, client
    """
    gateway = Gateway(
     "http://web-public-l2lhub-prod-dask-gateway",
     proxy_address="tls://scheduler-public-l2lhub-prod-dask-gateway:8786")
    
    cluster = gateway.new_cluster(image=os.environ["JUPYTER_IMAGE_SPEC"])
    cluster.scale(n_workers)
    client = client = cluster.get_client()
    return cluster, client


def batch_process(func, params, client):
    """ 
    Map `params` onto `func` and submit to a dask kube cluster.
    
    Parameters
    ----------
    func : callable
        For now, accepts a single tuple as input and unpacks that tuple internally. 
        See below in `params`.
    
    params : sequence 
        Has the form [(a1, b1, c1, ...), (a2, b2, c2, ...), ...., (an, bn, cn, ...)], 
        where each tuple is the inputs to one process.
        
    client : a dask Client to an initialized cluster, optional. 
        Defaults to start a new client.
    """
    results = client.map(func, params)
    
    all_done = False 
    pbar = tqdm(total=len(params))
    n_done = 0
    while not all_done:
        n_done_now = sum([r.done() for r in results])
        if n_done_now > n_done:
            pbar.update(n_done_now - n_done)
            n_done = n_done_now

        all_done = n_done == len(params)
    
    exceptions = {}
    outputs = {}
    for ii, rr in enumerate(results): 
        if rr.status == 'error':
            exceptions[ii] = rr.exception()
        else:
            outputs[ii] = rr.result()
            
    return outputs, exceptions

