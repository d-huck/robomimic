from copy import deepcopy
import json

from torch.utils.data import default_collate
import numpy as np

from torchdata.datapipes.iter import IterableWrapper, FileOpener, TarArchiveLoader
from torchdata.datapipes.iter import Demultiplexer, Mapper, InMemoryCacheHolder
from torchdata.datapipes.iter import Shuffler, Batcher, Collator

import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.log_utils as LogUtils

DP_CONFIG = {
    "obs_keys": [],
    "dataset_keys": [],
    "load_next_obs": True,
    "frame_stack": 1,  # no frame stacking
    "seq_length": 1,
    "pad_frame_stack": True,
    "pad_seq_length": True,  # pad last obs per trajectory to ensure all sequences are sampled
    "get_pad_mask": False,
    "goal_mode": None,
    "cache_mode": None,
    "normalize_obs": False,
    "normalization_stats": {}
}

DP_INFO = {
    "attributes": {},
    "index_to_demo_id": {},
    "demo_id_to_start_indices": {},
    "demo_id_to_demo_length": {}
}

def load_demo_info(demos, filter_by_attribute=None):
    pass

def get_obs_normalization_stats():
    """
    Returns dictionary of mean and std for each observation key if using
    observation normalization, otherwise None.

    Returns:
        obs_normalization_stats (dict): a dictionary for observation normalization.
        This maps observation keys to dicts with a "mean" and "std" of shape (1, ...)
        where ... is the default shape for the observation
    """
    assert DP_CONFIG["normalization_stats"], "Not using observation normalization!"
    return deepcopy(DP_CONFIG["normalization_stats"])

def normalize_obs(demos):
    """
    Computes a dataset wide mean and standard deviation for the observations
    (per dimension and per obs key) and returns it

    Args:
        demos (list): List of demos to calculate statistics over. Supplied from
        DataPipeline.
    """    

    def _compute_traj_stats(traj_obs_dict):
        """
        Helper function to compute statistics over a single trajectory of observations.
        """
        traj_stats = { k : {} for k in traj_obs_dict }
        for k in traj_obs_dict:
            traj_stats[k]["n"] = traj_obs_dict[k].shape[0]
            traj_stats[k]["mean"] = traj_obs_dict[k].mean(axis=0, keepdims=True) # [1, ...]
            traj_stats[k]["sqdiff"] = ((traj_obs_dict[k] - traj_stats[k]["mean"]) ** 2).sum(axis=0, keepdims=True) # [1, ...]
        return traj_stats

    def _aggregate_traj_stats(traj_stats_a, traj_stats_b):
            """
            Helper function to aggregate trajectory statistics.
            See https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
            for more information.
            """
            merged_stats = {}
            for k in traj_stats_a:
                n_a, avg_a, M2_a = traj_stats_a[k]["n"], traj_stats_a[k]["mean"], traj_stats_a[k]["sqdiff"]
                n_b, avg_b, M2_b = traj_stats_b[k]["n"], traj_stats_b[k]["mean"], traj_stats_b[k]["sqdiff"]
                n = n_a + n_b
                mean = (n_a * avg_a + n_b * avg_b) / n
                delta = (avg_b - avg_a)
                M2 = M2_a + M2_b + (delta ** 2) * (n_a * n_b) / n
                merged_stats[k] = dict(n=n, mean=mean, sqdiff=M2)
            return merged_stats

    # Run through all trajectories. For each one, compute minimal observation statistics, and then aggregate
    # with the previous statistics.
    
    assert DP_CONFIG["obs_keys"] is not None and isinstance(DP_CONFIG["obs_keys"], list), "OBS_KEYS has not been initialized."
    obs_traj = {k: demos[0][k] for k in DP_CONFIG["obs_keys"]}
    # obs_traj = {k: self.hdf5_file["data/{}/obs/{}".format(ep, k)][()].astype('float32') for k in self.obs_keys}
    obs_traj = ObsUtils.process_obs_dict(obs_traj)
    merged_stats = _compute_traj_stats(obs_traj)
    print("SequenceDataPipeline: normalizing observations...")
    for demo in LogUtils.custom_tqdm(demos[1:]):
        obs_traj = {k: demo[k] for k in DP_CONFIG["obs_keys"]}
        obs_traj = ObsUtils.process_obs_dict(obs_traj)
        traj_stats = _compute_traj_stats(obs_traj)
        merged_stats = _aggregate_traj_stats(merged_stats, traj_stats)

    obs_normalization_stats = { k : {} for k in merged_stats }
    for k in merged_stats:
        # note we add a small tolerance of 1e-3 for std
        obs_normalization_stats[k]["mean"] = merged_stats[k]["mean"]
        obs_normalization_stats[k]["std"] = np.sqrt(merged_stats[k]["sqdiff"] / merged_stats[k]["n"]) + 1e-3
    DP_CONFIG["normalization_stats"] = obs_normalization_stats

def _parse_config(config):
    DP_CONFIG["dataset_keys"] = config.train.dataset_keys
    assert config.train.seq_length >= 1
    DP_CONFIG["seq_length"] = config.train.seq_length
    DP_CONFIG["goal_mode"] = config.train.goal_mode
    if DP_CONFIG["goal_mode"] is not None:
        assert DP_CONFIG["goal_mode"] in ["last"]
    if not DP_CONFIG["load_next_obs"]:
        assert DP_CONFIG["goal_mode"] != "last"

    DP_CONFIG["cache_mode"] = config.train.hdf5_cache_mode
    DP_CONFIG["normalize_obs"] = config.train.hdf5_normalize_obs

def _unpack_data(data):
    data_dict = dict(np.load(data[1], allow_pickle=True))
    # TODO: verify this works
    index = int(data[0].split('/')[-1].split('.')[0].split('_')[-1])
    out = {}
    out['obs'] = {}
    out['next_obs'] = {}
    for k, v in data_dict.items():
        if k.startswith('obs_'):
            if k[4:] in DP_CONFIG['obs_keys']:
                out['obs'][k[4:]] = v.astype('float32')
        elif k.startswith('next_obs_'):
            if k[9:] in DP_CONFIG['obs_keys']:
                out['next_obs'][k[9:]] = v.astype('float32')
        else:
            out[k] = v
    return (int(index), out)

def _split_data(data):
    if "demo" in data[0]:
        return 0
    if data[0].endswith('mask.npz'):
        return 1
    if data[0].endswith('demo_attrs.json')
        return 2
    return 3

def create_data_pipeline(path, config, all_obs_keys):
    """
    Main access for data pipelines. Sets up a dataflow graph for use in data 
    loading using torchdata data pipelines. Still in Beta

    Args:
        path (str): filepath to the dataset
        config (dict): config dictionary 
        all_obs_keys (list): all keys used in observation

    Returns:
        _type_: _description_
    """

    # initial set up
    DP_CONFIG["obs_keys"] = tuple(all_obs_keys)
    _parse_config(config)
    batch_size = config.train.batch_size

    # open pipeline
    dp = IterableWrapper([path])
    dp = FileOpener(dp, 'b')
    dp = TarArchiveLoader(dp)
    demos_dp, mask_dp, demo_attrs, _ = Demultiplexer(dp, num_instances=4, classifier_fn=_split_data)

    # preproc pipeline
    with open(demo_attrs) as f:
        DP_INFO["attributes"] = json.loads(f)
    demos_dp = Mapper(demos_dp, _unpack_data)
    # TODO: split train/validation sets

    if DP_CONFIG["normalize_obs"]:
        normalize_obs(demos_dp)

    demos_dp = InMemoryCacheHolder(demos_dp)  # TODO: add size param to config
    
    # output pipeline
    output = Shuffler(demos_dp)
    output = Batcher(output, batch_size=batch_size, drop_last=True)
    output = Collator(output, collate_fn=default_collate)

    return output


