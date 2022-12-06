from torch.utils.data import default_collate
import numpy as np

from torchdata.datapipes.iter import IterableWrapper, FileOpener, TarArchiveLoader
from torchdata.datapipes.iter import Demultiplexer, Mapper, InMemoryCacheHolder
from torchdata.datapipes.iter import Shuffler, Batcher, Collator

DATASET_KEYS = None
OBS_NORMALIZATIONS_STATS = None
OBS_KEYS = None
DATASET_KEYS = None
GOAL_MODE = None
PAD_SEQ_LENGTH = 1
PAD_FRAME_STACK = None
GET_PAD_MASK = None

def _parse_config(config):
    pass

def normalize_obs(demos):

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
    
    assert OBS_KEYS is not None and isinstance(OBS_KEYS, list), "OBS_KEYS has not been initialized."
    obs_traj = {k: demos[0][k] for k in OBS_KEYS}
    # obs_traj = {k: self.hdf5_file["data/{}/obs/{}".format(ep, k)][()].astype('float32') for k in self.obs_keys}
    obs_traj = ObsUtils.process_obs_dict(obs_traj)
    merged_stats = _compute_traj_stats(obs_traj)
    print("SequenceDataPipeline: normalizing observations...")
    for ep in LogUtils.custom_tqdm(demos[1:]):
        obs_traj = {k: demo[k] for k in OBS_KEYS}
        obs_traj = ObsUtils.process_obs_dict(obs_traj)
        traj_stats = _compute_traj_stats(obs_traj)
        merged_stats = _aggregate_traj_stats(merged_stats, traj_stats)

    obs_normalization_stats = { k : {} for k in merged_stats }
    for k in merged_stats:
        # note we add a small tolerance of 1e-3 for std
        obs_normalization_stats[k]["mean"] = merged_stats[k]["mean"]
        obs_normalization_stats[k]["std"] = np.sqrt(merged_stats[k]["sqdiff"] / merged_stats[k]["n"]) + 1e-3
    return obs_normalization_stats

def _unpack_data(data):
    data_dict = dict(np.load(data[1], allow_pickle=True))
    out = {}
    out['obs'] = {}
    out['next_obs'] = {}
    for k, v in data_dict.items():
        if k.startswith('obs_'):
            out['obs'][k[4:]] = v.astype('float32')
        elif k.startswith('next_obs_'):
            out['next_obs'][k[9:]] = v.astype('float32')
        else:
            out[k] = v
    return out

def _split_data(data):
    if "demo" in data[0]:
        return 0
    if data[0].endswith('mask.npz'):
        return 1
    return 2

def create_data_pipeline(path, config):

    _parse_config(config)
    # open pipeline
    dp = IterableWrapper([path])
    dp = FileOpener(dp, 'b')
    dp = TarArchiveLoader(dp)
    demos_dp, _, _ = Demultiplexer(dp, num_instances=3, classifier_fn=_split_data)

    # preproc pipeline
    demos_dp = Mapper(demos_dp, _unpack_data)
    demos_dp = InMemoryCacheHolder(demos_dp)  # TODO: add size param to config
    
    # output pipeline
    output = Shuffler(demos_dp)
    output = Batcher(output, batch_size=64, drop_last=True)
    output = Collator(output, collate_fn=default_collate)

    return output

