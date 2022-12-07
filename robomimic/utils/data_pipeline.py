from copy import deepcopy
import json

from torch.utils.data import default_collate
import numpy as np

from torchdata.datapipes.iter import IterableWrapper, FileOpener, TarArchiveLoader
from torchdata.datapipes.iter import Demultiplexer, Mapper, InMemoryCacheHolder
from torchdata.datapipes.iter import Shuffler, Batcher, Collator, JsonParser
from torchdata.datapipes.map import IterToMapConverter

import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.log_utils as LogUtils


DP_STATE = {
    "obs_keys": [],
    "dataset_keys": [],
    "demos": [],
    "demos_dp": None,
    "n_demos": 0,
    "load_next_obs": True,
    "n_frame_stack": 1,  # no frame stacking
    "seq_length": 1,
    "pad_frame_stack": True,
    "pad_seq_length": True,  # pad last obs per trajectory to ensure all sequences are sampled
    "get_pad_mask": False,
    "goal_mode": None,
    "cache_mode": None,
    "normalize_obs": False,
    "normalization_stats": {},
    "attributes": {},
    "index_to_demo_id": {},
    "demo_id_to_start_indices": {},
    "demo_id_to_demo_length": {},
    "total_num_sequences": 0
}

def load_demo_info(demos):
    """
    Loads demo information into a cached dictionary. Should not be called directly,
    called by `create_data_pipeline`.

    Args:
        config (BCConfig): config for the training run
        demos (MapperIterDataPipe): 
        mask (dict): Mask keys
    """

    # get, sort, set demo keys
    keys = [demo[0] for demo in demos]
    inds = np.argsort([int(elem[5:]) for elem in keys])
    DP_STATE["demos"] = [keys[i] for i in inds]
    DP_STATE["n_demos"] = len(DP_STATE["demos"])

    demos = demos.to_map_datapipe()
    
    print("SequenceDataPipeline: Loading dataset info...")
    for demo in LogUtils.custom_tqdm(DP_STATE["demos"]):
        demo_id = demo
        demo_len = DP_STATE["attributes"][demo_id]["num_samples"]
        DP_STATE["demo_id_to_start_indices"][demo_id] = DP_STATE["total_num_sequences"]
        DP_STATE["demo_id_to_demo_length"][demo_id] = demo_len

        num_seq = demo_len

        if not DP_STATE["pad_frame_stack"]:
            num_seq -= DP_STATE["n_frame_stack"] - 1
        if not DP_STATE["pad_seq_length"]:
            num_seq -= DP_STATE["seq_length"] - 1

        if DP_STATE["pad_seq_length"]:
            assert demo_len >= 1
            num_seq = max(num_seq, 1)
        else:
            assert num_seq >= 1

        for _ in range(num_seq):
            DP_STATE["index_to_demo_id"][DP_STATE["total_num_sequences"]] = demo_id
            DP_STATE["total_num_sequences"] += 1

def get_obs_normalization_stats():
    """
    Returns dictionary of mean and std for each observation key if using
    observation normalization, otherwise None.

    Returns:
        obs_normalization_stats (dict): a dictionary for observation normalization.
        This maps observation keys to dicts with a "mean" and "std" of shape (1, ...)
        where ... is the default shape for the observation
    """
    assert DP_STATE["normalization_stats"], "Not using observation normalization!"
    return deepcopy(DP_STATE["normalization_stats"])

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
    assert DP_STATE["obs_keys"] is not None and isinstance(DP_STATE["obs_keys"], tuple), "OBS_KEYS has not been initialized."
    print("SequenceDataPipeline: normalizing observations...")
    merged_stats = {}
    for i, demo in LogUtils.custom_tqdm(enumerate(demos), total=DP_STATE["n_demos"]):
        obs_traj = {k: demo[1]['obs'][k] for k in DP_STATE["obs_keys"]}
        obs_traj = ObsUtils.process_obs_dict(obs_traj)
        if i == 0:
            merged_stats = _compute_traj_stats(obs_traj)
        else:
            traj_stats = _compute_traj_stats(obs_traj)
            merged_stats = _aggregate_traj_stats(merged_stats, traj_stats)

    obs_normalization_stats = { k : {} for k in merged_stats }
    for k in merged_stats:
        # note we add a small tolerance of 1e-3 for std
        obs_normalization_stats[k]["mean"] = merged_stats[k]["mean"]
        obs_normalization_stats[k]["std"] = np.sqrt(merged_stats[k]["sqdiff"] / merged_stats[k]["n"]) + 1e-3
    DP_STATE["normalization_stats"] = obs_normalization_stats

def _parse_config(config):
    DP_STATE["dataset_keys"] = config.train.dataset_keys
    assert config.train.seq_length >= 1
    DP_STATE["seq_length"] = config.train.seq_length
    DP_STATE["goal_mode"] = config.train.goal_mode
    if DP_STATE["goal_mode"] is not None:
        assert DP_STATE["goal_mode"] in ["last"]
    if not DP_STATE["load_next_obs"]:
        assert DP_STATE["goal_mode"] != "last"

    DP_STATE["cache_mode"] = config.train.hdf5_cache_mode
    DP_STATE["normalize_obs"] = config.train.hdf5_normalize_obs

def _unpack_data(data):
    data_dict = dict(np.load(data[1], allow_pickle=True))
    # TODO: verify this works
    data_key = data[0].split('/')[-1].split('.')[0]
    out = {}
    out['obs'] = {}
    out['next_obs'] = {}
    for k, v in data_dict.items():
        if k.startswith('obs_'):
            if k[4:] in DP_STATE['obs_keys']:
                out['obs'][k[4:]] = v.astype('float32')
        elif k.startswith('next_obs_'):
            if k[9:] in DP_STATE['obs_keys']:
                out['next_obs'][k[9:]] = v.astype('float32')
        else:
            out[k] = v
    return (data_key, out)

def _split_data(data):
    if "demo" in data[0] and data[0].endswith('npz'):
        return 0
    if data[0].endswith('mask.npz'):
        return 1
    if data[0].endswith('demo_attrs.json'):
        return 2
    return 3


def _get_sequence_from_demo(demo_id, index_in_demo, keys, num_frames_to_stack=0, seq_length=1, obs=None):
    assert num_frames_to_stack >= 0
    assert seq_length >= 1
    assert obs in [None, "obs", "next_obs"]

    demo_length = DP_STATE["demo_id_to_demo_length"][demo_id]
    assert index_in_demo < demo_length

    # determine begin and end of sequence
    seq_begin_index = max(0, index_in_demo - num_frames_to_stack)
    seq_end_index = min(demo_length, index_in_demo + seq_length)

    # determine sequence padding
    seq_begin_pad = max(0, num_frames_to_stack - index_in_demo)  # pad for frame stacking
    seq_end_pad = max(0, index_in_demo + seq_length - demo_length)  # pad for sequence length

    if not DP_STATE["pad_frame_stack"]:
        assert seq_begin_pad == 0
    if not DP_STATE["pad_seq_length"]:
        assert seq_end_pad == 0

    seq = dict()
    for k in keys:
        if obs is not None:
            data = DP_STATE["demos_dp"][demo_id][obs][k]
        else:
            data = DP_STATE["demos_dp"][demo_id][k]
        seq[k] = data[seq_begin_index: seq_end_index].astype("float32")
    
    seq = TensorUtils.pad_sequence(seq, padding=(seq_begin_pad, seq_end_pad), pad_same=True)
    if DP_STATE["get_pad_mask"]:
        pad_mask = np.array([0] * seq_begin_pad + [1] * (seq_end_index - seq_begin_index) + [0] * seq_end_pad)
        pad_mask = pad_mask[:, None].astype(np.bool)
        seq["pad_mask"] = pad_mask

    return seq

def get_item(data):
    
    index = data[0]
    demo_id = data[1]
    demo_start_index = DP_STATE["demo_id_to_start_indices"][demo_id]
    demo_length = DP_STATE["demo_id_to_demo_length"][demo_id]
    
    # start at offset index if not padding for frame stacking
    demo_index_offset = 0 if DP_STATE["pad_frame_stack"] else (DP_STATE["n_frame_stack"] - 1)
    index_in_demo = index - demo_start_index + demo_index_offset

    demo_length_offset = 0 if DP_STATE["pad_seq_length"] else (DP_STATE["seq_length"] - 1)
    end_index_in_demo = demo_length - demo_length_offset

    goal_index = None 
    if DP_STATE["goal_mode"] == "last":
        goal_index = end_index_in_demo - 1

    out = _get_sequence_from_demo(
            demo_id, 
            index_in_demo=index_in_demo, 
            keys=DP_STATE["dataset_keys"], 
            num_frames_to_stack=DP_STATE["n_frame_stack"] - 1,
            seq_length=DP_STATE["seq_length"]
        )

    out["obs"] = _get_sequence_from_demo(
            demo_id,
            index_in_demo=index_in_demo,
            keys=DP_STATE["obs_keys"],
            num_frames_to_stack=DP_STATE["n_frame_stack"] - 1,
            seq_length=DP_STATE["seq_length"],
            obs="obs"
    )

    if DP_STATE["normalize_obs"]:
        out["obs"] = ObsUtils.normalize_obs(out["obs"], obs_normalization_stats=DP_STATE["normalization_stats"])

    if DP_STATE["load_next_obs"]:
        out["next_obs"] = _get_sequence_from_demo(
            demo_id,
            index_in_demo=index_in_demo,
            keys=DP_STATE["obs_keys"],
            num_frames_to_stack=DP_STATE["n_frame_stack"] - 1,
            seq_length=DP_STATE["seq_length"],
            obs="next_obs"
        )

        if DP_STATE["normalize_obs"]:
            out["next_obs"] = ObsUtils.normalize_obs(out["obs"], obs_normalization_stats=DP_STATE["normalization_stats"])
    
    if goal_index is not None:
        goal = _get_sequence_from_demo(
            demo_id,
            index_in_demo=index_in_demo,
            keys=DP_STATE["obs_keys"],
            num_frames_to_stack=DP_STATE["n_frame_stack"] - 1,
            seq_length=DP_STATE["seq_length"],
            obs="next_obs"
        )

        if DP_STATE["normalize_obs"]:
            goal = ObsUtils.normalize_obs(goal, obs_normalization_stats=DP_STATE["normalization_stats"])
        out["goal_obs"] = {k : goal[k][0] for k in goal}

    return out

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
    DP_STATE["obs_keys"] = tuple(all_obs_keys)
    _parse_config(config)
    batch_size = config.train.batch_size

    # open pipeline
    dp = IterableWrapper([path])
    dp = FileOpener(dp, 'b')
    dp = TarArchiveLoader(dp)
    demos_dp, mask_dp, demo_attrs, _ = Demultiplexer(dp, num_instances=4, classifier_fn=_split_data)

    # preproc pipeline
    demo_attrs = JsonParser(demo_attrs)
    mask = dict(np.load(next(iter(mask_dp))[1]))
    DP_STATE["attributes"] = next(iter(demo_attrs))[1]

    demos_dp = Mapper(demos_dp, _unpack_data)
    # TODO: split train/validation sets

    load_demo_info(demos_dp)

    if DP_STATE["normalize_obs"]:
        normalize_obs(demos_dp)
    
    # demos_dp = InMemoryCacheHolder(demos_dp)  # TODO: add size param to config
    DP_STATE["demos_dp"] = demos_dp.to_map_datapipe()

    output = IterableWrapper([(k, v) for k, v in DP_STATE["index_to_demo_id"].items()])
    output = Shuffler(output)
    output = Mapper(output, get_item)
    # demos_dp = Mapper(demos_dp, get_item)

    # output pipeline
    output = Batcher(output, batch_size=batch_size, drop_last=True)
    output = Collator(output, collate_fn=default_collate)

    return output


