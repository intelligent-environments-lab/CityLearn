import torch as th
import numpy as np
from types import SimpleNamespace as SN


class EpisodeBatch:

    def __init__(self,
                 scheme,
                 groups,
                 batch_size,
                 max_seq_length,
                 data=None,
                 preprocess=None,
                 device="cpu",
                 out_device=None):
        self.scheme = scheme.copy()
        self.groups = groups
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.preprocess = {} if preprocess is None else preprocess
        self.device = device
        self.out_device = out_device if out_device is not None else device

        if data is not None:
            self.data = data
        else:
            self.data = SN()
            self.data.transition_data = {}
            self.data.episode_data = {}
            self._setup_data(self.scheme, self.groups, batch_size, max_seq_length, self.preprocess)

    def _setup_data(self, scheme, groups, batch_size, max_seq_length, preprocess):
        if preprocess is not None:
            for k in preprocess:
                assert k in scheme
                new_k = preprocess[k][0]
                transforms = preprocess[k][1]

                vshape = self.scheme[k]["vshape"]
                dtype = self.scheme[k]["dtype"]
                for transform in transforms:
                    vshape, dtype = transform.infer_output_info(vshape, dtype)

                self.scheme[new_k] = {
                    "vshape": vshape,
                    "dtype": dtype
                }
                if "group" in self.scheme[k]:
                    self.scheme[new_k]["group"] = self.scheme[k]["group"]
                if "episode_const" in self.scheme[k]:
                    self.scheme[new_k]["episode_const"] = self.scheme[k]["episode_const"]

        assert "filled" not in scheme, '"filled" is a reserved key for masking.'
        scheme.update({
            "filled": {"vshape": (1,), "dtype": th.long},
        })

        for field_key, field_info in scheme.items():
            assert "vshape" in field_info, "Scheme must define vshape for {}".format(field_key)
            vshape = field_info["vshape"]
            episode_const = field_info.get("episode_const", False)
            group = field_info.get("group", None)
            dtype = field_info.get("dtype", th.float32)

            if isinstance(vshape, int):
                vshape = (vshape,)

            if group:
                assert group in groups, "Group {} must have its number of members defined in _groups_".format(group)
                shape = (groups[group], *vshape)
            else:
                shape = vshape

            if episode_const:
                self.data.episode_data[field_key] = th.zeros((batch_size, *shape), dtype=dtype, device=self.device)
            else:
                self.data.transition_data[field_key] = th.zeros((batch_size, max_seq_length, *shape), dtype=dtype, device=self.device)

    def extend(self, scheme, groups=None):
        self._setup_data(scheme, self.groups if groups is None else groups, self.batch_size, self.max_seq_length)

    def to(self, device):
        for k, v in self.data.transition_data.items():
            self.data.transition_data[k] = v.to(device)
        for k, v in self.data.episode_data.items():
            self.data.episode_data[k] = v.to(device)
        self.device = device
        return self

    def update(self, data, bs=slice(None), ts=slice(None), mark_filled=True):
        slices = self._parse_slices((bs, ts))
        for k, v in data.items():
            if k in self.data.transition_data:
                target = self.data.transition_data
                if mark_filled:
                    target["filled"][slices] = 1
                    mark_filled = False
                _slices = slices

            elif k in self.data.episode_data:
                target = self.data.episode_data
                _slices = slices[0]
            else:
                raise KeyError("{} not found in transition or episode data".format(k))

            dtype = self.scheme[k].get("dtype", th.float32)
            v = th.tensor(v, dtype=dtype, device=self.device)
            try:
                self._check_safe_view(v, target[k][_slices])
            except Exception as e:
                a = 5
                pass
            target[k][_slices] = v.view_as(target[k][_slices])

            if k in self.preprocess:
                try:
                    new_k = self.preprocess[k][0]
                except Exception as e:
                    a = 5
                    pass
                v = target[k][_slices]
                for transform in self.preprocess[k][1]:
                    v = transform.transform(v)
                    target[new_k][_slices] = v.view_as(target[new_k][_slices])
            pass

    def _check_safe_view(self, v, dest):
        idx = len(v.shape) - 1
        for s in dest.shape[::-1]:
            if v.shape[idx] != s:
                if s != 1:
                    raise ValueError("Unsafe reshape of {} to {}".format(v.shape, dest.shape))
            else:
                idx -= 1

    def __getitem__(self, item):
        if isinstance(item, str):
            if item in self.data.episode_data:
                return self.data.episode_data[item].to(self.device)
            elif item in self.data.transition_data:
                return self.data.transition_data[item].to(self.device)
            else:
                raise ValueError
        elif isinstance(item, tuple) and all([isinstance(it, str) for it in item]):
            new_data = self._new_data_sn()
            for key in item:
                if key in self.data.transition_data:
                    new_data.transition_data[key] = self.data.transition_data[key]
                elif key in self.data.episode_data:
                    new_data.episode_data[key] = self.data.episode_data[key]
                else:
                    raise KeyError("Unrecognised key {}".format(key))

            # Update the scheme to only have the requested keys
            new_scheme = {key: self.scheme[key] for key in item}
            new_groups = {self.scheme[key]["group"]: self.groups[self.scheme[key]["group"]]
                          for key in item if "group" in self.scheme[key]}
            ret = EpisodeBatch(new_scheme, new_groups, self.batch_size, self.max_seq_length, data=new_data, device=self.device)
            return ret.to(self.device)
        else:
            item = self._parse_slices(item)
            new_data = self._new_data_sn()
            for k, v in self.data.transition_data.items():
                new_data.transition_data[k] = v[item]
            for k, v in self.data.episode_data.items():
                new_data.episode_data[k] = v[item[0]]

            ret_bs = self._get_num_items(item[0], self.batch_size)
            ret_max_t = self._get_num_items(item[1], self.max_seq_length)

            ret = EpisodeBatch(self.scheme, self.groups, ret_bs, ret_max_t, data=new_data, device=self.device)
            return ret.to(self.device)

    def _get_num_items(self, indexing_item, max_size):
        if isinstance(indexing_item, list) or isinstance(indexing_item, np.ndarray):
            return len(indexing_item)
        elif isinstance(indexing_item, slice):
            _range = indexing_item.indices(max_size)
            return 1 + (_range[1] - _range[0] - 1)//_range[2]

    def _new_data_sn(self):
        new_data = SN()
        new_data.transition_data = {}
        new_data.episode_data = {}
        return new_data

    def _parse_slices(self, items):
        parsed = []
        # Only batch slice given, add full time slice
        if (isinstance(items, slice)  # slice a:b
            or isinstance(items, int)  # int i
            or (isinstance(items, (list, np.ndarray, th.LongTensor, th.cuda.LongTensor)))  # [a,b,c]
            ):
            items = (items, slice(None))

        # Need the time indexing to be contiguous
        if isinstance(items[1], list):
            raise IndexError("Indexing across Time must be contiguous")

        for item in items:
            if isinstance(item, int):
                # Convert single indices to slices
                parsed.append(slice(item, item+1))
            else:
                # Leave slices and lists as is
                parsed.append(item)
        return parsed

    def max_t_filled(self):
        return th.sum(self.data.transition_data["filled"], 1).max(0)[0]

    def __repr__(self):
        return "EpisodeBatch. Batch Size:{} Max_seq_len:{} Keys:{} Groups:{}".format(self.batch_size,
                                                                                     self.max_seq_length,
                                                                                     self.scheme.keys(),
                                                                                     self.groups.keys())

    def share(self):
        {v.share_memory_() for _, v in self.data.transition_data.items()}
        {v.share_memory_() for _, v in self.data.episode_data.items()}
        return self

    def clone(self):
        self.data.transition_data = {k:v.clone() for k, v in self.data.transition_data.items()}
        self.data.episode_data = {k: v.clone() for k, v in self.data.episode_data.items()}
        return self

    def to_df(self):
        # convert to pandas dataframe so can be viewed easily in pycharm
        import pandas as pd

        # transition data
        cols = list(self.data.transition_data.keys())
        cln_cols = [] # clean for agent dimension
        cln_data = []
        for col in cols:
            if self.data.transition_data[col].dim() == 4:
                n_agents = self.data.transition_data[col].shape[-2]
                for aid in range(n_agents):
                    cln_cols.append(col + "__agent{}".format(aid))
                    cln_data.append(self.data.transition_data[col][:,:,aid,:].cpu().numpy())
            else:
                cln_cols.append(col)
                cln_data.append(self.data.transition_data[col].cpu().numpy())

        batch_size = self.data.transition_data[cols[0]].shape[0]
        seq_len = self.data.transition_data[cols[0]].shape[1]
        transition_pds = []
        for b in range(batch_size):
            pds = pd.DataFrame(columns=cln_cols,
                              data=[[cln_data[j][b, t, :][0] if len(cln_data[j][b, t, :]) == 1 else cln_data[j][b, t, :] for j, _ in enumerate(cln_cols)] for t in range(seq_len)])
            transition_pds.append(pds)

        # episode data
        episode_pds = []
        cols = list(self.data.episode_data.keys())
        if self.data.episode_data != {}:
            for b in range(self.data.episode_data[cols[0]].shape[0]):
                pd = pd.DataFrame(columns=cln_cols,
                                  data=[[cln_data[j][b, :] for j, _ in enumerate(cln_cols)] for t in range(1)])
                episode_pds.append(pd)
        return transition_pds, episode_pds

# import blosc
class CompressibleBatchTensor():

    def __init__(self, batch_size, shape, dtype, device, out_device, chunk_size=10, algo="zstd"):
        assert batch_size % chunk_size==0, "batch_size must be multiple of chunk size!"
        self._storage = {_i:None for _i in range(batch_size // chunk_size)}
        self.chunk_size = chunk_size
        self.algo = algo
        self.batch_size = batch_size
        self.device = device
        self.dtype = dtype
        self.np_dtype = th.Tensor(1).type(self.dtype).numpy().dtype
        self.shape = shape
        self.out_device = out_device
        pass

    def __getitem__(self, item):
        batch_idx = item[0]
        other_idx = item[1:]

        if isinstance(batch_idx, slice):
            batch_idxs = list(range(0 if batch_idx.start is None else batch_idx.start,
                                    self.batch_size if batch_idx.stop is None else batch_idx.stop,
                                    1 if batch_idx.step is None else batch_idx.step))
        elif isinstance(batch_idx, list):
            batch_idxs = batch_idx

        else:
            batch_idxs = [batch_idx]

        # cluster batch_idxs by chunk
        chunk_dict = {}
        for _a, idx in enumerate(batch_idxs):
            chunk_id = idx // self.chunk_size
            if not (chunk_id in chunk_dict):
                chunk_dict[chunk_id] = []
            id_in_chunk = idx % self.chunk_size
            chunk_dict[chunk_id].append((id_in_chunk, _a))

        tmp_list = []
        for chunk_id in chunk_dict.keys():
            if self._storage[chunk_id] is None:
                for _, _a in chunk_dict[chunk_id]:
                    tmp_list.append((th.zeros(self.shape,
                                             dtype=self.dtype,
                                             device=self.out_device), _a))
            else:
                # decompress and read out
                tmp = self._decompress(self._storage[chunk_id], shape=(self.chunk_size, *self.shape))
                for in_chunk_idx, _a in chunk_dict[chunk_id]:
                    tmp_list.append((tmp.__getitem__((in_chunk_idx, *other_idx)), _a))

        # re-order elements
        tmp_list.sort(key=lambda x: x[1])
        rtn_item = th.stack([a[0] for a in tmp_list], 0)
        return rtn_item.to(device=self.out_device)


    def _decompress(self, compressed_tensor, shape):
        decompressed_string = blosc.decompress(compressed_tensor, self.np_dtype)
        np_arr = np.fromstring(bytes(decompressed_string), dtype=self.np_dtype).reshape(shape)
        th_tensor = th.from_numpy(np_arr)
        return th_tensor

    def _compress(self, tensor):
        np_tensor = tensor.cpu().numpy()
        compressed_tensor = blosc.compress(np_tensor.tostring(), typesize=np_tensor.itemsize, cname=self.algo)
        return compressed_tensor

    def __setitem__(self, item, val):
        batch_idx = item[0]
        other_idx = item[1:]
        if isinstance(batch_idx, slice):
            batch_idxs = list(range(0 if batch_idx.start is None else batch_idx.start,
                                    self.batch_size if batch_idx.stop is None else batch_idx.stop,
                                    1 if batch_idx.step is None else batch_idx.step))
        else:
            batch_idxs = [batch_idx]

        assert list(sorted(batch_idxs)) == batch_idxs, "batch_idxs have to be in order!"

        # cluster batch_idxs by chunk
        chunk_dict = {}
        for _a, idx in enumerate(batch_idxs):
            chunk_id = idx // self.chunk_size
            if not (chunk_id in chunk_dict):
                chunk_dict[chunk_id] = {}
            id_in_chunk = idx % self.chunk_size
            chunk_dict[chunk_id][id_in_chunk] = _a

        for chunk_id in chunk_dict.keys():
            if self._storage[chunk_id] is None:
                tmp_tensor = th.zeros((self.chunk_size, *self.shape),
                                dtype=self.dtype,
                                device=self.out_device)
            else:

                # decompress and read out
                tmp_tensor = self._decompress(self._storage[chunk_id], shape=(self.chunk_size, *self.shape))
            for in_chunk_id, val_idx in chunk_dict[chunk_id].items():
                tmp_tensor.__setitem__([in_chunk_id, *other_idx], val[val_idx])
            # store tmp_tensor back
            self._storage[chunk_id] = self._compress(tmp_tensor)
        pass

    def get_compression_stats(self):
        stats = {}

        # calculate how many entries are actually filled in the buffer
        nonempty_chunks = [i for i, (_, _x) in enumerate(self._storage.items()) if _x is not None]
        stats["fill_level"] = float(len(nonempty_chunks)) / float(len(self._storage.keys()))

        # calculate compression ratio
        from itertools import product
        chunk_compression_ratios = [len(_x) / (np.asscalar(np.prod(np.array(self.shape))) * self.chunk_size * self.np_dtype.itemsize) for
                                    k, _x in self._storage.items() if _x is not None]
        stats["compression_ratio"] = np.asscalar(np.array(chunk_compression_ratios).mean())

        stats["predicted_full_size_compressed"] = stats["compression_ratio"] * self.chunk_size * len(self._storage.keys()) * np.asscalar(np.prod(np.array(self.shape))*self.np_dtype.itemsize)
        stats["predicted_full_size_uncompressed"] = self.chunk_size * len(self._storage.keys()) * np.asscalar(np.prod(np.array(self.shape))*self.np_dtype.itemsize)
        return stats

    pass

class CompressibleEpisodeBatch(EpisodeBatch):

    def __init__(self, scheme, groups, batch_size,
                  max_seq_length, data, preprocess,
                  device,
                  out_device,
                 chunk_size=10,
                 algo="zstd"):
        self.out_device = out_device
        self.chunk_size = chunk_size
        self.algo = algo
        EpisodeBatch.__init__(self,
                              scheme=scheme,
                              groups=groups,
                              batch_size=batch_size,
                              max_seq_length=max_seq_length,
                              data=None,
                              preprocess=preprocess,
                              device=device)
        pass

    def _setup_data(self, scheme, groups, batch_size, max_seq_length, preprocess):
        super()._setup_data(scheme, groups, batch_size=1, max_seq_length=1, preprocess=preprocess)

        # assert "filled" not in scheme, '"filled" is a reserved key for masking.'
        scheme.update({
            "filled": {"vshape": (1,), "dtype": th.long},
        })

        for field_key, field_info in scheme.items():
            assert "vshape" in field_info, "Scheme must define vshape for {}".format(field_key)
            vshape = field_info["vshape"]
            episode_const = field_info.get("episode_const", False)
            group = field_info.get("group", None)
            dtype = field_info.get("dtype", th.float32)

            if isinstance(vshape, int):
                vshape = (vshape,)

            if group:
                assert group in groups, "Group {} must have its number of members defined in _groups_".format(group)
                shape = (groups[group], *vshape)
            else:
                shape = vshape

            if episode_const:
                self.data.episode_data[field_key] = CompressibleBatchTensor(batch_size=batch_size,
                                                                            shape=shape,
                                                                            dtype=dtype,
                                                                            device=self.device,
                                                                            out_device=self.out_device,
                                                                            chunk_size=self.chunk_size,
                                                                            algo=self.algo)
            else:
                self.data.transition_data[field_key] = CompressibleBatchTensor(batch_size=batch_size,
                                                                               shape=(max_seq_length, *shape),
                                                                               dtype=dtype,
                                                                               device=self.device,
                                                                               out_device=self.out_device,
                                                                               chunk_size=self.chunk_size,
                                                                               algo=self.algo)

    def get_compression_stats(self):
        stats = {}

        stats_list_ep = {}
        for k, v in self.data.episode_data.items():
            stats_list_ep[k] = v.get_compression_stats()

        stats_list_trans = {}
        for k, v in self.data.transition_data.items():
            stats_list_trans[k] = v.get_compression_stats()

        stats["fill_level"] = np.asscalar(np.mean([ v["fill_level"] for _, v in stats_list_trans.items() ]))
        # stats["compression_ratio"] = np.asscalar(np.mean([v["compression_ratio"] for _, v in stats_list_trans.items()]))
        stats["compression_ratio"] = np.asscalar(np.sum([v["predicted_full_size_compressed"] for _, v in stats_list_trans.items()]))\
                                     / np.asscalar(np.sum([v["predicted_full_size_uncompressed"] for _, v in stats_list_trans.items()]))
        stats["predicted_full_size_compressed"] = np.asscalar(np.sum([v["predicted_full_size_compressed"] for _, v in stats_list_trans.items()]))
        stats["predicted_full_size_uncompressed"] = np.asscalar(
            np.sum([v["predicted_full_size_uncompressed"] for _, v in stats_list_trans.items()]))
        return stats

class ReplayBuffer(EpisodeBatch):

    def __init__(self, scheme, groups, buffer_size, max_seq_length, preprocess=None, device="cpu", out_device=None):
        super(ReplayBuffer, self).__init__(scheme, groups, buffer_size, max_seq_length, preprocess=preprocess, device=device, out_device=out_device)
        self.buffer_size = buffer_size  # same as self.batch_size but more explicit
        self.buffer_index = 0
        self.episodes_in_buffer = 0
        self.out_device = out_device if out_device is not None else device

    def insert_episode_batch(self, ep_batch):
        if self.buffer_index + ep_batch.batch_size <= self.buffer_size:
            self.update(ep_batch.data.transition_data,
                        slice(self.buffer_index, self.buffer_index + ep_batch.batch_size),
                        slice(0, ep_batch.max_seq_length),
                        mark_filled=False)
            self.update(ep_batch.data.episode_data,
                        slice(self.buffer_index, self.buffer_index + ep_batch.batch_size))
            self.buffer_index = (self.buffer_index + ep_batch.batch_size)
            self.episodes_in_buffer = max(self.episodes_in_buffer, self.buffer_index)
            self.buffer_index = self.buffer_index % self.buffer_size
            assert self.buffer_index < self.buffer_size
        else:
            buffer_left = self.buffer_size - self.buffer_index
            self.insert_episode_batch(ep_batch[0:buffer_left, :].to(self.device))
            self.insert_episode_batch(ep_batch[buffer_left:, :].to(self.device))

    def can_sample(self, batch_size):
        return self.episodes_in_buffer >= batch_size

    def sample(self, batch_size):
        assert self.can_sample(batch_size)
        if self.episodes_in_buffer == batch_size:
            out_batch = self[:batch_size].clone().share().to(self.out_device)
            return out_batch.to(self.out_device)
        else:
            # Uniform sampling only atm
            ep_ids = np.random.choice(self.episodes_in_buffer, batch_size, replace=False).tolist()
            out_batch = self[ep_ids].clone().share().to(self.out_device)
            return out_batch.to(self.out_device)

    def __repr__(self):
        return "ReplayBuffer. {}/{} episodes.q Keys:{} Groups:{}".format(self.episodes_in_buffer,
                                                                         self.buffer_size,
                                                                         self.scheme.keys(),
                                                                         self.groups.keys())


class CompressibleReplayBuffer(CompressibleEpisodeBatch, ReplayBuffer):

    def __init__(self, scheme, groups, buffer_size, max_seq_length, preprocess=None, device="cpu", out_device="cpu", compress=True, chunk_size=10, algo="zstd"):

        CompressibleEpisodeBatch.__init__(self, scheme=scheme, groups=groups, batch_size=buffer_size,
                                          max_seq_length=max_seq_length, data=None, preprocess=preprocess,
                                          device=device,
                                          out_device=out_device,
                                          chunk_size=chunk_size,
                                          algo=algo)
        self.buffer_size = buffer_size  # same as self.batch_size but more explicit
        self.buffer_index = 0
        self.episodes_in_buffer = 0
        self.out_device = out_device
        self.chunk_size = chunk_size
        self.algo = algo
        pass

    pass