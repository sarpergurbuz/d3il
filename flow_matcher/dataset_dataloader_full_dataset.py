import os
import glob
import bisect
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class MultiFileDiversePathsDataset(Dataset):
    """
    Treat multiple .npz shard files as one logical dataset.

    Each global sample corresponds to one row in `paths` from one shard file.
    Family metadata is resolved using the file-local `path_family_id`.
    """

    def __init__(self, root_dir, preload_to_ram=True, flag_value=1.0, disable_obs_centers=False):
        self.root_dir = root_dir
        self.preload_to_ram = preload_to_ram
        self.flag_value = flag_value
        self.disable_obs_centers = disable_obs_centers

        self.file_paths = sorted(glob.glob(os.path.join(root_dir, "*.npz")))
        if len(self.file_paths) == 0:
            raise FileNotFoundError(f"No .npz files found in: {root_dir}")

        self.shard_lengths = []
        self.shards = []              # only used if preload_to_ram=True
        self.cumulative_sizes = []

        running_total = 0

        for fp in self.file_paths:
            with np.load(fp, allow_pickle=False) as f:
                n_paths = len(f["paths"])
                self.shard_lengths.append(n_paths)

                if self.preload_to_ram:
                    shard = {k: f[k] for k in f.files}
                    self.shards.append(shard)
                else:
                    self.shards.append(None)

            running_total += n_paths
            self.cumulative_sizes.append(running_total)

        self.cumulative_sizes = np.asarray(self.cumulative_sizes, dtype=np.int64)

    def __len__(self):
        return int(self.cumulative_sizes[-1])

    def _locate_index(self, global_idx):
        """
        Map global index -> (shard_idx, local_idx)
        """
        if global_idx < 0 or global_idx >= len(self):
            raise IndexError(f"Index {global_idx} out of range for dataset of size {len(self)}")

        shard_idx = bisect.bisect_right(self.cumulative_sizes, global_idx)
        prev_cum = 0 if shard_idx == 0 else self.cumulative_sizes[shard_idx - 1]
        local_idx = global_idx - prev_cum
        return shard_idx, local_idx

    def _get_shard(self, shard_idx):
        """
        Return shard data as a dict of numpy arrays.
        If preload_to_ram=False, load on demand.
        """
        if self.preload_to_ram:
            return self.shards[shard_idx]

        fp = self.file_paths[shard_idx]
        with np.load(fp, allow_pickle=False) as f:
            shard = {k: f[k] for k in f.files}
        return shard

    def __getitem__(self, idx):
        shard_idx, local_idx = self._locate_index(idx)
        f = self._get_shard(shard_idx)

        # -------------------------
        # Path-level sample
        # -------------------------
        path = f["paths"][local_idx].astype(np.float32)                           # (104, 2)
        start_specific = f["start_point_specific"][local_idx].astype(np.float32) # (2,)
        end_specific = f["end_point_specific"][local_idx].astype(np.float32)     # (2,)
        fid = int(f["path_family_id"][local_idx])                                 # file-local family id
        path_local_idx = int(f["path_idx_in_family"][local_idx])

        # -------------------------
        # Family-level metadata
        # -------------------------
        obstacle_mask = f["family_obstacle_mask"][fid].astype(np.float32)         # (200, 200), cast once for torch
        obstacle_centers = f["family_obstacle_centers"][fid].astype(np.float32) if not self.disable_obs_centers else None    # (N_obs, 2)
        phi = f["family_phi"][fid].astype(np.float32)                             # (200, 200)
        start_family = f["family_start"][fid].astype(np.float32)                  # (2,)
        end_family = f["family_end"][fid].astype(np.float32)                      # (2,)

        pf_count = int(f["family_potential_field_paths_count"][fid])
        pf_paths = f["family_potential_field_paths"][fid, :pf_count].astype(np.float32)   # (pf_count, 104, 2)

        jerky_count = int(f["family_jerky_paths_count"][fid])
        jerky_paths = f["family_jerky_paths"][fid, :jerky_count].astype(np.float32)        # (jerky_count, 104, 2)

        ilqr_pose = f["ilqr_pose"][local_idx].astype(np.float32)      # (104, 3)
        ilqr_vel = f["ilqr_vel"][local_idx].astype(np.float32)        # (104, 2)
        ilqr_omega = f["ilqr_omega"][local_idx].astype(np.float32)    # (104,)
        ilqr_acc = f["ilqr_acc"][local_idx].astype(np.float32)        # (104, 2)
        ilqr_alpha = f["ilqr_alpha"][local_idx].astype(np.float32)    # (104,)
        ilqr_jerk = f["ilqr_jerk"][local_idx].astype(np.float32)      # (103, 2)
        ilqr_jtheta = f["ilqr_jtheta"][local_idx].astype(np.float32)  # (103,)
        ilqr_dt = f["ilqr_dt"][local_idx].astype(np.float32)         # scalar
        start_specific_ilqr_pose = ilqr_pose[0].astype(np.float32)     # (3,)
        end_specific_ilqr_pose = ilqr_pose[-1].astype(np.float32)      # (3,)

        # Presence flag for CFG, kept as scalar tensor-compatible value
        flag = np.float32(self.flag_value)

        # Return as tuple so your batch[0], batch[1], ... style continues to work
        return (
            path,            # batch[0]
            start_specific,  # batch[1]
            end_specific,    # batch[2]
            obstacle_mask,   # batch[3]
            np.int64(fid),   # batch[4]
            phi,             # batch[5]
            start_family,    # batch[6]
            end_family,      # batch[7]
            pf_paths,        # batch[8]   variable count -> padded in collate_fn
            jerky_paths,     # batch[9]   variable count -> padded in collate_fn
            flag,            # batch[10]
            np.int64(pf_count),     # batch[11]
            np.int64(jerky_count),  # batch[12]
            np.int64(shard_idx),    # batch[13] optional debug
            np.int64(path_local_idx),  # batch[14] optional debug
            np.int64(idx),          # batch[15] optional debug global idx
            obstacle_centers if not self.disable_obs_centers else None,  # batch[16]
            ilqr_pose,          # batch[17]
            ilqr_vel,           # batch[18]
            ilqr_omega,         # batch[19]
            ilqr_acc,           # batch[20]
            ilqr_alpha,         # batch[21]
            ilqr_jerk,          # batch[22]
            ilqr_jtheta,        # batch[23]
            ilqr_dt,            # batch[24]
            start_specific_ilqr_pose,  # batch[25]
            end_specific_ilqr_pose,    # batch[26]
        )


def diverse_paths_collate_fn(batch):
    """
    Collate function that:
    - stacks fixed-size arrays normally
    - pads variable-length family path sets across the batch

    Output order is intentionally matched to the Dataset tuple order.
    """

    paths = []
    start_specifics = []
    end_specifics = []
    obstacle_masks = []
    family_ids = []
    phis = []
    start_families = []
    end_families = []
    pf_paths_list = []
    jerky_paths_list = []
    flags = []
    pf_counts = []
    jerky_counts = []
    shard_idxs = []
    path_local_idxs = []
    global_idxs = []
    obstacle_centers = []
    obstacle_counts = []
    ilqr_pose_list = []
    ilqr_vel_list = []
    ilqr_omega_list = []
    ilqr_acc_list = []
    ilqr_alpha_list = []
    ilqr_jerk_list = []
    ilqr_jtheta_list = []
    ilqr_dt_list = []
    start_specific_ilqr_pose_list = []
    end_specific_ilqr_pose_list = []

    for sample in batch:
        paths.append(torch.from_numpy(sample[0]))                 # (104, 2)
        start_specifics.append(torch.from_numpy(sample[1]))       # (2,)
        end_specifics.append(torch.from_numpy(sample[2]))         # (2,)
        obstacle_masks.append(torch.from_numpy(sample[3]))        # (200, 200)
        family_ids.append(sample[4])
        phis.append(torch.from_numpy(sample[5]))                  # (200, 200)
        start_families.append(torch.from_numpy(sample[6]))        # (2,)
        end_families.append(torch.from_numpy(sample[7]))          # (2,)
        pf_paths_list.append(torch.from_numpy(sample[8]))         # (pf_count, 104, 2)
        jerky_paths_list.append(torch.from_numpy(sample[9]))      # (jerky_count, 104, 2)
        flags.append(sample[10])
        pf_counts.append(sample[11])
        jerky_counts.append(sample[12])
        shard_idxs.append(sample[13])
        path_local_idxs.append(sample[14])
        global_idxs.append(sample[15])
        if sample[16] is not None:
            obstacle_centers.append(torch.from_numpy(sample[16]))      # (N_obs, 2)
            obstacle_counts.append(sample[16].shape[0])
        else:
            obstacle_counts.append(0)
        ilqr_pose_list.append(torch.from_numpy(sample[17]))         # (104, 3)
        ilqr_vel_list.append(torch.from_numpy(sample[18]))          # (104, 2)
        ilqr_omega_list.append(torch.from_numpy(sample[19]))        # (104,)
        ilqr_acc_list.append(torch.from_numpy(sample[20]))          # (104, 2)
        ilqr_alpha_list.append(torch.from_numpy(sample[21]))        # (104,)
        ilqr_jerk_list.append(torch.from_numpy(sample[22]))         # (103, 2)
        ilqr_jtheta_list.append(torch.from_numpy(sample[23]))       # (103,)
        ilqr_dt_list.append(sample[24])                             # scalar
        start_specific_ilqr_pose_list.append(torch.from_numpy(sample[25]))  # (3,)
        end_specific_ilqr_pose_list.append(torch.from_numpy(sample[26]))    # (3,)

    # Stack fixed-size tensors
    paths = torch.stack(paths, dim=0)                  # (B, 104, 2)
    start_specifics = torch.stack(start_specifics, dim=0)  # (B, 2)
    end_specifics = torch.stack(end_specifics, dim=0)      # (B, 2)
    obstacle_masks = torch.stack(obstacle_masks, dim=0)    # (B, 200, 200)
    family_ids = torch.tensor(family_ids, dtype=torch.long)  # (B,)
    phis = torch.stack(phis, dim=0)                    # (B, 200, 200)
    start_families = torch.stack(start_families, dim=0)    # (B, 2)
    end_families = torch.stack(end_families, dim=0)        # (B, 2)
    flags = torch.tensor(flags, dtype=torch.float32)       # (B,)
    pf_counts = torch.tensor(pf_counts, dtype=torch.long)  # (B,)
    jerky_counts = torch.tensor(jerky_counts, dtype=torch.long)  # (B,)
    shard_idxs = torch.tensor(shard_idxs, dtype=torch.long)
    path_local_idxs = torch.tensor(path_local_idxs, dtype=torch.long)
    global_idxs = torch.tensor(global_idxs, dtype=torch.long)
    ilqr_pose = torch.stack(ilqr_pose_list, dim=0)          # (B, 104, 3)
    ilqr_vel = torch.stack(ilqr_vel_list, dim=0)            # (B, 104, 2)
    ilqr_omega = torch.stack(ilqr_omega_list, dim=0)        # (B, 104)
    ilqr_acc = torch.stack(ilqr_acc_list, dim=0)            # (B, 104, 2)
    ilqr_alpha = torch.stack(ilqr_alpha_list, dim=0)        # (B, 104)
    ilqr_jerk = torch.stack(ilqr_jerk_list, dim=0)          # (B, 103, 2)
    ilqr_jtheta = torch.stack(ilqr_jtheta_list, dim=0)      # (B, 103)
    ilqr_dt = torch.tensor(ilqr_dt_list, dtype=torch.float32)  # (B,)
    start_specific_ilqr_pose = torch.stack(start_specific_ilqr_pose_list, dim=0)  # (B, 3)
    end_specific_ilqr_pose = torch.stack(end_specific_ilqr_pose_list, dim=0)      # (B, 3)
    
    # Pad variable-length obstacle centers across batch
    if len(obstacle_centers) > 0:
        max_obs = int(max(obstacle_counts))
        obstacle_centers_padded = torch.zeros((len(batch), max_obs, 2), dtype=torch.float32)
        obs_idx = 0
        for i, count in enumerate(obstacle_counts):
            if count > 0:
                obstacle_centers_padded[i, :count] = obstacle_centers[obs_idx]
                obs_idx += 1
        obstacle_centers = obstacle_centers_padded
    else:
        obstacle_centers = None
    obstacle_counts = torch.tensor(obstacle_counts, dtype=torch.long)

    # Pad variable-length PF family paths
    max_pf = int(pf_counts.max().item()) if len(pf_counts) > 0 else 0
    if max_pf > 0:
        T = pf_paths_list[0].shape[1]
        D = pf_paths_list[0].shape[2]
        pf_paths_padded = torch.zeros((len(batch), max_pf, T, D), dtype=torch.float32)
        for i, arr in enumerate(pf_paths_list):
            count = arr.shape[0]
            if count > 0:
                pf_paths_padded[i, :count] = arr
    else:
        pf_paths_padded = torch.zeros((len(batch), 0, 104, 2), dtype=torch.float32)

    # Pad variable-length jerky family paths
    max_jerky = int(jerky_counts.max().item()) if len(jerky_counts) > 0 else 0
    if max_jerky > 0:
        T = jerky_paths_list[0].shape[1]
        D = jerky_paths_list[0].shape[2]
        jerky_paths_padded = torch.zeros((len(batch), max_jerky, T, D), dtype=torch.float32)
        for i, arr in enumerate(jerky_paths_list):
            count = arr.shape[0]
            if count > 0:
                jerky_paths_padded[i, :count] = arr
    else:
        jerky_paths_padded = torch.zeros((len(batch), 0, 104, 2), dtype=torch.float32)

    return (
        paths,               # batch[0]  -> tau1
        start_specifics,     # batch[1]
        end_specifics,       # batch[2]
        obstacle_masks,      # batch[3]
        family_ids,          # batch[4]
        phis,                # batch[5]
        start_families,      # batch[6]
        end_families,        # batch[7]
        pf_paths_padded,     # batch[8]
        jerky_paths_padded,  # batch[9]
        flags,               # batch[10]
        pf_counts,           # batch[11]
        jerky_counts,        # batch[12]
        shard_idxs,          # batch[13] optional debug
        path_local_idxs,     # batch[14] optional debug
        global_idxs,         # batch[15] optional debug
        obstacle_centers,    # batch[16] padded (B, max_obs_in_batch, 2) or None
        obstacle_counts,     # batch[17] true obstacle counts per sample
        ilqr_pose,           # batch[18]
        ilqr_vel,            # batch[19]
        ilqr_omega,          # batch[20]
        ilqr_acc,            # batch[21]
        ilqr_alpha,          # batch[22]
        ilqr_jerk,           # batch[23]
        ilqr_jtheta,         # batch[24]
        ilqr_dt,             # batch[25]
        start_specific_ilqr_pose,  # batch[26]
        end_specific_ilqr_pose,    # batch[27]
    )


def build_diverse_paths_dataloader(
    root_dir,
    batch_size,
    shuffle=True,
    num_workers=0,
    pin_memory=True,
    preload_to_ram=True,
    flag_value=1.0,
    drop_last=False,
    disable_obs_centers=False,
):
    dataset = MultiFileDiversePathsDataset(
        root_dir=root_dir,
        preload_to_ram=preload_to_ram,
        flag_value=flag_value,
        disable_obs_centers=disable_obs_centers,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=diverse_paths_collate_fn,
    )

    return dataset, dataloader