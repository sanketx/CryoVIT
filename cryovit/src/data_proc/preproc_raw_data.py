"""Downsample and copy float data.

This program downsamples and normalized the high res float tomograms
which were originally provided to us. Regardless of the format, they
are saved as HDF5 files.

Specify sample as "All" to copy all samples
Otherwise specify individual samples

Usage:
  copy_old_data.py <sample>
  copy_old_data.py (-h | --help)

Options:
  -h --help         Show this screen

Notes:
1. ipsc_q109_square19_tomo02_bin4_filtered_hairball.rec
   has been moved from hairball to mitochondria folder
2. 20211201_BACHD_control_RNA_tomo00026_bin4_filtered_rec.mrc
   needs to be ignored
3. 20211201_BACHD_control_RNA_tomo000012_redo1_bin4_filtered.rec
   has an identical copy in hdf format which needs to be ignored
4. 20210427_Q66_ipdc_grid2_tomo00012__bin4_large_granules_preproc.hdf
   has an indistinct mitochondrion and is excluded
5. Q77_ipscell06possagg_bin2.hdf
   needs to be ignored
6. Remove "normal" and "normal_beautiful" suffix from WT names
   `rename bin4_normal.hdf bin4.hdf *`
   `rename bin4_normal_beautiful.hdf bin4.hdf *`
   `rename bin4_normal_beautifult.hdf bin4.hdf *`


"""
import os

import h5py
import mrcfile
import numpy as np
import torch
from docopt import docopt

from .. import config

pooler = torch.nn.AvgPool3d(kernel_size=2, stride=2, ceil_mode=True)


sample_list = {
    "BACHD": "BACHD",
    "BACHD_controlRNA": "BACHD_controlRNA",
    "BACHD_pias1": "BACHD_pias1",
    "dN17_BACHD": "dN17_BACHD",
    "Q18": "Q18",
    "Q20": "Q20",
    "Q53": "Q53",
    "Q53_KD": "Q53_KD",
    "Q66": "Q66/Q66/mitochondria",
    "Q66_KD": "Q66_AFISi",
    "Q66_GRFS1": "Q66_GRFS1",
    "Q77": "Q77/mitochondria",
    "Q109": "Q109/20200808-C036_TEM2_Q109/mitochondria",
    "WT": "WT",
    "cancer": "cancer",
}


def copy_file(src_path, dst_path):
    if src_path.endswith(".hdf"):
        with h5py.File(src_path) as fh:
            data = fh["MDF"]["images"]["0"]["image"][()]

    else:
        with mrcfile.open(src_path) as fh:
            data = fh.data

    data = np.expand_dims(data, [0, 1])
    data = torch.tensor(data)

    data = pooler(data)
    # data = torch.nn.functional.interpolate(data, size=(128, 512, 512))

    data = data.squeeze().numpy()
    print(f"Shape is {data.shape}, {data.dtype}")

    # Contrast normalized samples are clipped to +/-3 std devs.
    # We further rescale to [-1, 1]
    data = np.clip(data, -3.0, 3.0) / 3.0

    with h5py.File(dst_path, "w") as fh:
        fh.create_dataset("data", data=data)


def copy_samples(sample):
    src_dir = os.path.join(config.HD_DIR, sample_list[sample])
    dst_dir = os.path.join(config.RAW_TOMO_DIR, sample)
    os.makedirs(dst_dir, exist_ok=True)

    print(f"Found {len(os.listdir(src_dir))} samples of {sample}")

    for file_name in os.listdir(src_dir):
        if not (
            file_name.endswith(".rec")
            or file_name.endswith(".mrc")
            or file_name.endswith(".hdf")
        ):
            continue

        print(f"Copying {file_name}")
        src_path = os.path.join(src_dir, file_name)
        dst_path = os.path.join(dst_dir, file_name)
        dst_path = dst_path.split(".")[0] + ".hdf"

        try:
            copy_file(src_path, dst_path)
        except Exception as e:
            print(e)


def main():
    args = docopt(__doc__)
    sample = args["<sample>"]

    if sample in sample_list:
        copy_samples(sample)

    elif sample == "All":
        for sample in sample_list:
            copy_samples(sample)

    else:
        print(f"Invalid sample: {sample}")


if __name__ == "__main__":
    main()
