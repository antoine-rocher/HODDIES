import struct
import numpy as np
import glob
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


# ========================================
#  Constants
# ========================================
GADGET_UNIT_LENGTH_IN_MPC = 1.0
GADGET_UNIT_MASS_IN_MSUN = 1.0e10
GADGET_UNIT_VELOCITY_IN_CM_PER_S = 1e5


# ========================================
#  Header Reader
# ========================================
def read_gadget_header(f):
    """Reads the 256-byte Gadget2 header block."""
    blk_size = struct.unpack("I", f.read(4))[0]

    header_fmt = (
        "6i"  # Npart[6]
        "6d"  # Massarr[6]
        "d"   # Time
        "d"   # Redshift
        "i"   # FlagSfr
        "i"   # FlagFeedback
        "6I"  # Nall[6]
        "i"   # FlagCooling
        "i"   # NumFiles
        "d"   # BoxSize
        "d"   # Omega0
        "d"   # OmegaLambda
        "d"   # HubbleParam
        "i"   # Flag_StellarAge
        "i"   # Flag_Metals
        "6I"  # NallHW[6]
        "i"   # flag_entr_ics
        "60x" # padding
    )
    header_size = struct.calcsize(header_fmt)
    header_data = struct.unpack(header_fmt, f.read(header_size))
    f.read(4)  # trailing block size

    header = {
        "Npart": np.array(header_data[0:6], dtype=np.int32),
        "Massarr": np.array(header_data[6:12], dtype=np.float64),
        "Time": header_data[12],
        "Redshift": header_data[13],
        "NumFiles": header_data[22],
        "BoxSize": round(header_data[24]),
        "Omega0": header_data[25],
        "OmegaLambda": header_data[26],
        "HubbleParam": header_data[27],
    }
    return header


# ========================================
#  Helper for chunk reading
# ========================================
def _read_chunk(f, offset, count, dtype, shape):
    """Reads a specific chunk from a binary file."""
    f.seek(offset)
    arr = np.fromfile(f, dtype=dtype, count=count)
    return arr.reshape(shape)


# ========================================
#  Single-file Reader
# ========================================
def read_gadget2_single(filename, long_ids=True, chunk_size=2_000_000, n_threads=4, subset_fraction=None):
    """
    Reads one Gadget-2 binary file efficiently, with threading, chunking, and optional subsampling.
    Returns numpy arrays instead of a DataFrame.
    """
    with open(filename, "rb") as f:
        header = read_gadget_header(f)
        npart = header["Npart"][1]
        if npart == 0:
            return {}, header

        mass = header["Massarr"][1]

        # Locate data blocks
        struct.unpack("I", f.read(4))
        pos_offset = f.tell()
        pos_bytes = 3 * npart * 4
        f.seek(pos_offset + pos_bytes + 4)

        struct.unpack("I", f.read(4))
        vel_offset = f.tell()
        vel_bytes = 3 * npart * 4
        f.seek(vel_offset + vel_bytes + 4)

        struct.unpack("I", f.read(4))
        id_offset = f.tell()
        id_dtype = np.int64 if long_ids else np.int32

    pos_dtype = np.float32
    vel_dtype = np.float32

    def process_chunk(chunk_index, subset_fraction=None):
        with open(filename, "rb") as f:
            start = chunk_index * chunk_size
            end = min(start + chunk_size, npart)
            nread = end - start
            if nread <= 0:
                return None
            
            if subset_fraction and 0 < subset_fraction < 1.0:
                mask = np.random.uniform(size=nread) < subset_fraction
            else:
                mask = np.ones(nread, dtype=bool)
            p_offset = pos_offset + start * 3 * 4
            v_offset = vel_offset + start * 3 * 4
            i_offset = id_offset + start * np.dtype(id_dtype).itemsize
        
            pos_chunk = _read_chunk(f, p_offset, nread * 3, pos_dtype, (nread, 3))[mask]
            vel_chunk = _read_chunk(f, v_offset, nread * 3, vel_dtype, (nread, 3))[mask]
            ids_chunk = _read_chunk(f, i_offset, nread, id_dtype, (nread,))[mask]

            return {"id": ids_chunk, "pos": pos_chunk, "vel": vel_chunk}

    chunks = range((npart + chunk_size - 1) // chunk_size)
    results = []

    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        futures = {executor.submit(process_chunk, i, subset_fraction=subset_fraction): i for i in chunks}
        for future in tqdm(as_completed(futures), total=len(chunks), desc=f"Reading {os.path.basename(filename)}"):
            r = future.result()
            if r is not None:
                results.append(r)

    if not results:
        return {}, header

    ids = np.concatenate([r["id"] for r in results])
    pos = np.vstack([r["pos"] for r in results])
    vel = np.vstack([r["vel"] for r in results])

    # # Optional random subset
    # if subset_fraction and 0 < subset_fraction < 1.0:
    #     mask = np.random.uniform(size=len(ids)) < subset_fraction
    #     ids = ids[mask]
    #     pos = pos[mask]
    #     vel = vel[mask]

    data = {
        "id": ids,
        "pos": pos,
        "vel": vel,
    }
    return data, header


# ========================================
#  Multi-file Reader
# ========================================
def read_gadget2_multi(snapshot_prefix, long_ids=True, chunk_size=2_000_000, n_threads=4, subset_fraction=None):
    """
    Reads a multi-file Gadget-2 snapshot (snapshot_XXX.0, snapshot_XXX.1, ...),
    returning Numpy arrays and a combined header.
    """
    pattern = f"{snapshot_prefix}*"
    subfiles = sorted(glob.glob(pattern))

    if not subfiles:
        if os.path.exists(snapshot_prefix):
            subfiles = [snapshot_prefix]
        else:
            raise FileNotFoundError(f"No Gadget-2 files found for {snapshot_prefix}")

    print(f"Found {len(subfiles)} Gadget subfiles.")
    all_ids, all_pos, all_vel, all_mass = [], [], [], []
    header_main = None

    for sub in subfiles:
        data, header = read_gadget2_single(sub, long_ids, chunk_size, n_threads, subset_fraction)
        if data:
            all_ids.append(data["id"])
            all_pos.append(data["pos"])
            all_vel.append(data["vel"])
        if header_main is None:
            header_main = header

    if not all_ids:
        raise ValueError("No particle data found in any subfile.")

    ids = np.concatenate(all_ids)
    pos = np.vstack(all_pos)
    vel = np.vstack(all_vel)

    return {"id": ids, "pos": pos, "vel": vel}, header_main


# ========================================
#  Example Usage
# ========================================
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python read_gadget2_final.py <snapshot_prefix> [subset_fraction]")
        sys.exit(1)

    prefix = sys.argv[1]
    subset = float(sys.argv[2]) if len(sys.argv) > 2 else None
    n_threads = float(sys.argv[3]) if len(sys.argv) > 3 else 16
    data, header = read_gadget2_multi(prefix, n_threads=n_threads, subset_fraction=subset)

    print("\n=== Snapshot Summary ===")
    print(f"Total particles read: {len(data['id']):,}")
    print(f"Particles Mass: {header['Massarr'][1]} 10^10 Msun/h")
    print(f"BoxSize: {header['BoxSize']} Mpc/h")
    print(f"Omega0: {header['Omega0']}, OmegaLambda: {header['OmegaLambda']}")
    print("\nSample positions and velocities:")
    print(data['pos'][:5])
    print(data['vel'][:5])

    print(header.keys()) 