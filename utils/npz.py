import numpy as np
from PIL import Image
from tqdm import tqdm

def create_npz_from_sample_folder(sample_dir, num=50000, labels=None):
    """
    Builds a single .npz file from a folder of .png samples.
    """
    samples = []
    for i in tqdm(range(num), desc="Building .npz file from samples"):
        sample_pil = Image.open(f"{sample_dir}/{i:06d}.png")
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    samples = np.stack(samples)
    assert samples.shape == (num, samples.shape[1], samples.shape[2], 3)
    npz_path = f"{sample_dir}.npz"
    arrays = {"arr_0": samples}
    if labels is not None:
        labels = np.asarray(labels, dtype=np.int64)
        if labels.shape != (num,):
            raise ValueError(f"Expected {num} labels, got shape {labels.shape}")
        arrays["arr_1"] = labels
    np.savez(npz_path, **arrays)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path
