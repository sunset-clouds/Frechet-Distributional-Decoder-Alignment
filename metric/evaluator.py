"""
FID/IS Evaluator using OpenAI's official Inception V3 model (TensorFlow).

Provides two classes:
- FIDEvaluator: Clean Python API for computing FID/IS from NPZ files.
- Evaluator: Original VQ-Transplant CLI-compatible evaluator (legacy).

Both use the same TF Inception V3 model from OpenAI's guided-diffusion,
ensuring consistency with ADM/DiT/SiT FID benchmarks.

Images should be uint8 [0, 255], NHWC format, stored as 'arr_0' key in NPZ.

Usage:
    from metric.evaluator import FIDEvaluator

    evaluator = FIDEvaluator()
    fid, is_score = evaluator.compute_fid_and_is(
        ref_npz="/path/to/VIRTUAL_imagenet256_labeled.npz",
        sample_npz="/path/to/Generated.npz"
    )
    evaluator.close()
"""

import io
import os
import random
import warnings
import zipfile
from abc import ABC, abstractmethod
from contextlib import contextmanager
from functools import partial
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool
from typing import Iterable, Optional, Tuple

import numpy as np
import tensorflow.compat.v1 as tf
from scipy import linalg
from tqdm.auto import tqdm

INCEPTION_V3_PATH = os.environ.get(
    "OPENAI_INCEPTION_GRAPH",
    os.path.join(os.path.dirname(__file__), "classify_image_graph_def.pb"),
)

FID_POOL_NAME = "pool_3:0"
FID_SPATIAL_NAME = "mixed_6/conv:0"


# ============================================================
# FID Statistics
# ============================================================

class FIDStatistics:
    def __init__(self, mu: np.ndarray, sigma: np.ndarray):
        self.mu = mu
        self.sigma = sigma

    def frechet_distance(self, other, eps=1e-6):
        mu1, sigma1 = self.mu, self.sigma
        mu2, sigma2 = other.mu, other.sigma

        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)
        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        assert mu1.shape == mu2.shape, \
            f"Mean vectors have different lengths: {mu1.shape}, {mu2.shape}"
        assert sigma1.shape == sigma2.shape, \
            f"Covariances have different dimensions: {sigma1.shape}, {sigma2.shape}"

        diff = mu1 - mu2
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            warnings.warn(
                f"FID calculation produces singular product; adding {eps} to diagonal"
            )
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError(f"Imaginary component {m}")
            covmean = covmean.real

        tr_covmean = np.trace(covmean)
        return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


# ============================================================
# FIDEvaluator — Clean Python API
# ============================================================

class FIDEvaluator:
    """
    FID/IS evaluator using OpenAI's TF Inception V3.
    Consistent with ADM/DiT/SiT benchmarks using VIRTUAL_imagenet256_labeled.npz.
    """

    def __init__(self, batch_size=64, softmax_batch_size=512):
        self.batch_size = batch_size
        self.softmax_batch_size = softmax_batch_size
        self.session = None
        self._initialized = False

    def _initialize(self):
        if self._initialized:
            return

        tf.disable_v2_behavior()

        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        self.session = tf.Session(config=config)

        with self.session.graph.as_default():
            self.image_input = tf.placeholder(tf.float32, shape=[None, None, None, 3])
            self.softmax_input = tf.placeholder(tf.float32, shape=[None, 2048])
            self.pool_features, self.spatial_features = _create_feature_graph(self.image_input)
            self.softmax = _create_softmax_graph(self.softmax_input)

        print("Warming up TensorFlow Inception model...")
        self._compute_activations_from_array(np.zeros([1, 64, 64, 3], dtype=np.uint8))
        self._initialized = True
        print("FIDEvaluator initialized (OpenAI TF Inception)")

    def _compute_activations_from_array(self, images: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        preds = []
        spatial_preds = []
        for i in range(0, len(images), self.batch_size):
            batch = images[i:i+self.batch_size].astype(np.float32)
            pred, spatial_pred = self.session.run(
                [self.pool_features, self.spatial_features],
                {self.image_input: batch}
            )
            preds.append(pred.reshape([pred.shape[0], -1]))
            spatial_preds.append(spatial_pred.reshape([spatial_pred.shape[0], -1]))
        return np.concatenate(preds, axis=0), np.concatenate(spatial_preds, axis=0)

    def _compute_activations_from_npz(self, npz_path: str) -> Tuple[np.ndarray, np.ndarray]:
        preds = []
        spatial_preds = []

        # Pre-computed reference stats (has 'mu' key) — load directly
        obj = np.load(npz_path)
        if "mu" in obj.files:
            pool = obj.get("pool_3", None)
            if pool is not None:
                return pool, pool
            mu = obj["mu"]
            sigma = obj["sigma"]
            return mu, None

        with open_npz_array(npz_path, "arr_0") as reader:
            for batch in tqdm(reader.read_batches(self.batch_size), desc="Computing activations"):
                batch = batch.astype(np.float32)
                pred, spatial_pred = self.session.run(
                    [self.pool_features, self.spatial_features],
                    {self.image_input: batch}
                )
                preds.append(pred.reshape([pred.shape[0], -1]))
                spatial_preds.append(spatial_pred.reshape([spatial_pred.shape[0], -1]))
        return np.concatenate(preds, axis=0), np.concatenate(spatial_preds, axis=0)

    def _read_statistics(self, npz_path: str) -> FIDStatistics:
        """Read stats from an NPZ: either pre-computed (mu/sigma) or compute from arr_0."""
        obj = np.load(npz_path)
        if "mu" in obj.files:
            return FIDStatistics(obj["mu"], obj["sigma"])
        acts, _ = self._compute_activations_from_npz(npz_path)
        mu = np.mean(acts, axis=0)
        sigma = np.cov(acts, rowvar=False)
        return FIDStatistics(mu, sigma)

    def _compute_inception_score(self, activations: np.ndarray, split_size: int = 5000) -> float:
        softmax_out = []
        activations = activations.copy()
        np.random.shuffle(activations)
        for i in range(0, len(activations), self.softmax_batch_size):
            acts = activations[i:i + self.softmax_batch_size]
            softmax_out.append(
                self.session.run(self.softmax, feed_dict={self.softmax_input: acts})
            )
        preds = np.concatenate(softmax_out, axis=0)
        scores = []
        for i in range(0, len(preds), split_size):
            part = preds[i:i + split_size]
            kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
            kl = np.mean(np.sum(kl, 1))
            scores.append(np.exp(kl))
        return float(np.mean(scores))

    def compute_fid_and_is(self, ref_npz: str, sample_npz: str) -> Tuple[float, float]:
        """
        Compute FID and Inception Score.

        ref_npz can be either:
        - Pre-computed stats (has 'mu'/'sigma' keys), e.g. VIRTUAL_imagenet256_labeled.npz
        - Raw images (has 'arr_0' key), e.g. Input.npz
        """
        self._initialize()

        print(f"Loading reference stats from {ref_npz}...")
        ref_stats = self._read_statistics(ref_npz)

        print(f"Computing sample activations from {sample_npz}...")
        sample_acts = self._compute_activations_from_npz(sample_npz)
        sample_mu = np.mean(sample_acts[0], axis=0)
        sample_sigma = np.cov(sample_acts[0], rowvar=False)
        sample_stats = FIDStatistics(sample_mu, sample_sigma)

        fid = sample_stats.frechet_distance(ref_stats)
        is_score = self._compute_inception_score(sample_acts[0])

        return float(fid), float(is_score)

    def compute_fid(self, ref_npz: str, sample_npz: str) -> float:
        fid, _ = self.compute_fid_and_is(ref_npz, sample_npz)
        return fid

    def close(self):
        if self.session is not None:
            self.session.close()
            self.session = None
            self._initialized = False


# ============================================================
# Legacy Evaluator (VQ-Transplant compatible CLI)
# ============================================================

class Evaluator:
    """Original VQ-Transplant evaluator. Kept for backward compatibility."""

    def __init__(self, session, batch_size=64, softmax_batch_size=512):
        self.sess = session
        self.batch_size = batch_size
        self.softmax_batch_size = softmax_batch_size
        with self.sess.graph.as_default():
            self.image_input = tf.placeholder(tf.float32, shape=[None, None, None, 3])
            self.softmax_input = tf.placeholder(tf.float32, shape=[None, 2048])
            self.pool_features, self.spatial_features = _create_feature_graph(self.image_input)
            self.softmax = _create_softmax_graph(self.softmax_input)

    def warmup(self):
        self.compute_activations(np.zeros([1, 8, 64, 64, 3]))

    def read_activations(self, npz_path: str) -> Tuple[np.ndarray, np.ndarray]:
        with open_npz_array(npz_path, "arr_0") as reader:
            return self.compute_activations(reader.read_batches(self.batch_size))

    def compute_activations(self, batches: Iterable[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        preds = []
        spatial_preds = []
        for batch in tqdm(batches):
            batch = batch.astype(np.float32)
            pred, spatial_pred = self.sess.run(
                [self.pool_features, self.spatial_features], {self.image_input: batch}
            )
            preds.append(pred.reshape([pred.shape[0], -1]))
            spatial_preds.append(spatial_pred.reshape([spatial_pred.shape[0], -1]))
        return np.concatenate(preds, axis=0), np.concatenate(spatial_preds, axis=0)

    def read_statistics(
        self, npz_path: str, activations: Tuple[np.ndarray, np.ndarray]
    ) -> Tuple[FIDStatistics, FIDStatistics]:
        obj = np.load(npz_path)
        if "mu" in list(obj.keys()):
            return FIDStatistics(obj["mu"], obj["sigma"]), FIDStatistics(
                obj["mu_s"], obj["sigma_s"]
            )
        return tuple(self.compute_statistics(x) for x in activations)

    def compute_statistics(self, activations: np.ndarray) -> FIDStatistics:
        mu = np.mean(activations, axis=0)
        sigma = np.cov(activations, rowvar=False)
        return FIDStatistics(mu, sigma)

    def compute_inception_score(self, activations: np.ndarray, split_size: int = 5000) -> float:
        softmax_out = []
        np.random.shuffle(activations)
        for i in range(0, len(activations), self.softmax_batch_size):
            acts = activations[i:i + self.softmax_batch_size]
            softmax_out.append(self.sess.run(self.softmax, feed_dict={self.softmax_input: acts}))
        preds = np.concatenate(softmax_out, axis=0)
        scores = []
        for i in range(0, len(preds), split_size):
            part = preds[i:i + split_size]
            kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
            kl = np.mean(np.sum(kl, 1))
            scores.append(np.exp(kl))
        return float(np.mean(scores))


# ============================================================
# NPZ Array Reader utilities
# ============================================================

class NpzArrayReader(ABC):
    @abstractmethod
    def read_batch(self, batch_size: int) -> Optional[np.ndarray]:
        pass

    @abstractmethod
    def remaining(self) -> int:
        pass

    def read_batches(self, batch_size: int) -> Iterable[np.ndarray]:
        def gen_fn():
            while True:
                batch = self.read_batch(batch_size)
                if batch is None:
                    break
                yield batch
        rem = self.remaining()
        num_batches = rem // batch_size + int(rem % batch_size != 0)
        return BatchIterator(gen_fn, num_batches)


class BatchIterator:
    def __init__(self, gen_fn, length):
        self.gen_fn = gen_fn
        self.length = length

    def __len__(self):
        return self.length

    def __iter__(self):
        return self.gen_fn()


class StreamingNpzArrayReader(NpzArrayReader):
    def __init__(self, arr_f, shape, dtype):
        self.arr_f = arr_f
        self.shape = shape
        self.dtype = dtype
        self.idx = 0

    def read_batch(self, batch_size: int) -> Optional[np.ndarray]:
        if self.idx >= self.shape[0]:
            return None
        bs = min(batch_size, self.shape[0] - self.idx)
        self.idx += bs
        if self.dtype.itemsize == 0:
            return np.ndarray([bs, *self.shape[1:]], dtype=self.dtype)
        read_count = bs * np.prod(self.shape[1:])
        read_size = int(read_count * self.dtype.itemsize)
        data = _read_bytes(self.arr_f, read_size, "array data")
        return np.frombuffer(data, dtype=self.dtype).reshape([bs, *self.shape[1:]])

    def remaining(self) -> int:
        return max(0, self.shape[0] - self.idx)


class MemoryNpzArrayReader(NpzArrayReader):
    def __init__(self, arr):
        self.arr = arr
        self.idx = 0

    @classmethod
    def load(cls, path: str, arr_name: str):
        with open(path, "rb") as f:
            arr = np.load(f)[arr_name]
        return cls(arr)

    def read_batch(self, batch_size: int) -> Optional[np.ndarray]:
        if self.idx >= self.arr.shape[0]:
            return None
        res = self.arr[self.idx:self.idx + batch_size]
        self.idx += batch_size
        return res

    def remaining(self) -> int:
        return max(0, self.arr.shape[0] - self.idx)


@contextmanager
def open_npz_array(path: str, arr_name: str) -> NpzArrayReader:
    with _open_npy_file(path, arr_name) as arr_f:
        version = np.lib.format.read_magic(arr_f)
        if version == (1, 0):
            header = np.lib.format.read_array_header_1_0(arr_f)
        elif version == (2, 0):
            header = np.lib.format.read_array_header_2_0(arr_f)
        else:
            yield MemoryNpzArrayReader.load(path, arr_name)
            return
        shape, fortran, dtype = header
        if fortran or dtype.hasobject:
            yield MemoryNpzArrayReader.load(path, arr_name)
        else:
            yield StreamingNpzArrayReader(arr_f, shape, dtype)


def _read_bytes(fp, size, error_template="ran out of data"):
    data = bytes()
    while True:
        try:
            r = fp.read(size - len(data))
            data += r
            if len(r) == 0 or len(data) == size:
                break
        except io.BlockingIOError:
            pass
    if len(data) != size:
        msg = "EOF: reading %s, expected %d bytes got %d"
        raise ValueError(msg % (error_template, size, len(data)))
    return data


@contextmanager
def _open_npy_file(path: str, arr_name: str):
    with open(path, "rb") as f:
        with zipfile.ZipFile(f, "r") as zip_f:
            if f"{arr_name}.npy" not in zip_f.namelist():
                raise ValueError(f"missing {arr_name} in npz file")
            with zip_f.open(f"{arr_name}.npy", "r") as arr_f:
                yield arr_f


# ============================================================
# Inception V3 model loading
# ============================================================

def _download_inception_model():
    if os.path.exists(INCEPTION_V3_PATH):
        return
    raise FileNotFoundError(
        "OpenAI Inception graph not found. Set OPENAI_INCEPTION_GRAPH to "
        "classify_image_graph_def.pb or place it at "
        f"{INCEPTION_V3_PATH}."
    )


def _create_feature_graph(input_batch):
    _download_inception_model()
    prefix = f"{random.randrange(2**32)}_{random.randrange(2**32)}"
    with open(INCEPTION_V3_PATH, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    pool3, spatial = tf.import_graph_def(
        graph_def,
        input_map={f"ExpandDims:0": input_batch},
        return_elements=[FID_POOL_NAME, FID_SPATIAL_NAME],
        name=prefix,
    )
    _update_shapes(pool3)
    spatial = spatial[..., :7]
    return pool3, spatial


def _create_softmax_graph(input_batch):
    _download_inception_model()
    prefix = f"{random.randrange(2**32)}_{random.randrange(2**32)}"
    with open(INCEPTION_V3_PATH, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    (matmul,) = tf.import_graph_def(
        graph_def, return_elements=[f"softmax/logits/MatMul"], name=prefix
    )
    w = matmul.inputs[1]
    logits = tf.matmul(input_batch, w)
    return tf.nn.softmax(logits)


def _update_shapes(pool3):
    ops = pool3.graph.get_operations()
    for op in ops:
        for o in op.outputs:
            shape = o.get_shape()
            if shape._dims is not None:
                shape = [s for s in shape]
                new_shape = []
                for j, s in enumerate(shape):
                    if s == 1 and j == 0:
                        new_shape.append(None)
                    else:
                        new_shape.append(s)
                o.__dict__["_shape_val"] = tf.TensorShape(new_shape)
    return pool3


# ============================================================
# Command-line interface
# ============================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Compute FID and IS from NPZ files")
    parser.add_argument("--ref_batch", required=True, help="Path to reference NPZ file")
    parser.add_argument("--sample_path", required=True, help="Directory containing sample NPZ")
    parser.add_argument("--sample_name", required=True, help="Sample NPZ filename")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    args = parser.parse_args()

    sample_batch = os.path.join(args.sample_path, args.sample_name)

    print("=" * 60)
    print("FID/IS Evaluation (OpenAI TF Inception)")
    print("=" * 60)
    print(f"Reference: {args.ref_batch}")
    print(f"Sample: {sample_batch}")
    print("=" * 60)

    evaluator = FIDEvaluator(batch_size=args.batch_size)
    try:
        fid, is_score = evaluator.compute_fid_and_is(args.ref_batch, sample_batch)
        print("\n" + "=" * 60)
        print(f"Inception Score: {is_score:.4f}")
        print(f"FID: {fid:.4f}")
        print("=" * 60)

        txt_path = sample_batch.replace('.npz', '.txt')
        with open(txt_path, 'w') as f:
            print(f"Inception Score: {is_score}", file=f)
            print(f"FID: {fid}", file=f)
        print(f"Results saved to: {txt_path}")
    finally:
        evaluator.close()


if __name__ == "__main__":
    main()
