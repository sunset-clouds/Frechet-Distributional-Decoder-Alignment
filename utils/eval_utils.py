import json
import os


def load_tokenizer_checkpoint(tokenizer, checkpoint_path):
    import torch

    if not checkpoint_path:
        return
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Evaluation checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
    else:
        state_dict = checkpoint
    state_dict = {
        key.removeprefix("module."): value for key, value in state_dict.items()
    }
    missing, unexpected = tokenizer.load_state_dict(state_dict, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected tokenizer checkpoint keys: {unexpected[:10]}")
    print(f"Loaded tokenizer checkpoint: {checkpoint_path}")
    if missing:
        print(f"Checkpoint missing {len(missing)} keys; pretrained values are retained.")


def compute_openai_fid(reference_path, sample_path, graph_path, batch_size):
    if not os.path.isfile(reference_path):
        raise FileNotFoundError(
            "OpenAI ImageNet reference not found. Set --fid_reference_path to "
            "VIRTUAL_imagenet256_labeled.npz."
        )
    if not os.path.isfile(sample_path):
        raise FileNotFoundError(f"Sample NPZ not found: {sample_path}")
    if graph_path:
        if not os.path.isfile(graph_path):
            raise FileNotFoundError(f"OpenAI Inception graph not found: {graph_path}")
        os.environ["OPENAI_INCEPTION_GRAPH"] = graph_path

    from metric.evaluator import FIDEvaluator

    evaluator = FIDEvaluator(batch_size=batch_size)
    try:
        return evaluator.compute_fid_and_is(reference_path, sample_path)
    finally:
        evaluator.close()


def save_eval_results(path, **values):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(values, handle, indent=2, sort_keys=True)
