import argparse

from utils.eval_utils import compute_openai_fid, save_eval_results


def main():
    parser = argparse.ArgumentParser(description="Compute OpenAI ImageNet-256 FID/IS from a sample NPZ.")
    parser.add_argument("--sample_npz", required=True)
    parser.add_argument("--fid_reference_path", required=True)
    parser.add_argument("--inception_graph_path", required=True)
    parser.add_argument("--fid_batch_size", type=int, default=64)
    parser.add_argument("--metric_name", choices=["rFID", "gFID"], required=True)
    parser.add_argument("--output_json", required=True)
    args = parser.parse_args()

    fid, is_score = compute_openai_fid(
        args.fid_reference_path,
        args.sample_npz,
        args.inception_graph_path,
        args.fid_batch_size,
    )
    save_eval_results(
        args.output_json,
        metric=args.metric_name,
        fid=fid,
        inception_score=is_score,
        reference_npz=args.fid_reference_path,
        sample_npz=args.sample_npz,
    )
    print(f"{args.metric_name}: {fid:.4f}, IS: {is_score:.4f}")
    print(f"Saved evaluation results to {args.output_json}")


if __name__ == "__main__":
    main()
