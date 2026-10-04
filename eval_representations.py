import argparse

from metric.representation_evaluator import evaluate_representations


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate an image folder in multiple frozen representation spaces."
    )
    parser.add_argument("--image_dir", required=True, help="flat directory of generated images")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--stats_root", default="reference_stats")
    parser.add_argument("--batch_size", type=int, default=64, help="per-GPU batch size")
    parser.add_argument("--workers", type=int, default=8, help="workers per GPU")
    parser.add_argument("--expected_samples", type=int, default=50000)
    parser.add_argument("--allow_nonstandard_sample_count", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate_representations(**vars(args))
