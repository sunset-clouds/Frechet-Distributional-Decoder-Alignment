import argparse
import os


def str2bool(v):
    """'yes'/'no'-style string -> bool."""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
    

def manage_checkpoints(save_dir, keep_last_n=10):
    """Remove all but the newest keep_last_n + 1 epoch-numbered .pt files in save_dir."""
    checkpoints = [f for f in os.listdir(save_dir) if f.endswith('.pt')]
    checkpoints = [f for f in checkpoints if 'best_ckpt' not in f]
    checkpoints.sort(key=lambda f: int(f.split('/')[-1].split('.')[0]))  # by epoch number

    if len(checkpoints) > keep_last_n + 1:
        for checkpoint_file in checkpoints[:-keep_last_n-1]:
            checkpoint_path = os.path.join(save_dir, checkpoint_file)
            if os.path.exists(checkpoint_path):
                os.remove(checkpoint_path)
                print(f"Removed old checkpoint: {checkpoint_path}")


def load_model_state_dict(orig_state_dict):
    model_state = {}
    for key, value in orig_state_dict.items():
        while key.startswith(("module.", "_orig_mod.")):
            key = key.split(".", 1)[1]
        model_state[key] = value
    return model_state