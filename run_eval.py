if __name__ == "__main__":
    import os
    import json
    import torch

    from transformers import logging
    from argparse import ArgumentParser

    from scripts import Evaluator, rank_zero_print

    parser = ArgumentParser()

    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--strategy", default=None, type=str)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--corpus_path", default="./WSD_Evaluation_Framework", type=str)
    parser.add_argument("--checkpoint_path", default="./checkpoints", type=str)
    parser.add_argument("--prediction_path", default="./predictions", type=str)
    parser.add_argument("--model_name_or_path", default="./deberta-base", type=str)
    parser.add_argument("--sbert_name_or_path", default="./all-mpnet-base-v2", type=str)
    parser.add_argument("--sbert_batch_size", default=256, type=int)
    parser.add_argument("--sbert_device", default=None, choices=["cpu", "cuda"])
    parser.add_argument("--simple_display", action="store_true")

    args = parser.parse_args()

    logging.set_verbosity_error()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    torch.multiprocessing.set_sharing_strategy("file_system")

    evaluator = Evaluator(args)
    rank_zero_print(f"Results:\n{json.dumps(evaluator.run(), indent=4, sort_keys=True)}\n")
