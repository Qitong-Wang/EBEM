if __name__ == "__main__":
    import os
    import sys
    import json
    import torch

    from transformers import logging
    from argparse import ArgumentParser
    from warnings import filterwarnings
    from pytorch_lightning.loggers import WandbLogger
    from pytorch_lightning.callbacks import Timer, ModelCheckpoint, LearningRateMonitor

    from scripts import get_rank, get_trainer, PlTqdm, PlModule, Evaluator, PlDataModule, PlPredictionWriter

    parser = ArgumentParser()

    parser.add_argument("--model_name_or_path", default="./deberta-base", type=str)
    parser.add_argument("--framework", default="EBEM", choices=["BEM", "EBEM"])
    parser.add_argument("--circle_loss_m", default=0.5, type=float)
    parser.add_argument("--circle_loss_gamma", default=32, type=int)
    parser.add_argument("--learning_rate", default=5e-6, type=float)
    parser.add_argument("--warmup_proportion", default=0.2, type=float)
    parser.add_argument("--min_semantic_sim", default=0.3, type=float)
    parser.add_argument("--optimizer", default="AdamW", choices=["AdamW", "Adafactor"])
    parser.add_argument("--polynomial_power", default=1.0, type=float)
    parser.add_argument("--weight_decay", default=0.01, type=float)
    parser.add_argument("--epsilon", default=None, type=float)
    parser.add_argument("--dropout", default=0.1, type=float)
    parser.add_argument("--gradient_clip_val", default=1.0, type=float)

    parser.add_argument("--seed", default=55, type=int)
    parser.add_argument("--corpus_path", default="./WSD_Evaluation_Framework", type=str)
    parser.add_argument("--train_data", default="semcor", type=str)
    parser.add_argument("--val_data", default="semeval2007", type=str)
    parser.add_argument("--eval_data", default="ALL", type=str)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--max_epochs", default=5, type=int)
    parser.add_argument("--max_context_seq_len", default=256, type=int)
    parser.add_argument("--max_gloss_seq_len", default=48, type=int)
    parser.add_argument("--precision", default=32, type=int, choices=[16, 32])
    parser.add_argument("--amp_backend", default="native", choices=["native", "apex"])
    parser.add_argument("--amp_level", default="O1", choices=["O0", "O1", "O2", "O3"])
    parser.add_argument("--accumulate_grad_batches", default=1, type=int)
    parser.add_argument("--sbert_name_or_path", default="./all-mpnet-base-v2", type=str)
    parser.add_argument("--sbert_batch_size", default=None, type=int)
    parser.add_argument("--sbert_device", default=None, choices=["cpu", "cuda"])
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--kshot", default=None, type=int)
    parser.add_argument("--no_shuffle_data", action="store_true")
    parser.add_argument("--no_correct_text", action="store_true")
    parser.add_argument("--no_add_multi_gold", action="store_true")
    parser.add_argument("--no_add_extra_gloss", action="store_true")
    parser.add_argument("--no_add_extra_context", action="store_true")

    parser.add_argument("--time_limit", default=None, type=float)
    parser.add_argument("--use_gradient_checkpointing", action="store_true")
    parser.add_argument("--log_every_n_steps", default=50, type=int)
    parser.add_argument("--val_check_interval", default=0.5, type=float)
    parser.add_argument("--resume_ckpt_path", default=None, type=str)
    parser.add_argument("--enable_checkpointing", action="store_true")
    parser.add_argument("--checkpoint_path", default="./checkpoints", type=str)
    parser.add_argument("--prediction_path", default="./predictions", type=str)
    parser.add_argument("--save_top_k", default=3, type=int)
    parser.add_argument("--wandb_name", default=None, type=str)
    parser.add_argument("--wandb_project", default="EBEM", type=str)
    parser.add_argument("--wandb_log_model", action="store_true")
    parser.add_argument("--wandb_offline", action="store_true")
    parser.add_argument("--wandb_disable", action="store_true")
    parser.add_argument("--no_upload_code", action="store_true")
    parser.add_argument("--simple_display", action="store_true")

    args = parser.parse_args()
    for key, value in vars(args).items():
        if key.startswith("no_") and isinstance(value, bool):
            setattr(args, key[3:], bool(1 - value))
            delattr(args, key)

    if args.wandb_disable:
        os.environ["WANDB_MODE"] = "disabled"
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["WANDB_SILENT"] = str(args.simple_display)
    torch.multiprocessing.set_sharing_strategy("file_system")

    logging.set_verbosity_error()
    if args.simple_display:
        filterwarnings("ignore", category=Warning)

    logger = WandbLogger(
            name=args.wandb_name,
            offline=args.wandb_offline,
            project=args.wandb_project,
            log_model=args.enable_checkpointing and args.wandb_log_model
    )
    logger.experiment.define_metric(name="val_loss", summary="min")
    logger.experiment.define_metric(name="val_acc", summary="max")
    if args.upload_code:
        logger.experiment.log_code(root=sys.argv[0])
        logger.experiment.log_code(root="./scripts", name="scripts")

    callbacks = {
            "pl_tqdm": PlTqdm(),
            "lr_monitor": LearningRateMonitor(logging_interval="step"),
            "prediction_writer": PlPredictionWriter(root=args.prediction_path)
    }
    if args.time_limit:
        callbacks["Timer"] = Timer(duration={"hours": args.time_limit}, interval="epoch")
    if args.enable_checkpointing:
        callbacks["checkpoint"] = ModelCheckpoint(
                monitor="val_acc",
                mode="max",
                save_last=True,
                save_top_k=args.save_top_k,
                dirpath=args.checkpoint_path,
                filename=args.framework + "-{epoch:02d}-{val_loss:.4f}-{val_acc:.4f}"
        )

    model = PlModule(args)
    datamodule = PlDataModule(args)
    trainer = get_trainer(args=args, logger=logger, callbacks=list(callbacks.values()))

    torch.use_deterministic_algorithms(mode=True, warn_only=True)
    trainer.fit(model=model, ckpt_path=args.resume_ckpt_path, datamodule=datamodule)

    if args.enable_checkpointing:
        trainer.predict(ckpt_path=callbacks["checkpoint"].best_model_path, datamodule=datamodule)
        scores = Evaluator(args).score_all(prediction_path=callbacks["prediction_writer"].last_save_path)
        if get_rank() == 0:
            print(f"Results:\n{json.dumps(scores, indent=4, sort_keys=True)}\n")
            logger.experiment.summary.update({f"performance/{name}": score["F1"] for name, score in scores.items()})
