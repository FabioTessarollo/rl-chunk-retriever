import argparse

from config import get_config, setup_logging
from etl.chunk_and_label import chunk_and_label
from etl.embed import embed
from etl.extract import extract
from retrieval.analysis import analyze
from retrieval.cosine_similarity import cos_sim
from retrieval.test import test
from retrieval.train import train

STAGES = ["extract", "chunk", "embed", "cos-sim", "train", "test", "analyze"]


def run_stage(stage, cfg, dataset):
    dispatch = {
        "extract": lambda: extract(dataset, cfg),
        "chunk": lambda: chunk_and_label(dataset, cfg),
        "embed": lambda: embed(dataset, cfg),
        "cos-sim": lambda: cos_sim(cfg),
        "train": lambda: train(cfg),
        "test": lambda: test(cfg),
        "analyze": lambda: analyze(cfg),
    }
    dispatch[stage]()


def main():
    parser = argparse.ArgumentParser(description="RL Chunks Retriever Pipeline")
    parser.add_argument("--config", default=None, help="Path to config YAML file")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--log-file", default=None, help="Optional log file path")

    sub = parser.add_subparsers(dest="command")

    # ETL stages that need --dataset
    for name in ("extract", "chunk", "embed"):
        p = sub.add_parser(name)
        p.add_argument("--dataset", required=True, choices=["train", "test"])

    # Stages without --dataset
    for name in ("cos-sim", "train", "test", "analyze"):
        sub.add_parser(name)

    # Pipeline: run all stages
    pipe = sub.add_parser("pipeline")
    pipe.add_argument("--from", dest="from_stage", default=None, choices=STAGES, help="Resume pipeline from this stage")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return

    cfg = get_config(args.config) if args.config else get_config()

    log_file = args.log_file
    if log_file is None and (
        args.command == "train"
        or (
            args.command == "pipeline"
            and "train" in STAGES[(STAGES.index(args.from_stage) if args.from_stage else 0) :]
        )
    ):
        log_file = "rl_training.log"

    setup_logging(args.log_level, log_file)

    if args.command == "pipeline":
        start = STAGES.index(args.from_stage) if args.from_stage else 0
        for stage in STAGES[start:]:
            dataset = "train"  # default for ETL stages in pipeline mode
            if stage in ("extract", "chunk", "embed"):
                # Run both train and test for ETL stages
                for ds in ("train", "test"):
                    run_stage(stage, cfg, ds)
            else:
                run_stage(stage, cfg, None)
    else:
        dataset = getattr(args, "dataset", None)
        run_stage(args.command, cfg, dataset)


if __name__ == "__main__":
    main()
