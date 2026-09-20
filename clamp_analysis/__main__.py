"""Command-line entry points; expensive fits are always explicit."""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Reproduce the analyses and figures in thesis Chapter 4."
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for command, help_text in [
        (
            "reproduce",
            "Regenerate participant summaries, statistics, and all six figures using saved fits.",
        ),
        ("summaries", "Recompute participant summaries from formatted trials."),
        ("statistics", "Run the thesis inferential tests, effect sizes, and bootstrap intervals."),
        ("figures", "Draw all six numbered figures from existing summaries and fits."),
        ("literature", "Rebuild the literature summaries from the included source datasets."),
    ]:
        sub.add_parser(command, help=help_text)
    sub.add_parser(
        "restore-data",
        help="Restore the five bundled large inputs, without third-party dependencies.",
    )
    preprocess = sub.add_parser("preprocess", help="Rebuild formatted trials from raw data.")
    preprocess.add_argument("--output", type=Path)
    trackers = sub.add_parser(
        "trackers", help="Rebuild first-bout radial analyses; raw tracker extraction is optional."
    )
    trackers.add_argument(
        "--rebuild-cache", action="store_true", help="Re-read the 8.85 GB tracker file."
    )
    fit = sub.add_parser("fit", help="Refit all three joint-device models; may take hours.")
    fit.add_argument(
        "--output", type=Path, required=True, help="Separate output directory for the new fits."
    )
    fit.add_argument("--cores", type=int, default=1)
    fit.add_argument("--restarts", type=int, default=20)
    fit.add_argument("--polish", type=int, default=10)
    fit.add_argument("--max-evals", type=int, default=6000)
    recovery = sub.add_parser(
        "recover", help="Rerun model and parameter recovery; publication runs are expensive."
    )
    recovery.add_argument("--output", type=Path, required=True)
    recovery.add_argument("--cores", type=int, default=1)
    recovery.add_argument(
        "--preset", choices=["smoke", "working", "publication"], default="publication"
    )
    recovery.add_argument("--restarts", type=int)
    recovery.add_argument("--max-evals", type=int)
    args = parser.parse_args()

    # These workloads parallelize independent restarts, not small BLAS products.
    import os

    for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(variable, "1")

    # A fresh Git checkout contains compact archives in place of the large inputs.
    from .data_archives import restore_data

    if args.command == "restore-data":
        for item in restore_data():
            print(f"{item['path']}: {item['status']}")
        return
    if args.command == "preprocess":
        required_inputs = ["data/raw/trialsV2.csv"]
    elif args.command == "literature":
        required_inputs = [
            f"data/literature/morehead_raw/data_MultiSizeClamp_{group}.mat"
            for group in ("1237", "15304560", "7080100110")
        ]
    else:
        required_inputs = ["data/processed/formattedData.csv"]
    restore_data(files=required_inputs)
    if args.command in ("reproduce", "summaries"):
        from .behavior.pipeline import run

        run()
    if args.command in ("reproduce", "statistics"):
        from .statistics import chapter_tests, clamp_tests

        print("Running clamp-level tests...", flush=True)
        clamp_tests.run_all_analyses()
        print("Running chapter tests and bootstrap intervals...", flush=True)
        chapter_tests.run_all_analyses()
    if args.command in ("reproduce", "figures"):
        from .figures.pipeline import run

        run()
    if args.command == "preprocess":
        from .behavior.preprocess import run

        run(args.output)
    if args.command == "trackers":
        from .kinematics.ballistic import load_ballistic_updates
        from .kinematics.radial import load_calibrated_tables

        load_ballistic_updates(force_rebuild=args.rebuild_cache)
        summary, _ = load_calibrated_tables(force_rebuild=True)
        print(summary.to_string(index=False))
    if args.command == "literature":
        from .literature import morehead, zhang

        zhang.main()
        morehead.main()
    if args.command == "fit":
        from .workflows.fit import run

        run(
            output_dir=args.output,
            cores=args.cores,
            restarts=args.restarts,
            polish=args.polish,
            max_evals=args.max_evals,
        )
    if args.command == "recover":
        from .workflows.recovery import run

        run(
            output_dir=args.output,
            cores=args.cores,
            preset=args.preset,
            restarts=args.restarts,
            max_evals=args.max_evals,
        )


if __name__ == "__main__":
    main()
