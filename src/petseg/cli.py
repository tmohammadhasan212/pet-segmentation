import argparse
import json


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="petseg",
        description="Oxford-IIIT Pet segmentation experiments and cutouts",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser(
        "prepare", help="Download optionally, audit and lock manifests/statistics"
    )
    p.add_argument("--data-root", default="data/oxford-pets")
    p.add_argument("--manifests", default="data/manifests")
    p.add_argument("--download", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-fraction", type=float, default=0.2)
    p = sub.add_parser("audit")
    p.add_argument("--data-root", default="data/oxford-pets")
    p = sub.add_parser("train")
    p.add_argument("--config", required=True)
    p.add_argument("--output", required=True)
    p = sub.add_parser("evaluate")
    p.add_argument("--checkpoint", help="Omit for central ellipse baseline")
    p.add_argument("--data-root", default="data/oxford-pets")
    p.add_argument("--manifests", default="data/manifests")
    p.add_argument("--output", required=True)
    p.add_argument("--split", choices=["val", "test"], default="test")
    p.add_argument(
        "--native", action="store_true", help="Score at original annotation dimensions"
    )
    p.add_argument(
        "--size",
        type=int,
        default=256,
        help="Baseline only; model uses checkpoint size",
    )
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p = sub.add_parser(
        "calibrate", help="Select inference settings on validation images only"
    )
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--export", dest="export_path")
    p.add_argument("--sizes", type=int, nargs="+")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p = sub.add_parser("predict")
    p.add_argument("image")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--threshold", type=float)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p = sub.add_parser("demo")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--port", type=int, default=7860)
    p = sub.add_parser("report")
    p.add_argument("evaluations", nargs="+")
    p.add_argument("--output", default="reports/generated/results.md")
    p.add_argument("--regenerate", action="store_true")
    p = sub.add_parser("gallery")
    p.add_argument("evaluation")
    p.add_argument(
        "--ids", help="Manifest of curated example IDs; otherwise 20 worst IoU"
    )
    args = parser.parse_args(argv)
    if args.command == "prepare":
        from .data import download, prepare

        if args.download:
            download(args.data_root)
        print(
            json.dumps(
                prepare(args.data_root, args.manifests, args.seed, args.val_fraction),
                indent=2,
            )
        )
    elif args.command == "audit":
        from .data import audit

        print(json.dumps(audit(args.data_root), indent=2))
    elif args.command == "train":
        from .training import train

        print(train(args.config, args.output))
    elif args.command == "evaluate":
        from .evaluation import evaluate

        print(
            json.dumps(
                evaluate(
                    args.data_root,
                    args.manifests,
                    args.output,
                    args.checkpoint,
                    args.split,
                    args.device,
                    args.size,
                    args.native,
                ),
                indent=2,
            )
        )
    elif args.command == "predict":
        from .inference import Predictor, export

        result = Predictor(args.checkpoint, args.device).predict(
            args.image, args.threshold
        )
        export(result, args.output)
        print(f"Saved {args.output}/cutout.png ({result['inference_ms']:.1f} ms)")
    elif args.command == "calibrate":
        from .calibration import calibrate

        report = calibrate(
            args.checkpoint, args.output, args.export_path, args.sizes, args.device
        )
        print(json.dumps(report["selected"], indent=2))
    elif args.command == "demo":
        from .demo import create_demo

        create_demo(args.checkpoint, args.device).launch(
            server_name="127.0.0.1",
            server_port=args.port,
            share=False,
            allowed_paths=[],
        )
    elif args.command == "report":
        from .evaluation import comparison, regenerate

        if args.regenerate:
            for directory in args.evaluations:
                regenerate(directory)
        print(comparison(args.evaluations, args.output))
    elif args.command == "gallery":
        from .data import read_ids
        from .evaluation import gallery

        gallery(args.evaluation, read_ids(args.ids) if args.ids else None)


if __name__ == "__main__":
    main()
