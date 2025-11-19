import argparse
import os
import torch

from config import load_config
from train.train import run_cv
from inference.predict import load_model, predict_one, predict_folder


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")

    # Train command
    p_train = sub.add_parser("train")
    p_train.add_argument("--config", type=str, default="config/config.yaml")
    p_train.add_argument("--fold", type=int, default=-1)

    # Predict command
    p_pred = sub.add_parser("predict")
    p_pred.add_argument("--config", type=str, default="config/config.yaml")
    p_pred.add_argument("--checkpoint", type=str, required=True)
    p_pred.add_argument("--file", type=str)
    p_pred.add_argument("--folder", type=str)
    p_pred.add_argument("--save", type=str)

    args = parser.parse_args()

    if args.command == "train":
        cfg = load_config(args.config)
        csv_path = cfg["DATA"]["CSV_PATH"]
        run_cv(cfg, csv_path=csv_path, fold_to_run=args.fold)

    elif args.command == "predict":
        cfg = load_config(args.config)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = load_model(args.checkpoint, device)

        if args.file:
            prob, label = predict_one(model, args.file, device, cfg)
            print("File:", args.file)
            print("Probability:", prob)
            print("Label:", label)

        if args.folder:
            df = predict_folder(model, args.folder, device, cfg, save_csv=args.save)
            print(df.head())

    else:
        print("Use: main.py train | main.py predict")


if __name__ == "__main__":
    main()
