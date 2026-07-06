import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from thesis.config import LABEL_MAP, RESULTS_DIR
from thesis.dataset import split_data


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--embeddings",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--label-col",
        default="label",
        help="target column in the embedding table (default 'label', the A/B/C classes).",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=None,
        help="optional parquet of external labels keyed by crater_id to join in.",
    )
    parser.add_argument(
        "--classes",
        default=None,
        help="comma-separated class order, e.g. 'A,B,C' (default: LABEL_MAP).",
    )
    parser.add_argument(
        "--restrict-to",
        type=Path,
        default=None,
        help="optional parquet with a crater_id column; keep only those craters.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="output dir (default: RESULTS_DIR/<embeddings-stem>[-<label-col>]).",
    )
    return parser.parse_args()


def resolve_label_map(args):
    """Return (class_names, label_to_int); defaults to LABEL_MAP."""
    if args.classes:
        names = [c.strip() for c in args.classes.split(",")]
    elif args.label_col == "label":
        names = list(LABEL_MAP.keys())
    else:
        raise SystemExit(
            f"--label-col '{args.label_col}' needs --classes (e.g. --classes 1,2,3,4)"
        )
    return names, {name: i for i, name in enumerate(names)}


def prepare_df(args) -> pd.DataFrame:
    df = pd.read_parquet(args.embeddings)

    if args.labels is not None:
        ext = pd.read_parquet(args.labels)
        keep = ["crater_id", args.label_col]
        df = df.drop(columns=[c for c in (args.label_col,) if c in df.columns])
        df = df.merge(ext[keep], on="crater_id", how="inner")
        print(f"joined {args.labels.name}: {len(df)} craters carry '{args.label_col}'")

    if args.restrict_to is not None:
        ids = set(pd.read_parquet(args.restrict_to)["crater_id"].tolist())
        before = len(df)
        df = df[df["crater_id"].isin(ids)].reset_index(drop=True)
        print(f"restricted to {args.restrict_to.name}: {before} -> {len(df)} craters")

    df = df[df[args.label_col].notna()].reset_index(drop=True)
    return df


def to_xy(df: pd.DataFrame, label_col: str, label_to_int: dict):
    x = np.stack(df["embeddings"].values)
    # labels may be ints or strings; map via string key
    y = df[label_col].apply(lambda v: label_to_int[str(v).strip()]).to_numpy()
    return x, y


def report_to_df(report: dict) -> pd.DataFrame:
    """classification_report dict -> per-row DataFrame (drops scalar 'accuracy')."""
    rows = {k: v for k, v in report.items() if isinstance(v, dict)}
    df = pd.DataFrame(rows).T
    df.index.name = "class"
    return df.reset_index()


def evaluate(clf, x, y, split_name: str, class_names, out_dir: Path) -> dict:
    y_pred = clf.predict(x)
    macro_f1 = f1_score(y, y_pred, average="macro")

    n_classes = len(class_names)
    report = classification_report(
        y, y_pred, labels=list(range(n_classes)), target_names=class_names,
        digits=4, output_dict=True, zero_division=0,
    )
    cm = confusion_matrix(y, y_pred, labels=list(range(n_classes)))

    print(f"\n── {split_name} ──")
    print(f"macro-F1: {macro_f1:.4f}")
    print(classification_report(
        y, y_pred, labels=list(range(n_classes)), target_names=class_names,
        digits=4, zero_division=0,
    ))
    print("confusion matrix (rows=true, cols=pred):")
    print(cm)

    prefix = split_name.lower()
    report_df = report_to_df(report)
    report_df.to_csv(out_dir / f"{prefix}_metrics.csv", index=False)

    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_df.index.name = "true\\pred"
    cm_df.to_csv(out_dir / f"{prefix}_confusion_matrix.csv")

    return {"macro_f1": float(macro_f1), "report": report}


def main():
    args = parse_args()
    class_names, label_to_int = resolve_label_map(args)

    df = prepare_df(args)
    train_df, val_df, test_df = split_data(df)

    x_train, y_train = to_xy(train_df, args.label_col, label_to_int)
    x_val, y_val = to_xy(val_df, args.label_col, label_to_int)
    x_test, y_test = to_xy(test_df, args.label_col, label_to_int)

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_val_scaled = scaler.transform(x_val)
    x_test_scaled = scaler.transform(x_test)

    clf = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        n_jobs=-1,
    )

    print("fitting logistic regression...")
    clf.fit(x_train_scaled, y_train)
    print("done")

    if args.out_dir is not None:
        out_dir = args.out_dir
    else:
        suffix = "" if args.label_col == "label" else f"-{args.label_col}"
        out_dir = RESULTS_DIR / f"{args.embeddings.stem}{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {}
    for split_name, x, y in [
        ("VAL", x_val_scaled, y_val),
        ("TEST", x_test_scaled, y_test),
    ]:
        summary[split_name] = evaluate(clf, x, y, split_name, class_names, out_dir)

    run = {
        "embeddings": str(args.embeddings),
        "checkpoint": args.embeddings.stem,
        "label_col": args.label_col,
        "classes": class_names,
        "labels_source": str(args.labels) if args.labels else None,
        "restrict_to": str(args.restrict_to) if args.restrict_to else None,
        "n_train": int(len(y_train)),
        "n_val": int(len(y_val)),
        "n_test": int(len(y_test)),
        "classifier": "LogisticRegression(max_iter=1000, class_weight='balanced')",
        "val_macro_f1": summary["VAL"]["macro_f1"],
        "test_macro_f1": summary["TEST"]["macro_f1"],
        "val_report": summary["VAL"]["report"],
        "test_report": summary["TEST"]["report"],
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(run, f, indent=2)

    print(f"\nsaved metrics to {out_dir}/")
    print("  val_metrics.csv, test_metrics.csv  (per-class P/R/F1 + macro/weighted avg)")
    print("  val_confusion_matrix.csv, test_confusion_matrix.csv")
    print("  summary.json")


if __name__ == "__main__":
    main()
