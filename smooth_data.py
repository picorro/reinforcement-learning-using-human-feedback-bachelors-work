import argparse
import pandas as pd


def smooth_csv(input_file, output_file, alpha):
    df = pd.read_csv(input_file)
    smoothed_df = df.ewm(alpha=(1 - alpha)).mean()
    smoothed_df.to_csv(output_file, index=False)


def main(input_file, output_file, alpha):
    smooth_csv(input_file, output_file, alpha)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smooth CSV data.")
    parser.add_argument("input_file", help="Path to input CSV file.")
    parser.add_argument("output_file", help="Path to output CSV file.")
    parser.add_argument(
        "-a",
        "--alpha",
        type=float,
        default=0.97,
        help="Smoothing factor (default: 0.97)",
    )
    args = parser.parse_args()

    main(args.input_file, args.output_file, args.alpha)
