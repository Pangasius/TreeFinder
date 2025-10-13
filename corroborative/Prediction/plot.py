import matplotlib.pyplot as plt
import numpy as np
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--factor", type=int, default=1, help="factor to split the input"
    )
    args = parser.parse_args()

    path = str(8000 // args.factor) + "_output.txt"
    # read csv
    data = np.loadtxt(
        path, delimiter="\t", skiprows=1, converters={3: lambda s: eval(s)}
    )

    # logprob on the x axis
    # sublogprob on the y axis
    # one color for ContainedHint == True, another for False

    figure = plt.figure(figsize=(10, 8))

    # add regression line for both
    x = np.exp(data[:, 4])
    y = np.exp(data[:, 5])
    contained_hint = data[:, 3].astype(bool)

    plt.scatter(x, y, s=8, c=contained_hint, cmap="coolwarm", alpha=0.5)

    # trendline for each group
    colors = ["#6773C1", "#BA2643"]  # coolwarm colormap colors
    for i, hint in enumerate([False, True]):
        mask = contained_hint == hint
        coeffs = np.polyfit(x[mask], y[mask], 1)
        trendline = np.polyval(coeffs, x[mask])
        plt.plot(x[mask], trendline, color=colors[i], label=f"ContainedHint={hint}", linewidth=2)

    plt.ylabel("P(A|C')", fontsize=22)
    plt.xlabel("P(A|C)", fontsize=22)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.colorbar(label="ContainedHint", ticks=[0, 1], format='%d')
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.title("P(A|C) vs P(A|C')", fontsize=24)

    figure.savefig(str(8000 // args.factor) + "_chart.pdf")
