"""
DAG of the means-tested two-asset HANK model, drawn in the style of
Auclert, Bardoczy, Rognlie, Straub (2021), Figure "DAG of the two-asset HANK model".

Block mapping (paper name <- code in the ikc replication):
  prod.     <- labor, investment         (inside labor_to_investment_..._combined [0])
  NKPC-p    <- pricing_solved            (inside [0])
  equity    <- dividend, val_firm_solved (inside [0])
  mon pol   <- taylor [2]
  finance   <- share_bond [1] + finance [3]
  fiscal    <- fiscalrule_mt [5]
  HA        <- Two Asset Model_mt [6]
  H1        <- target 'fisher'   (computed in finance)
  H2        <- target 'goods_mkt' (mkt_clearing [7])
  H3        <- target 'wnkpc'    (wage_inflation [4] + union [8])
Variable mapping: Tr -> tau, J_end_period -> p, UCE -> U, rpost -> r^post.
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, FancyArrowPatch

plt.rcParams.update({
    "font.family": "cmr10",
    "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True,
})

GREEN, BROWN, PINK = "#2E7D32", "#A07850", "#D07FB8"

# ------------------------------------------------------------------ nodes
# kind: 'c' circle (radius r) or 'b' box (w, h)
nodes = {
    "EX":    dict(kind="b", xy=(1.35, 2.55), size=(2.05, 1.0), color="black",
                  text=r"exog. $\mathbf{B},\,\tau$" + "\n" + r"unknowns $\mathbf{Y},\,\mathbf{w},\,\mathbf{r}$"),
    "PROD":  dict(kind="c", xy=(3.45, 4.15), size=0.42, color=BROWN, text="prod."),
    "NKPCP": dict(kind="c", xy=(5.00, 4.15), size=0.42, color=GREEN, text=r"NKPC-$p$"),
    "FIN":   dict(kind="c", xy=(6.55, 4.15), size=0.42, color=GREEN, text="finance"),
    "HA":    dict(kind="c", xy=(7.10, 2.95), size=0.42, color=PINK, text="HA"),
    "MP":    dict(kind="c", xy=(6.55, 1.45), size=0.42, color=GREEN, text="mon pol"),
    "FISC":  dict(kind="c", xy=(3.45, 1.30), size=0.42, color=GREEN, text="fiscal"),
    "EQ":    dict(kind="c", xy=(5.00, 1.30), size=0.42, color=GREEN, text="equity"),
    "H1":    dict(kind="b", xy=(9.45, 4.55), size=(1.45, 0.72), color="black",
                  text="Fisher eq.\n" + r"$\mathbf{(H_1)}$"),
    "H2":    dict(kind="b", xy=(9.45, 2.60), size=(1.45, 0.90), color="black",
                  text="goods mkt.\nclearing " + r"$\mathbf{(H_2)}$"),
    "H3":    dict(kind="b", xy=(9.45, 0.80), size=(1.45, 0.72), color="black",
                  text=r"NKPC-$w$" + "\n" + r"$\mathbf{(H_3)}$"),
}

# --------------------------------------------------------------- label text
SPECIAL = {
    "pi":   r"\pi",
    "tau":  r"\tau",
    "U":    r"\mathcal{U}",
    "rpost": r"\mathbf{r}^{\mathbf{post}}",
    "padj": r"\mathbf{p}^{\mathbf{adj}}",
    "kadj": r"\mathbf{k}^{\mathbf{adj}}",
}
def L(*names):
    parts = [SPECIAL.get(n, r"\mathbf{%s}" % n) for n in names]
    return "$" + r",\,".join(parts) + "$"

# ------------------------------------------------------------------- edges
# (src, dst, labels, rad, t, (dx, dy), start_anchor, end_anchor)
E = [
    ("EX", "PROD",  L("Y", "w", "r"),                 0.00, 0.50, (0, 0)),
    ("EX", "NKPCP", L("r", "Y"),                      -0.30, 0.50, (0, 0)),
    ("EX", "FIN",   L("B", "r"),                      -0.22, 0.50, (0, 0)),
    ("EX", "EQ",    L("Y", "w", "r"),                  0.42, 0.50, (0, 0)),
    ("EX", "FISC",  L("B", "tau", "r", "w"),           0.00, 0.50, (0, 0)),
    ("EX", "MP",    L("r"),                            0.00, 0.50, (0, 0)),
    ("EX", "HA",    L("tau"),                          0.00, 0.50, (0, 0)),
    ("EX", "H2",    L("Y", "B"),                       0.00, 0.50, (0, 0)),
    ("EX", "H3",    L("w"),                            0.40, 0.50, (0, 0)),

    ("PROD", "NKPCP", L("mc"),                         0.00, 0.50, (0, 0)),
    ("PROD", "FISC",  L("N"),                          0.00, 0.50, (0, 0)),
    ("PROD", "EQ",    L("N", "K"),                    -0.15, 0.50, (0, 0)),
    ("PROD", "H3",    L("N"),                          0.00, 0.50, (0, 0)),

    ("NKPCP", "FIN", L("pi"),                          0.00, 0.50, (0, 0)),
    ("NKPCP", "EQ",  L("pi"),                          0.00, 0.50, (0, 0)),
    ("NKPCP", "MP",  L("pi"),                          0.00, 0.50, (0, 0)),
    ("NKPCP", "H3",  L("pi"),                          0.00, 0.50, (0, 0)),

    ("EQ", "FIN", L("J", "p"),                         0.00, 0.50, (0, 0)),
    ("EQ", "H2",  L("p", "I", "padj", "kadj"),         0.00, 0.50, (0, 0)),
    ("MP", "FIN", L("i"),                              0.00, 0.50, (0, 0)),
    ("FIN", "HA", L("rpost"),                          0.00, 0.50, (0, 0)),
    ("FIN", "H1", L("i", "r", "pi"),                   0.00, 0.50, (0, 0)),

    ("FISC", "HA", L("Z"),                             0.00, 0.50, (0, 0)),
    ("FISC", "H3", L("Z"),                             0.00, 0.50, (0, 0)),
    ("FISC", "H2", L("G"),                             0.00, 0.50, (0, 0)),
    ("HA", "H2",   L("A", "C"),                        0.00, 0.50, (0, 0)),
    ("HA", "H3",   L("U"),                             0.00, 0.50, (0, 0)),
]

# --------------------------------------------------------------------- draw
def draw(path_png=None, path_pdf=None, figsize=(13.2, 6.6)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0.15, 10.5)
    ax.set_ylim(0.1, 5.25)
    ax.set_aspect("equal")
    ax.axis("off")

    patches = {}
    for k, n in nodes.items():
        x, y = n["xy"]
        if n["kind"] == "c":
            p = Circle((x, y), n["size"], fc="white", ec=n["color"], lw=1.1, zorder=2)
            ax.add_patch(p)
            ax.text(x, y, n["text"], color=n["color"], ha="center", va="center",
                    fontsize=12.5, zorder=3)
        else:
            w, h = n["size"]
            p = Rectangle((x - w / 2, y - h / 2), w, h, fc="white", ec="black", lw=1.0, zorder=2)
            ax.add_patch(p)
            ax.text(x, y, n["text"], color="black", ha="center", va="center",
                    fontsize=12.5, zorder=3, linespacing=1.35)
        patches[k] = p

    for (s, d, lab, rad, t, off) in E:
        A = np.array(nodes[s]["xy"], float)
        B = np.array(nodes[d]["xy"], float)
        arr = FancyArrowPatch(A, B, connectionstyle=f"arc3,rad={rad}",
                              arrowstyle="-|>", mutation_scale=11, lw=0.8, color="black",
                              patchA=patches[s], patchB=patches[d],
                              shrinkA=0, shrinkB=1.0, zorder=1)
        ax.add_patch(arr)
        # label position on the quadratic Bezier A -> c -> B
        mid = (A + B) / 2
        dx, dy = B - A
        c = mid + rad * np.array([dy, -dx])
        P = (1 - t) ** 2 * A + 2 * (1 - t) * t * c + t ** 2 * B
        ax.text(P[0] + off[0], P[1] + off[1], lab, ha="center", va="center", fontsize=10.5,
                zorder=4, bbox=dict(fc="white", ec="none", pad=1.2))

    fig.tight_layout(pad=0.2)
    if path_png:
        fig.savefig(path_png, dpi=220, bbox_inches="tight", facecolor="white")
    if path_pdf:
        fig.savefig(path_pdf, bbox_inches="tight", facecolor="white")
    return fig


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "dag_hank_mt"
    draw(out + ".png", out + ".pdf")
