"""Failures against needless switches, mean over the five push families, new safe policies."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
# (failed %, needless %) means over the 5 families, from <robot>_v2.md (this folder)
data = {
 "Go2": {"ours": [("walking rule", 16.3, 18.6), ("0.5", 15.6, 19.6), ("0.9", 10.7, 31.4), ("0.95", 10.8, 37.2), ("0.99", 7.3, 49.6)],
         "abs": [("0.05", 8.0, 19.6), ("0.3", 6.1, 29.8)],
         "td": [("V_thr+L", 9.7, 76.6), ("walking rule", 40.2, 17.4)],
         "mc": [("0.8+L", 45.3, 1.8)],
         "walk": (48.7, 0), "safe": (1.5, 100)},
 "Aliengo": {"ours": [("0.5", 16.1, 47.4), ("walking rule", 13.5, 58.4), ("0.9", 9.7, 77.0), ("0.95", 7.9, 84.4), ("0.99", 3.6, 99.6)],
         "abs": [("0.05", 16.9, 33.0), ("0.3", 15.4, 42.6)],
         "td": [("V_thr+L", 29.8, 47.4), ("walking rule", 45.7, 13.4)],
         "mc": [("0.8+L", 49.5, 4.2)],
         "walk": (51.8, 0), "safe": (3.2, 100)}}
style = {"ours": ("#2a78d6", "o", "-", "Ours, Q(o, 1) ≥ τ"), "abs": ("#eb6834", "s", "-", "ABS, −V_RA ≥ τ"),
         "td": ("#1baf7a", "^", "", "Paper V(o), TD(0)"), "mc": ("#e87ba4", "D", "", "Paper V(o), Monte Carlo")}
OFFSETS = {("Go2", "ours", "walking rule"): (-6, 4), ("Go2", "ours", "0.5"): (4, 6),
           ("Go2", "ours", "0.9"): (-6, 6), ("Aliengo", "ours", "0.5"): (2, 7),
           ("Go2", "mc", "0.8+L"): (8, -4), ("Aliengo", "mc", "0.8+L"): (8, -4), ("Aliengo", "ours", "0.99"): (-8, 6)}
ink, muted = "#2b2b2a", "#7a7974"
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
for ax, (robot, d) in zip(axes, data.items()):
    for k, m, label in (("walk", "x", "Walking policy alone"), ("safe", "P", "Safe policy alone, hand-over at push start")):
        y, x = d[k]
        ax.plot(x, y, m, color=muted, ms=9, mew=2 if m == "x" else 1, ls="none", label=label)
    for k, (c, m, ls, label) in style.items():
        pts = sorted(d[k], key=lambda p: p[2])
        ax.plot([p[2] for p in pts], [p[1] for p in pts], marker=m, ls=ls or "none", color=c, lw=2, ms=8,
                mec="white", mew=1.5, label=label)
        for name, y, x in pts:
            off = OFFSETS.get((robot, k, name), (5, 5) if k != "abs" else (5, -13))
            ax.annotate(name, (x, y), textcoords="offset points", xytext=off, fontsize=7.5, color=ink,
                        ha="right" if off[0] < 0 else "left")
    ax.set_title(robot, fontsize=11, color=ink, loc="left")
    ax.set_xlabel("needless switches, % of the pushes the walking policy survives", fontsize=9, color=ink)
    ax.set_xlim(-5, 108); ax.set_ylim(-2, 58)
    ax.grid(alpha=0.25); [s.set_visible(False) for s in (ax.spines["top"], ax.spines["right"])]
    ax.tick_params(colors=muted, labelsize=8)
axes[0].set_ylabel("failed, % of pushes", fontsize=9, color=ink)
axes[0].legend(fontsize=8, frameon=False, loc="upper right", bbox_to_anchor=(1.0, 0.97))
fig.suptitle("Failures against needless switches, mean of 5 push families × 200 pushes (new safe policies). Lower left is better.",
             fontsize=9.5, color=ink, x=0.01, ha="left")
fig.tight_layout()
fig.savefig("tradeoff_new_safe.png", dpi=160)
