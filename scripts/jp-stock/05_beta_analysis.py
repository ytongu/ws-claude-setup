#!/usr/bin/env python3
"""
目的: 仮想ポートフォリオの値動きが市場で説明できる部分（beta）と、
      説明できない部分（alpha・個別要因）を分離する。

入力:
  - portfolio/history.csv  （04_track_portfolio.py が market_pct 列まで記録する）

出力:
  - 標準出力に回帰結果
  - --out を指定すると {out}/beta_{TS}.md に書き出す

なぜこれを作ったか:
  日次の騰落を目で追っていると、「上げで取れない」「下げで粘る」といった
  パターンが見えてしまう。実際このポートフォリオでは観測54営業日のあいだに
  4つの仮説を立てて4つとも外した。いずれも以下が原因だった。

    - tail だけを抜き出して表にした（上げ幅の大きい8日だけ見て113%の全体像を見落とす）
    - 数日の連続を傾向と呼んだ（翌日に反転する）

  日次全件の回帰なら、この2つの罠をどちらも踏まない。
  beta が1に近く alpha がゼロなら、どんなパターンが見えても
  「市場に連動しているだけ」が答えである。

使い方:
  python 05_beta_analysis.py
  python 05_beta_analysis.py --out ~/projects/20260803_jpstock/results/05_beta
"""

from __future__ import annotations

import argparse
import csv
import statistics as st
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent
HIST = ROOT / "portfolio" / "history.csv"


def load(path: Path) -> list[tuple[str, float, float]]:
    """(日付, PF累計%, 市場累計%) を日付順で返す。market_pct が空の行は捨てる。"""
    out = []
    with path.open(encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            try:
                out.append((r["asof"], float(r["pl_pct"]), float(r["market_pct"])))
            except (TypeError, ValueError, KeyError):
                continue
    return sorted(out)


def ols(X: list[float], Y: list[float]) -> dict:
    n = len(X)
    mx, my = st.fmean(X), st.fmean(Y)
    sxx = sum((x - mx) ** 2 for x in X)
    beta = sum((x - mx) * (y - my) for x, y in zip(X, Y)) / sxx
    alpha = my - beta * mx
    res = [y - (alpha + beta * x) for x, y in zip(X, Y)]
    sse = sum(r * r for r in res)
    sst = sum((y - my) ** 2 for y in Y)
    se_beta = (sse / (n - 2) / sxx) ** 0.5
    se_alpha = (sse / (n - 2) * (1 / n + mx * mx / sxx)) ** 0.5
    return {"n": n, "beta": beta, "se_beta": se_beta, "alpha": alpha,
            "se_alpha": se_alpha, "r2": 1 - sse / sst if sst else float("nan")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--history", default=str(HIST))
    ap.add_argument("--out", help="レポートの出力先ディレクトリ")
    args = ap.parse_args()

    rows = load(Path(args.history))
    if len(rows) < 5:
        print("データ不足（market_pct 列が埋まった行が5日未満）")
        return 1

    # 累計%から日次変化を作る
    X = [rows[i][2] - rows[i - 1][2] for i in range(1, len(rows))]
    Y = [rows[i][1] - rows[i - 1][1] for i in range(1, len(rows))]
    f = ols(X, Y)
    ex = [y - x for x, y in zip(X, Y)]
    t_ex = st.fmean(ex) / (st.stdev(ex) / len(ex) ** 0.5)

    up = [(x, y) for x, y in zip(X, Y) if x > 0]
    dn = [(x, y) for x, y in zip(X, Y) if x < 0]

    L = [f"# ベータ分解 {rows[0][0]} 〜 {rows[-1][0]}", "",
         f"観測 {f['n']} 営業日　PF累計 {rows[-1][1]:+.2f}% / "
         f"市場累計 {rows[-1][2]:+.2f}% / 差 {rows[-1][1] - rows[-1][2]:+.2f}pt", "",
         "## 回帰  PF日次 = alpha + beta × 市場日次", "",
         f"- **beta  = {f['beta']:.3f} ± {f['se_beta']:.3f}**　"
         f"(beta=1 との差 t = {(f['beta'] - 1) / f['se_beta']:+.2f})",
         f"- alpha = {f['alpha']:+.3f} ± {f['se_alpha']:.3f} pt/日　"
         f"(t = {f['alpha'] / f['se_alpha']:+.2f})",
         f"- R² = {f['r2']:.3f} → 値動きの {f['r2'] * 100:.0f}% が市場で説明でき、"
         f"残り {100 - f['r2'] * 100:.0f}% は個別要因", ""]

    L += ["## 上げ日・下げ日の追随率", "",
          "| 局面 | 日数 | 市場計 | PF計 | 追随率 |", "|---|---:|---:|---:|---:|"]
    for nm, s in [("上げ日", up), ("下げ日", dn)]:
        if not s:
            continue
        sx, sy = sum(x for x, _ in s), sum(y for _, y in s)
        L.append(f"| {nm} | {len(s)} | {sx:+.2f}pt | {sy:+.2f}pt | {sy / sx * 100:.0f}% |")

    L += ["", "## 超過リターン", "",
          f"日次超過の平均 {st.fmean(ex):+.3f}pt、標準偏差 {st.stdev(ex):.3f}pt、"
          f"t = {t_ex:+.2f}", "",
          "## 読み方", "",
          "beta が1と区別できず alpha もゼロなら、日々の値動きに見えるパターンは"
          "市場連動の裏返しにすぎない。上げ日・下げ日の追随率が両方とも100%近いなら、"
          "「上げで取れない」「下げで粘る」といった非対称は存在しない。",
          "",
          "R² が低い（0.5前後）場合、値動きの半分が個別要因であることを意味する。"
          "銘柄数が少ないほどこの比率は上がり、選別の効果は個別要因のノイズに埋もれる。"]

    print("\n".join(L))
    if args.out:
        d = Path(args.out).expanduser()
        d.mkdir(parents=True, exist_ok=True)
        p = d / f"beta_{datetime.now():%Y%m%d_%H%M%S}.md"
        p.write_text("\n".join(L) + "\n", encoding="utf-8")
        print(f"\n  {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
