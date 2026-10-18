"""data/ の CSV から指標を計算し、要約（JSON）とチャート付きHTMLページを作る。

使い方:
  python build_report.py summary                 # 指標の要約を out/summary.json に出力し、表示
  python build_report.py html commentary.json    # 分析コメントを埋め込んだ out/report.html を生成

commentary.json の形式（すべて任意。無い項目は表示しない）:
{
  "headline": "今日の結論（2〜3行）",
  "market": "市場サマリー（複数行可）",
  "events": "今日の注目イベント・リスク",
  "stocks": {"8306": {"verdict": "保有", "comment": "根拠", "levels": "注視価格帯"}},
  "candidates": "注目候補",
  "sources": [{"title": "...", "url": "https://..."}]
}
verdict は 買い増し / 保有 / 縮小 / 撤退 / 監視 / 新規候補 のいずれか。
"""
from __future__ import annotations

import csv
import datetime as dt
import html
import json
import math
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "out"
CHART_DAYS = 130          # チャートに表示する営業日数（約6ヶ月）
BOX_DAYS = 120            # ボックス／ブレイク判定の参照期間（約6ヶ月）


def _np(o):
    """numpy の型を JSON に変換"""
    return o.item() if hasattr(o, "item") else str(o)


# ---------------------------------------------------------------- 指標計算
def rsi_wilder(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0)
    dn = -d.clip(upper=0)
    au = up.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    ad = dn.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = au / ad.replace(0, math.nan)
    r = 100 - 100 / (1 + rs)
    return r.where(ad != 0, 100.0)


def load_watchlist() -> list[dict]:
    rows = []
    with open(ROOT / "watchlist.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            code = (r.get("code") or "").strip()
            if code:
                rows.append({"code": code[:4], "group": (r.get("group") or "").strip(),
                             "memo": (r.get("memo") or "").strip()})
    return rows


def load_master() -> dict[str, dict]:
    p = DATA / "master.csv"
    if not p.exists():
        return {}
    m = pd.read_csv(p, dtype=str)
    out = {}
    for _, r in m.iterrows():
        out[str(r.get("Code", ""))[:4]] = {"name": r.get("CoName") or "",
                                           "sector": r.get("S33Nm") or "",
                                           "market": r.get("MktNm") or ""}
    return out


def load_earnings() -> dict[str, str]:
    p = DATA / "earnings.csv"
    if not p.exists():
        return {}
    e = pd.read_csv(p, dtype=str)
    code_col = next((c for c in e.columns if c.lower() == "code"), None)
    date_col = next((c for c in e.columns if c.lower() == "date"), None)
    if not code_col or not date_col:
        return {}
    return {str(r[code_col])[:4]: str(r[date_col])[:10] for _, r in e.iterrows()}


def margin_info(code: str) -> dict | None:
    p = DATA / "margin" / f"{code}.csv"
    if not p.exists():
        return None
    m = pd.read_csv(p)
    if m.empty or "LongVol" not in m.columns:
        return None
    # 同日に複数区分がある場合は合計
    g = m.groupby("Date")[["LongVol", "ShrtVol"]].sum().sort_index()
    last = g.iloc[-1]
    prev = g.iloc[-6] if len(g) >= 6 else (g.iloc[0] if len(g) > 1 else None)

    def ratio(row):
        return round(float(row["LongVol"]) / float(row["ShrtVol"]), 2) if row["ShrtVol"] else None

    return {
        "date": str(g.index[-1]),
        "long": int(last["LongVol"]),
        "short": int(last["ShrtVol"]),
        "ratio": ratio(last),
        "long_chg_pct": (round((float(last["LongVol"]) / float(prev["LongVol"]) - 1) * 100, 1)
                         if prev is not None and prev["LongVol"] else None),
        "prev_ratio": ratio(prev) if prev is not None else None,
    }


def compute(code: str, topix: pd.DataFrame | None) -> dict | None:
    p = DATA / "bars" / f"{code}.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    if df.empty:
        return None
    df = df.rename(columns={"AdjO": "o", "AdjH": "h", "AdjL": "l", "AdjC": "c", "AdjVo": "v"})
    df = df.dropna(subset=["c"]).reset_index(drop=True)
    c = df["c"]
    df["ma5"] = c.rolling(5).mean()
    df["ma25"] = c.rolling(25).mean()
    df["ma75"] = c.rolling(75).mean()
    df["rsi"] = rsi_wilder(c)
    df["vavg20"] = df["v"].shift(1).rolling(20).mean()

    L = df.iloc[-1]
    P = df.iloc[-2] if len(df) > 1 else L
    n = len(df)

    def ret(k):
        return round((c.iloc[-1] / c.iloc[-1 - k] - 1) * 100, 2) if n > k else None

    prior = df.iloc[-1 - BOX_DAYS:-1] if n > BOX_DAYS + 1 else df.iloc[:-1]
    box_hi, box_lo = float(prior["h"].max()), float(prior["l"].min())
    box_width = round((box_hi / box_lo - 1) * 100, 1) if box_lo else None
    vol_ratio = round(float(L["v"] / L["vavg20"]), 2) if L["vavg20"] and not math.isnan(L["vavg20"]) else None
    breakout = bool(L["c"] > box_hi and (vol_ratio or 0) >= 1.5)
    # 半年以上の下落：ブレイク直前（1ヶ月前）時点で、そのさらに6ヶ月前比 -10% 以下
    pre_trend = (round((c.iloc[-21] / c.iloc[-141] - 1) * 100, 1) if n > 141 else None)

    y = df.iloc[-245:] if n > 245 else df
    hi52, lo52 = float(y["h"].max()), float(y["l"].min())

    dev25 = round((L["c"] / L["ma25"] - 1) * 100, 2) if not math.isnan(L["ma25"]) else None
    ma25_slope = (round((df["ma25"].iloc[-1] / df["ma25"].iloc[-6] - 1) * 100, 2)
                  if n > 30 else None)

    rel20 = None
    if topix is not None and not topix.empty and n > 20:
        t = topix.set_index("Date")["C"]
        d0, d1 = df["Date"].iloc[-21], df["Date"].iloc[-1]
        if d0 in t.index and d1 in t.index:
            rel20 = round(ret(20) - (t[d1] / t[d0] - 1) * 100, 2)

    mg = margin_info(code)
    rsi = round(float(L["rsi"]), 1) if not math.isnan(L["rsi"]) else None
    entry = {
        "margin_ok": (mg["ratio"] is not None and mg["ratio"] < 50) if mg else None,
        "rsi_ok": (rsi is not None and 50 <= rsi < 60),
        "dev_ok": (dev25 is not None and abs(dev25) < 5),
    }

    chart = df.iloc[-CHART_DAYS:]

    def r2(x):
        return None if x is None or (isinstance(x, float) and math.isnan(x)) else round(float(x), 2)

    return {
        "code": code,
        "date": str(L["Date"]),
        "close": r2(L["c"]),
        "chg": r2(L["c"] - P["c"]),
        "chg_pct": r2((L["c"] / P["c"] - 1) * 100),
        "ma5": r2(L["ma5"]), "ma25": r2(L["ma25"]), "ma75": r2(L["ma75"]),
        "dev25": dev25, "ma25_slope5d": ma25_slope,
        "above_ma25": bool(L["c"] > L["ma25"]) if dev25 is not None else None,
        "ma25_cross_up_today": bool(P["c"] <= P["ma25"] and L["c"] > L["ma25"]) if n > 26 else None,
        "rsi": rsi,
        "vol": int(L["v"]), "vol_ratio20": vol_ratio,
        "ret5": ret(5), "ret20": ret(20), "ret60": ret(60), "ret120": ret(120),
        "rel_topix20": rel20,
        "hi52": r2(hi52), "lo52": r2(lo52),
        "from_hi52_pct": r2((L["c"] / hi52 - 1) * 100),
        "box120_hi": r2(box_hi), "box120_lo": r2(box_lo), "box120_width_pct": box_width,
        "pretrend_6m_pct": pre_trend,
        "breakout_with_volume": breakout,
        "margin": mg,
        "entry_check": entry,
        "_chart": {
            "candles": [{"time": r.Date, "open": r2(r.o), "high": r2(r.h), "low": r2(r.l), "close": r2(r.c)}
                        for r in chart.itertuples()],
            "vol": [{"time": r.Date, "value": int(r.v), "up": bool(r.c >= r.o)} for r in chart.itertuples()],
            "ma25": [{"time": r.Date, "value": r2(r.ma25)} for r in chart.itertuples() if not math.isnan(r.ma25)],
            "ma75": [{"time": r.Date, "value": r2(r.ma75)} for r in chart.itertuples() if not math.isnan(r.ma75)],
            "rsi": [{"time": r.Date, "value": r2(r.rsi)} for r in chart.itertuples() if not math.isnan(r.rsi)],
        },
    }


def build_all() -> tuple[list[dict], dict]:
    watch = load_watchlist()
    master = load_master()
    earn = load_earnings()
    tp = pd.read_csv(DATA / "topix.csv") if (DATA / "topix.csv").exists() else None
    meta = json.loads((DATA / "meta.json").read_text(encoding="utf-8")) if (DATA / "meta.json").exists() else {}
    rows = []
    for w in watch:
        s = compute(w["code"], tp)
        info = master.get(w["code"], {})
        base = {"code": w["code"], "group": w["group"], "memo": w["memo"],
                "name": info.get("name", ""), "sector": info.get("sector", ""),
                "earnings_date": earn.get(w["code"])}
        rows.append({**base, **s} if s else {**base, "missing": True})
    return rows, meta


def cmd_summary() -> None:
    rows, meta = build_all()
    OUT.mkdir(exist_ok=True)
    slim = [{k: v for k, v in r.items() if k != "_chart"} for r in rows]
    payload = {"meta": meta, "stocks": slim}
    (OUT / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=1, default=_np), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=1, default=_np))


# ---------------------------------------------------------------- HTML
def esc(s) -> str:
    return html.escape(str(s or ""))


def fmt(x, nd=1, sign=False, suffix=""):
    if x is None:
        return "—"
    s = f"{x:+,.{nd}f}" if sign else f"{x:,.{nd}f}"
    return s + suffix


def paras(text: str) -> str:
    if not text:
        return ""
    return "".join(f"<p>{esc(t)}</p>" for t in str(text).split("\n") if t.strip())


VERDICT_CLASS = {"買い増し": "v-add", "新規候補": "v-add", "保有": "v-hold", "監視": "v-hold",
                 "縮小": "v-trim", "撤退": "v-exit"}


def check_cell(ok) -> str:
    if ok is None:
        return '<span class="ck na">—</span>'
    return '<span class="ck ok">○</span>' if ok else '<span class="ck ng">×</span>'


def cmd_html(commentary_path: str | None) -> None:
    rows, meta = build_all()
    cm = {}
    if commentary_path and Path(commentary_path).exists():
        cm = json.loads(Path(commentary_path).read_text(encoding="utf-8"))
    stocks_cm = cm.get("stocks", {})
    latest = meta.get("latest_bar_date") or max((r.get("date") or "" for r in rows), default="")
    now = dt.datetime.now(dt.timezone(dt.timedelta(hours=9)))

    # 一覧表
    trs = []
    for r in rows:
        v = (stocks_cm.get(r["code"]) or {}).get("verdict", "")
        if r.get("missing"):
            trs.append(f'<tr><td><a href="#s{r["code"]}">{esc(r["code"])}</a></td><td class="nm">{esc(r["name"])}</td>'
                       f'<td colspan="6" class="muted">データなし</td></tr>')
            continue
        mg = r.get("margin") or {}
        updown = "up" if (r["chg"] or 0) > 0 else ("dn" if (r["chg"] or 0) < 0 else "")
        pill = f'<span class="pill {VERDICT_CLASS.get(v, "v-hold")}">{esc(v)}</span>' if v else ""
        trs.append(
            f'<tr><td><a href="#s{r["code"]}">{esc(r["code"])}</a></td>'
            f'<td class="nm">{esc(r["name"])}</td>'
            f'<td class="num">{fmt(r["close"], 1)}</td>'
            f'<td class="num {updown}">{fmt(r["chg_pct"], 2, True, "%")}</td>'
            f'<td class="num">{fmt(r["rsi"], 1)}</td>'
            f'<td class="num">{fmt(r["dev25"], 1, True, "%")}</td>'
            f'<td class="num">{fmt(mg.get("ratio"), 1, False, "倍") if mg else "—"}</td>'
            f'<td>{pill}</td></tr>')

    # 銘柄カード
    cards = []
    chart_data = {}
    for r in rows:
        code = r["code"]
        c = stocks_cm.get(code) or {}
        v = c.get("verdict", "")
        if r.get("missing"):
            cards.append(f'<section class="stock" id="s{code}"><header class="sh"><div class="id"><span class="code">{code}</span>'
                         f'<h2>{esc(r["name"]) or code}</h2></div></header><p class="muted">日足データを取得できませんでした。</p></section>')
            continue
        chart_data[code] = r["_chart"]
        mg = r.get("margin") or {}
        e = r["entry_check"]
        updown = "up" if (r["chg"] or 0) > 0 else ("dn" if (r["chg"] or 0) < 0 else "")
        flags = []
        if r.get("breakout_with_volume"):
            flags.append("6ヶ月高値を出来高伴い上抜け")
        if r.get("ma25_cross_up_today"):
            flags.append("本日25日線を上抜け")
        if r.get("earnings_date"):
            flags.append(f"決算予定 {r['earnings_date']}")
        flag_html = "".join(f'<span class="flag">{esc(f)}</span>' for f in flags)
        cards.append(f'''
<section class="stock" id="s{code}">
  <header class="sh">
    <div class="id">
      <span class="code">{code}</span>
      <h2>{esc(r["name"]) or code}</h2>
      <span class="meta">{esc(r["sector"])}{" · " + esc(r["group"]) if r["group"] else ""}</span>
    </div>
    <div class="px">
      <span class="last">{fmt(r["close"], 1)}</span>
      <span class="chg {updown}">{fmt(r["chg"], 1, True)}（{fmt(r["chg_pct"], 2, True, "%")}）</span>
    </div>
  </header>
  {f'<div class="flags">{flag_html}</div>' if flags else ''}
  <dl class="kpi">
    <div><dt>RSI(14)</dt><dd>{fmt(r["rsi"], 1)}</dd></div>
    <div><dt>25日線乖離</dt><dd>{fmt(r["dev25"], 1, True, "%")}</dd></div>
    <div><dt>信用倍率</dt><dd>{fmt(mg.get("ratio"), 2, False, "倍") if mg else "—"}</dd></div>
    <div><dt>出来高/20日平均</dt><dd>{fmt(r["vol_ratio20"], 2, False, "倍")}</dd></div>
    <div><dt>20日 対TOPIX</dt><dd>{fmt(r["rel_topix20"], 1, True, "pt")}</dd></div>
    <div><dt>52週高値から</dt><dd>{fmt(r["from_hi52_pct"], 1, True, "%")}</dd></div>
  </dl>
  <div class="entry">エントリー条件
    <span>信用倍率&lt;50 {check_cell(e["margin_ok"])}</span>
    <span>RSI 50台 {check_cell(e["rsi_ok"])}</span>
    <span>乖離&lt;5% {check_cell(e["dev_ok"])}</span>
  </div>
  <div class="legend" id="lg{code}" aria-live="polite"></div>
  <div class="chart main" id="c{code}" role="img" aria-label="{esc(r["name"])} 日足チャート（約6ヶ月）"></div>
  <div class="chart rsi" id="r{code}" role="img" aria-label="{esc(r["name"])} RSI"></div>
  <div class="analysis">
    {f'<span class="pill {VERDICT_CLASS.get(v, "v-hold")}">{esc(v)}</span>' if v else ''}
    {paras(c.get("comment", ""))}
    {f'<p class="levels"><b>注視価格帯</b> {esc(c.get("levels"))}</p>' if c.get("levels") else ''}
    {f'<p class="memo">メモ：{esc(r["memo"])}</p>' if r.get("memo") else ''}
  </div>
</section>''')

    src = cm.get("sources") or []
    src_html = "".join(f'<li><a href="{esc(s.get("url"))}" target="_blank" rel="noopener">{esc(s.get("title") or s.get("url"))}</a></li>'
                       for s in src if s.get("url"))
    errs = meta.get("errors") or {}
    err_html = (f'<p class="warn">取得エラー: {esc(", ".join(errs.keys()))}</p>' if errs else "")

    page = TEMPLATE
    repl = {
        "{{DATE}}": esc(latest),
        "{{GENERATED}}": esc(now.strftime("%Y-%m-%d %H:%M")),
        "{{FETCHED}}": esc(meta.get("fetched_at", "")[:16].replace("T", " ")),
        "{{HEADLINE}}": paras(cm.get("headline", "")) or '<p class="muted">分析コメントなし（チャートと指標のみ）</p>',
        "{{MARKET}}": (f'<section class="block"><h3>市場サマリー</h3>{paras(cm["market"])}</section>' if cm.get("market") else ""),
        "{{EVENTS}}": (f'<section class="block"><h3>今日の注目イベント・リスク</h3>{paras(cm["events"])}</section>' if cm.get("events") else ""),
        "{{CANDIDATES}}": (f'<section class="block"><h3>注目候補</h3>{paras(cm["candidates"])}</section>' if cm.get("candidates") else ""),
        "{{ROWS}}": "".join(trs),
        "{{CARDS}}": "".join(cards),
        "{{SOURCES}}": (f'<section class="block src"><h3>出典</h3><ul>{src_html}</ul></section>' if src_html else ""),
        "{{ERRORS}}": err_html,
        "{{DATA}}": json.dumps(chart_data, ensure_ascii=False, separators=(",", ":"), default=_np).replace("</", "<\\/"),
    }
    for k, val in repl.items():
        page = page.replace(k, val)
    OUT.mkdir(exist_ok=True)
    (OUT / "report.html").write_text(page, encoding="utf-8")
    print(f"wrote {OUT / 'report.html'} ({len(page):,} bytes)")


TEMPLATE = r"""<title>朝のチャート分析</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=BIZ+UDPGothic:wght@400;700&family=Roboto+Mono:wght@400;500&display=swap">
<style>
/* Layout: one reading column; summary table first, then one block per stock (chart → numbers → analysis) */
:root{
  --bg:#f6f7f9; --panel:#ffffff; --ink:#15171c; --ink-2:#4f5563; --ink-3:#7a8090; --rule:#dde1e8;
  --accent:#34408f;
  --up:#e34948; --dn:#2a78d6; --ma25:#c98a00; --ma75:#6250d6; --rsi:#4f5563; --band:rgba(52,64,143,.08);
  --vol-up:rgba(227,73,72,.35); --vol-dn:rgba(42,120,214,.35);
  --good:#0f7a3d; --good-bg:#e3f4ea; --warn:#9a6400; --warn-bg:#fbf0d9; --bad:#b42323; --bad-bg:#fbe4e4; --neu-bg:#eceff4;
  --f-body:"BIZ UDPGothic","Hiragino Kaku Gothic ProN","Yu Gothic",Meiryo,system-ui,sans-serif;
  --f-num:"Roboto Mono",ui-monospace,"SFMono-Regular",Menlo,monospace;
}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){
  --bg:#121418; --panel:#1a1d23; --ink:#eceef2; --ink-2:#b4bac6; --ink-3:#8a91a0; --rule:#2c313a;
  --accent:#9aa5f0;
  --up:#e66767; --dn:#3987e5; --ma25:#d9a020; --ma75:#9085e9; --rsi:#b4bac6; --band:rgba(154,165,240,.10);
  --vol-up:rgba(230,103,103,.35); --vol-dn:rgba(57,135,229,.35);
  --good:#6fd39a; --good-bg:#183324; --warn:#e8b75a; --warn-bg:#352a12; --bad:#f08a8a; --bad-bg:#3a1c1c; --neu-bg:#262a32;
  color-scheme:dark}}
:root[data-theme="dark"]{
  --bg:#121418; --panel:#1a1d23; --ink:#eceef2; --ink-2:#b4bac6; --ink-3:#8a91a0; --rule:#2c313a;
  --accent:#9aa5f0;
  --up:#e66767; --dn:#3987e5; --ma25:#d9a020; --ma75:#9085e9; --rsi:#b4bac6; --band:rgba(154,165,240,.10);
  --vol-up:rgba(230,103,103,.35); --vol-dn:rgba(57,135,229,.35);
  --good:#6fd39a; --good-bg:#183324; --warn:#e8b75a; --warn-bg:#352a12; --bad:#f08a8a; --bad-bg:#3a1c1c; --neu-bg:#262a32;
  color-scheme:dark}
*{box-sizing:border-box}
body{background:var(--bg);color:var(--ink);font-family:var(--f-body);font-size:15px;line-height:1.7;
  padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px));padding-block:20px 48px}
.wrap{max-width:860px;margin:0 auto;display:flex;flex-direction:column;gap:20px}
a{color:var(--accent)}
:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
.top{display:flex;flex-direction:column;gap:4px}
.eyebrow{font-size:12px;letter-spacing:.08em;color:var(--ink-3)}
h1{font-size:24px;line-height:1.3;margin:0;text-wrap:balance}
.stamp{font-family:var(--f-num);font-size:12px;color:var(--ink-3);font-variant-numeric:tabular-nums}
.headline{background:var(--panel);border:1px solid var(--rule);border-radius:10px;padding:14px 16px}
.headline p{margin:0 0 4px;font-size:16px;font-weight:700}
.block h3{font-size:14px;margin:0 0 6px;color:var(--ink-2);letter-spacing:.04em}
.block p{margin:0 0 6px;max-width:65ch}
.tbl{overflow-x:auto;border:1px solid var(--rule);border-radius:10px;background:var(--panel)}
table{border-collapse:collapse;width:100%;font-size:13px;min-width:560px}
th,td{padding:8px 10px;border-bottom:1px solid var(--rule);text-align:left;white-space:nowrap}
th{font-weight:400;color:var(--ink-3);font-size:12px}
tr:last-child td{border-bottom:0}
td.num,th.num{text-align:right;font-family:var(--f-num);font-variant-numeric:tabular-nums}
td.nm{white-space:normal;min-width:7em}
.up{color:var(--up)} .dn{color:var(--dn)} .muted{color:var(--ink-3)}
.pill{display:inline-block;font-size:12px;font-weight:700;padding:1px 10px;border-radius:999px;background:var(--neu-bg);color:var(--ink-2)}
.v-add{background:var(--good-bg);color:var(--good)} .v-add::before{content:"▲ "}
.v-hold{background:var(--neu-bg);color:var(--ink-2)} .v-hold::before{content:"■ "}
.v-trim{background:var(--warn-bg);color:var(--warn)} .v-trim::before{content:"▼ "}
.v-exit{background:var(--bad-bg);color:var(--bad)} .v-exit::before{content:"✕ "}
.stock{background:var(--panel);border:1px solid var(--rule);border-radius:10px;padding:16px;display:flex;flex-direction:column;gap:12px;scroll-margin-top:12px}
.sh{display:flex;justify-content:space-between;align-items:flex-end;gap:12px;flex-wrap:wrap}
.id{display:flex;flex-direction:column;min-width:0}
.code{font-family:var(--f-num);font-size:12px;color:var(--ink-3)}
.id h2{font-size:20px;margin:0;line-height:1.3}
.id .meta{font-size:12px;color:var(--ink-3)}
.px{display:flex;flex-direction:column;align-items:flex-end;font-family:var(--f-num);font-variant-numeric:tabular-nums}
.last{font-size:24px;font-weight:500;line-height:1.2}
.chg{font-size:13px}
.flags{display:flex;gap:6px;flex-wrap:wrap}
.flag{font-size:12px;border:1px solid var(--accent);color:var(--accent);border-radius:4px;padding:0 8px}
.kpi{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:8px 12px;margin:0}
.kpi div{display:flex;flex-direction:column;border-top:1px solid var(--rule);padding-top:6px}
.kpi dt{font-size:11px;color:var(--ink-3);letter-spacing:.04em}
.kpi dd{margin:0;font-family:var(--f-num);font-size:16px;font-variant-numeric:tabular-nums}
@media (max-width:480px){.kpi{grid-template-columns:repeat(2,minmax(0,1fr))}}
.entry{font-size:12px;color:var(--ink-2);display:flex;gap:4px 14px;flex-wrap:wrap;align-items:center}
.ck{font-weight:700}.ck.ok{color:var(--good)}.ck.ng{color:var(--bad)}.ck.na{color:var(--ink-3)}
.legend{font-family:var(--f-num);font-size:12px;color:var(--ink-2);min-height:3.2em;font-variant-numeric:tabular-nums;display:flex;flex-wrap:wrap;gap:0 12px}
.legend .k{color:var(--ink-3)}
.legend .sw{display:inline-block;width:12px;height:2px;vertical-align:middle;margin-right:4px}
.chart{width:100%;position:relative}
.chart.main{height:300px}.chart.rsi{height:110px}
.analysis{display:flex;flex-direction:column;gap:6px;align-items:flex-start}
.analysis p{margin:0;max-width:65ch}
.levels{font-size:14px}.levels b{font-weight:700;margin-right:6px}
.memo{font-size:12px;color:var(--ink-3)}
.note{font-size:12px;color:var(--ink-3)}
.warn{font-size:12px;color:var(--warn)}
.src ul{margin:0;padding-left:1.2em;font-size:13px;word-break:break-all}
@media (prefers-reduced-motion:reduce){*{scroll-behavior:auto}}
</style>

<div class="wrap">
  <header class="top">
    <span class="eyebrow">寄り前チャート分析</span>
    <h1>{{DATE}} 終値ベース</h1>
    <span class="stamp">データ取得 {{FETCHED}} ／ 作成 {{GENERATED}} JST</span>
    {{ERRORS}}
  </header>

  <section class="headline">{{HEADLINE}}</section>
  {{MARKET}}

  <section class="block">
    <h3>銘柄一覧</h3>
    <div class="tbl"><table>
      <thead><tr><th>コード</th><th>銘柄</th><th class="num">終値</th><th class="num">前日比</th><th class="num">RSI</th><th class="num">25日乖離</th><th class="num">信用倍率</th><th>判断</th></tr></thead>
      <tbody>{{ROWS}}</tbody>
    </table></div>
  </section>

  {{EVENTS}}

  {{CARDS}}

  {{CANDIDATES}}
  {{SOURCES}}
  <p class="note">日足は J-Quants の調整後株価。陽線＝赤、陰線＝青。線は25日移動平均（黄）と75日移動平均（紫）。RSI欄の帯は50〜60（エントリー目安）、点線は30と70。チャートは横にスワイプで期間移動、タップで値を表示。投資判断の材料であり、売買を推奨するものではありません。</p>
</div>

<script src="https://cdn.jsdelivr.net/npm/lightweight-charts@4.2.3/dist/lightweight-charts.standalone.production.js"></script>
<script>
(function(){
  const DATA = {{DATA}};
  const LWC = window.LightweightCharts;
  if(!LWC){document.querySelectorAll('.chart').forEach(el=>{el.textContent='チャートを読み込めませんでした';el.style.height='auto';});return;}
  const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
  const nf = (x,d=1)=> x==null?'—':Number(x).toLocaleString('ja-JP',{minimumFractionDigits:d,maximumFractionDigits:d});
  const charts = [];

  function theme(){
    return {
      layout:{background:{type:'solid',color:css('--panel')},textColor:css('--ink-3'),fontFamily:css('--f-num'),fontSize:11,attributionLogo:false},
      grid:{vertLines:{visible:false},horzLines:{color:css('--rule')}},
      rightPriceScale:{borderColor:css('--rule')},
      timeScale:{borderColor:css('--rule')},
      crosshair:{vertLine:{color:css('--ink-3'),labelBackgroundColor:css('--ink-2')},horzLine:{color:css('--ink-3'),labelBackgroundColor:css('--ink-2')}}
    };
  }
  const common = {
    autoSize:true,
    localization:{locale:'ja-JP',dateFormat:'yyyy/MM/dd'},
    handleScroll:{mouseWheel:false,pressedMouseMove:true,horzTouchDrag:true,vertTouchDrag:false},
    handleScale:{mouseWheel:false,pinch:true,axisPressedMouseMove:false,axisDoubleClickReset:true},
    timeScale:{rightOffset:2,fixLeftEdge:true,fixRightEdge:true}
  };

  Object.entries(DATA).forEach(([code,d])=>{
    const el = document.getElementById('c'+code), elr = document.getElementById('r'+code), lg = document.getElementById('lg'+code);
    if(!el) return;
    const main = LWC.createChart(el, Object.assign({}, common, theme()));
    const candle = main.addCandlestickSeries({borderVisible:false,priceLineVisible:false});
    candle.setData(d.candles);
    const m25 = main.addLineSeries({lineWidth:2,priceLineVisible:false,lastValueVisible:false,crosshairMarkerVisible:false});
    m25.setData(d.ma25);
    const m75 = main.addLineSeries({lineWidth:2,priceLineVisible:false,lastValueVisible:false,crosshairMarkerVisible:false});
    m75.setData(d.ma75);
    const vol = main.addHistogramSeries({priceFormat:{type:'volume'},priceScaleId:'vol',lastValueVisible:false,priceLineVisible:false});
    main.priceScale('vol').applyOptions({scaleMargins:{top:0.8,bottom:0},visible:false});
    main.priceScale('right').applyOptions({scaleMargins:{top:0.06,bottom:0.24}});

    const rc = LWC.createChart(elr, Object.assign({}, common, theme(), {rightPriceScale:{borderColor:css('--rule'),scaleMargins:{top:0.08,bottom:0.08}}}));
    const band = rc.addBaselineSeries({baseValue:{type:'price',price:50},lineVisible:false,priceLineVisible:false,lastValueVisible:false,crosshairMarkerVisible:false,topLineColor:'transparent',bottomLineColor:'transparent',bottomFillColor1:'transparent',bottomFillColor2:'transparent'});
    band.setData(d.rsi.map(p=>({time:p.time,value:60})));
    const rsi = rc.addLineSeries({lineWidth:2,priceLineVisible:false,lastValueVisible:true});
    rsi.setData(d.rsi);
    const pl = [];
    [30,70].forEach(v=>pl.push(rsi.createPriceLine({price:v,lineWidth:1,lineStyle:2,axisLabelVisible:true,title:''})));
    // 縦軸を0-100に固定
    rsi.applyOptions({autoscaleInfoProvider:()=>({priceRange:{minValue:0,maxValue:100}})});
    band.applyOptions({autoscaleInfoProvider:()=>({priceRange:{minValue:0,maxValue:100}})});

    function paint(){
      main.applyOptions(theme()); rc.applyOptions(theme());
      const up=css('--up'), dn=css('--dn');
      candle.applyOptions({upColor:up,downColor:dn,wickUpColor:up,wickDownColor:dn});
      m25.applyOptions({color:css('--ma25')}); m75.applyOptions({color:css('--ma75')});
      const vu=css('--vol-up'), vd=css('--vol-dn');
      vol.setData(d.vol.map(p=>({time:p.time,value:p.value,color:p.up?vu:vd})));
      band.applyOptions({topFillColor1:css('--band'),topFillColor2:css('--band')});
      rsi.applyOptions({color:css('--rsi')});
      pl.forEach(p=>p.applyOptions({color:css('--ink-3')}));
    }
    paint();
    main.timeScale().fitContent(); rc.timeScale().fitContent();

    // 2つのチャートの期間を同期
    let lock=false;
    const sync=(a,b)=>a.timeScale().subscribeVisibleLogicalRangeChange(r=>{if(lock||!r)return;lock=true;b.timeScale().setVisibleLogicalRange(r);lock=false;});
    sync(main,rc); sync(rc,main);

    // 凡例＋ホバー値
    const byTime = {}; d.candles.forEach((c,i)=>byTime[c.time]=i);
    const mapOf = arr => { const m={}; arr.forEach(p=>m[p.time]=p.value); return m; };
    const M25=mapOf(d.ma25), M75=mapOf(d.ma75), R=mapOf(d.rsi), V={}; d.vol.forEach(p=>V[p.time]=p.value);
    function show(t){
      const i = byTime[t]; if(i==null) return;
      const c=d.candles[i];
      lg.innerHTML =
        `<span><span class="k">${c.time}</span></span>`+
        `<span><span class="k">始</span>${nf(c.open)} <span class="k">高</span>${nf(c.high)} <span class="k">安</span>${nf(c.low)} <span class="k">終</span>${nf(c.close)}</span>`+
        `<span><span class="sw" style="background:var(--ma25)"></span><span class="k">25日</span>${nf(M25[t])}</span>`+
        `<span><span class="sw" style="background:var(--ma75)"></span><span class="k">75日</span>${nf(M75[t])}</span>`+
        `<span><span class="k">出来高</span>${nf(V[t],0)}</span>`+
        `<span><span class="k">RSI</span>${nf(R[t])}</span>`;
    }
    const last = d.candles[d.candles.length-1].time;
    show(last);
    const onMove = p => { if(p && p.time){ show(p.time); } else { show(last); } };
    main.subscribeCrosshairMove(onMove); rc.subscribeCrosshairMove(onMove);
    charts.push(paint);
  });

  const repaint = ()=>charts.forEach(f=>f());
  try{ matchMedia('(prefers-color-scheme: dark)').addEventListener('change',repaint); }catch(e){}
  new MutationObserver(repaint).observe(document.documentElement,{attributes:true,attributeFilter:['data-theme']});
})();
</script>
"""


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args or args[0] == "summary":
        cmd_summary()
    elif args[0] == "html":
        cmd_html(args[1] if len(args) > 1 else None)
    else:
        print(__doc__)
        sys.exit(2)
