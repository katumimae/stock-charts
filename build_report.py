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
  "earnings": {"8306": {"date": "2026-11-14", "confirmed": true, "period": "第2四半期（中間）"}},
  "sources": [{"title": "...", "url": "https://..."}]
}
verdict は 買い増し / 保有 / 縮小 / 撤退 / 監視 / 新規候補 のいずれか。
stocks[code] に "plan" を入れると、自動計算した売買の目安を上書きできる（すべて任意・円）:
  {"breakout": 3520, "breakout_basis": "10/9戻り高値", "breakout_stop": 3400,
   "pullback_lo": 3380, "pullback_hi": 3420, "pullback_basis": "75日線", "stop": 3300,
   "note": "一言の補足"}
earnings は会社発表で確定した決算日を上書きする（推定→確定）。
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


EARNINGS_WARN_DAYS = 45   # 決算の何日前から表示するか（約1ヶ月半）


def next_earnings(code: str, today: dt.date, confirmed: str | None) -> dict | None:
    """次回の決算発表日。確定日（翌営業日分の決算発表予定）があればそれを、
    なければ前年同四半期の開示日 + 364日（同じ曜日）で推定する。"""
    if confirmed:
        d = dt.date.fromisoformat(confirmed[:10])
        if d >= today:
            return {"date": d.isoformat(), "confirmed": True, "period": "", "days": (d - today).days}
    p = DATA / "fins" / f"{code}.csv"
    if not p.exists():
        return None
    f = pd.read_csv(p, dtype=str)
    if f.empty or "DocType" not in f.columns:
        return None
    f = f[f["DocType"].str.contains("FinancialStatements", na=False)].dropna(subset=["DiscDate"])
    if f.empty:
        return None
    f["d"] = pd.to_datetime(f["DiscDate"]).dt.date
    last_disc = f["d"].max()
    # 前年同期の開示日 + 364日のうち、最新開示より後で、今日以降の最も早いもの
    best = None
    for _, r in f.iterrows():
        est = r["d"] + dt.timedelta(days=364)
        if est > last_disc + dt.timedelta(days=20) and est >= today - dt.timedelta(days=3):
            if best is None or est < best[0]:
                best = (est, str(r.get("CurPerType") or ""))
    if not best:
        return None
    est, per = best
    label = {"1Q": "第1四半期", "2Q": "第2四半期（中間）", "3Q": "第3四半期", "FY": "本決算"}.get(per, per)
    return {"date": est.isoformat(), "confirmed": False, "period": label, "days": (est - today).days}


def trade_plan(df: pd.DataFrame) -> dict:
    """チャートの節目から、ブレイク買い・押し目買い・損切り・追いかけ上限の価格を出す。"""
    n = len(df)
    c = float(df["c"].iloc[-1])
    ma25 = float(df["ma25"].iloc[-1])
    ma75 = float(df["ma75"].iloc[-1]) if not math.isnan(df["ma75"].iloc[-1]) else None
    tr = pd.concat([df["h"] - df["l"], (df["h"] - df["c"].shift()).abs(), (df["l"] - df["c"].shift()).abs()], axis=1).max(axis=1)
    atr = float(tr.rolling(14).mean().iloc[-1])
    slope25 = float(df["ma25"].iloc[-1] / df["ma25"].iloc[-6] - 1) if n > 30 else 0.0

    look = df.iloc[-120:] if n > 120 else df
    off = n - len(look)
    hi_piv, lo_piv = [], []
    K = 5
    for i in range(off + K, n - K):
        h, l = df["h"].iloc[i], df["l"].iloc[i]
        if h == df["h"].iloc[i - K:i + K + 1].max():
            hi_piv.append((float(h), str(df["Date"].iloc[i])))
        if l == df["l"].iloc[i - K:i + K + 1].min():
            lo_piv.append((float(l), str(df["Date"].iloc[i])))

    def md(d):
        return f"{int(d[5:7])}/{int(d[8:10])}"

    res, sup = [], []   # (価格, ラベル, 種別)
    h20, l20 = float(df["h"].iloc[-20:].max()), float(df["l"].iloc[-20:].min())
    if h20 > c * 1.003:
        res.append((h20, "20日高値", "high"))
    for p, d in hi_piv:
        if p > c * 1.003:
            res.append((p, f"戻り高値({md(d)})", "high"))
    box_hi = float(look["h"].max())
    if box_hi > c * 1.003:
        res.append((box_hi, "6ヶ月高値", "high"))
    if ma25 > c:
        res.append((ma25, "25日線", "ma"))
    if ma75 and ma75 > c:
        res.append((ma75, "75日線", "ma"))
    if ma25 < c:
        sup.append((ma25, "25日線", "ma"))
    if ma75 and ma75 < c:
        sup.append((ma75, "75日線", "ma"))
    for p, d in lo_piv:
        if p < c * 0.997:
            sup.append((p, f"押し安値({md(d)})", "low"))
    if l20 < c * 0.997:
        sup.append((l20, "20日安値", "low"))

    def dedupe(items, reverse):
        items = sorted(items, key=lambda x: x[0], reverse=reverse)
        out = []
        for it in items:
            if out and abs(it[0] / out[-1][0] - 1) < 0.012:
                # 近い節目はまとめる（移動平均より価格の節目を優先しラベルを併記）
                prev = out[-1]
                lab = prev[1] if prev[1].count("・") >= 1 else prev[1] + "・" + it[1]
                out[-1] = (prev[0], lab, prev[2] if prev[2] != "ma" else it[2])
                continue
            out.append(it)
        return out

    res = dedupe(res, reverse=False)
    sup = dedupe(sup, reverse=True)

    def yen(x):
        return None if x is None else int(round(x))

    # ブレイク買い：現値から1%以上上にある、最も近い「価格の高値」（移動平均は除く）
    highs = dedupe([x for x in res if x[2] == "high" and x[0] > c * 1.01], reverse=False)
    brk = highs[0] if highs else None
    reclaim = ma25 if c < ma25 else None
    downtrend = c < ma25 and slope25 < 0 and (ma75 is None or c < ma75)

    # 押し目買い：最も近い下値の節目 〜 +0.5ATR
    pb = None
    sup_pb = [x for x in sup if x[0] < c * 0.99]   # 現値から1%以上下の節目を押し目候補に
    if sup_pb:
        s1 = sup_pb[0]
        lo_, hi_ = s1[0], min(s1[0] + 0.5 * atr, c * 0.995)
        s2 = sup_pb[1][0] if len(sup_pb) > 1 else None
        # 損切り：次の節目が近ければその少し下、遠ければ押し目下限 − 1ATR
        stop = lo_ - 1.0 * atr
        if s2 and lo_ - 1.0 * atr < s2 - 0.3 * atr <= lo_ - 0.5 * atr:
            stop = s2 - 0.3 * atr
        pb = {"lo": yen(lo_), "hi": yen(hi_), "basis": s1[1], "stop": yen(stop),
              "risk_pct": round((stop / ((lo_ + hi_) / 2) - 1) * 100, 1),
              "caution": "下降トレンド中：節目で下げ止まり（陽線・出来高）を確認してから" if downtrend else ""}
    bo = None
    if brk:
        bstop = max(brk[0] - 1.5 * atr, ma25 if ma25 < brk[0] else 0)
        bo = {"price": yen(brk[0]), "basis": brk[1], "stop": yen(bstop),
              "risk_pct": round((bstop / brk[0] - 1) * 100, 1)}
    return {
        "atr14": round(atr, 1),
        "trend": "下降" if downtrend else ("上昇" if c > ma25 and slope25 > 0 else "横ばい"),
        "breakout": bo,
        "reclaim_ma25": yen(reclaim),
        "pullback": pb,
        "chase_limit": yen(ma25 * 1.05),   # 25日線乖離+5%＝これ以上は追わない
        "resistances": [{"price": yen(p), "label": lb} for p, lb, _ in res[:4]],
        "supports": [{"price": yen(p), "label": lb} for p, lb, _ in sup[:4]],
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
        "plan": trade_plan(df) if n > 30 else None,
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
    today = dt.datetime.now(dt.timezone(dt.timedelta(hours=9))).date()
    rows = []
    for w in watch:
        s = compute(w["code"], tp)
        info = master.get(w["code"], {})
        ne = next_earnings(w["code"], today, earn.get(w["code"]))
        base = {"code": w["code"], "group": w["group"], "memo": w["memo"],
                "name": info.get("name", ""), "sector": info.get("sector", ""),
                "earnings": ne,
                "earnings_soon": bool(ne and ne["days"] <= EARNINGS_WARN_DAYS)}
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


def yen(x) -> str:
    return "—" if x is None else f"{int(round(x)):,}"


def md_date(iso: str) -> str:
    d = dt.date.fromisoformat(iso[:10])
    return f"{d.month}/{d.day}（{'月火水木金土日'[d.weekday()]}）"


def effective_plan(r: dict, c: dict) -> dict | None:
    """自動計算の価格に、分析コメント側の上書き（commentary.stocks[code].plan）を反映"""
    base = r.get("plan")
    if not base:
        return None
    p = json.loads(json.dumps(base, default=_np))
    o = c.get("plan") or {}
    if o.get("breakout"):
        p["breakout"] = {**(p.get("breakout") or {}), "price": o["breakout"],
                         "basis": o.get("breakout_basis") or (p.get("breakout") or {}).get("basis", "")}
        if o.get("breakout_stop"):
            p["breakout"]["stop"] = o["breakout_stop"]
    if o.get("pullback_lo") and o.get("pullback_hi"):
        p["pullback"] = {**(p.get("pullback") or {}), "lo": o["pullback_lo"], "hi": o["pullback_hi"],
                         "basis": o.get("pullback_basis") or (p.get("pullback") or {}).get("basis", "")}
    if o.get("stop"):
        p.setdefault("pullback", {})["stop"] = o["stop"]
    for k in ("breakout", "pullback"):
        q = p.get(k) or {}
        ref = q.get("price") or ((q.get("lo", 0) + q.get("hi", 0)) / 2 if q.get("lo") else None)
        if ref and q.get("stop"):
            q["risk_pct"] = round((q["stop"] / ref - 1) * 100, 1)
    p["note"] = o.get("note", "")
    return p


def effective_earnings(r: dict, cm: dict) -> dict | None:
    o = (cm.get("earnings") or {}).get(r["code"])
    if o and o.get("date"):
        today = dt.datetime.now(dt.timezone(dt.timedelta(hours=9))).date()
        d = dt.date.fromisoformat(o["date"][:10])
        base = r.get("earnings") or {}
        return {"date": d.isoformat(), "confirmed": bool(o.get("confirmed", True)),
                "period": o.get("period") or base.get("period", ""), "days": (d - today).days}
    return r.get("earnings")


def earn_text(e: dict) -> str:
    tag = "確定" if e["confirmed"] else "推定"
    days = "本日" if e["days"] == 0 else f"あと{e['days']}日"
    per = f"{e['period']} " if e.get("period") else ""
    return f"{per}{md_date(e['date'])} {days}（{tag}）"


def plan_html(p: dict | None) -> str:
    if not p:
        return ""
    rows = []
    b = p.get("breakout")
    if b:
        rows.append(f'<div class="pl-row pl-brk"><dt>ブレイク買い</dt><dd><b>{yen(b["price"])}円</b> を終値で上抜け＋出来高1.5倍'
                    f'<span class="pl-sub">根拠：{esc(b.get("basis"))}／損切り {yen(b.get("stop"))}円（{fmt(b.get("risk_pct"), 1, True, "%")}）</span></dd></div>')
    if p.get("reclaim_ma25"):
        rows.append(f'<div class="pl-row pl-brk"><dt>25日線奪回</dt><dd><b>{yen(p["reclaim_ma25"])}円</b> を終値で回復'
                    f'<span class="pl-sub">下落基調から戻りに転じたかの最初の確認ライン</span></dd></div>')
    q = p.get("pullback")
    if q and q.get("lo"):
        caution = f'<span class="pl-caution">{esc(q["caution"])}</span>' if q.get("caution") else ""
        rows.append(f'<div class="pl-row pl-pb"><dt>押し目買い</dt><dd><b>{yen(q["lo"])}〜{yen(q["hi"])}円</b>'
                    f'<span class="pl-sub">根拠：{esc(q.get("basis"))}／損切り {yen(q.get("stop"))}円（{fmt(q.get("risk_pct"), 1, True, "%")}）</span>{caution}</dd></div>')
    if p.get("chase_limit"):
        rows.append(f'<div class="pl-row pl-lim"><dt>追いかけ上限</dt><dd><b>{yen(p["chase_limit"])}円</b>'
                    f'<span class="pl-sub">25日線＋5%。これより上は乖離が大きく、新規では買わない</span></dd></div>')
    note = f'<p class="pl-note">{esc(p["note"])}</p>' if p.get("note") else ""
    lv = ""
    if p.get("resistances") or p.get("supports"):
        rs = "　".join(f'{yen(x["price"])}（{esc(x["label"])}）' for x in p.get("resistances", [])[:3]) or "—"
        ss = "　".join(f'{yen(x["price"])}（{esc(x["label"])}）' for x in p.get("supports", [])[:3]) or "—"
        lv = f'<p class="pl-lv"><span>上値の節目</span>{rs}<br><span>下値の節目</span>{ss}</p>'
    return (f'<div class="plan"><h4>売買の目安 <span class="pl-trend">トレンド：{esc(p.get("trend"))}</span></h4>'
            f'<dl>{"".join(rows)}</dl>{note}{lv}</div>')


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
                       f'<td colspan="9" class="muted">データなし</td></tr>')
            continue
        mg = r.get("margin") or {}
        updown = "up" if (r["chg"] or 0) > 0 else ("dn" if (r["chg"] or 0) < 0 else "")
        pill = f'<span class="pill {VERDICT_CLASS.get(v, "v-hold")}">{esc(v)}</span>' if v else ""
        pe = effective_plan(r, stocks_cm.get(r["code"]) or {}) or {}
        bo = (pe.get("breakout") or {}).get("price")
        pb = pe.get("pullback") or {}
        ee = effective_earnings(r, cm)
        ecell = (f'<span class="ecell{" hot" if ee["days"] <= 14 else ""}">{md_date(ee["date"])}{"" if ee["confirmed"] else "推"}</span>'
                 if ee and ee["days"] <= EARNINGS_WARN_DAYS else "")
        trs.append(
            f'<tr><td><a href="#s{r["code"]}">{esc(r["code"])}</a></td>'
            f'<td class="nm">{esc(r["name"])}</td>'
            f'<td class="num">{fmt(r["close"], 1)}</td>'
            f'<td class="num {updown}">{fmt(r["chg_pct"], 2, True, "%")}</td>'
            f'<td class="num">{fmt(r["rsi"], 1)}</td>'
            f'<td class="num">{fmt(r["dev25"], 1, True, "%")}</td>'
            f'<td class="num">{fmt(mg.get("ratio"), 1, False, "倍") if mg else "—"}</td>'
            f'<td class="num">{yen(bo) if bo else "—"}</td>'
            f'<td class="num">{(yen(pb["lo"]) + "〜" + yen(pb["hi"])) if pb.get("lo") else "—"}</td>'
            f'<td>{ecell}</td>'
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
        mg = r.get("margin") or {}
        e = r["entry_check"]
        updown = "up" if (r["chg"] or 0) > 0 else ("dn" if (r["chg"] or 0) < 0 else "")
        flags = []
        if r.get("breakout_with_volume"):
            flags.append("6ヶ月高値を出来高伴い上抜け")
        if r.get("ma25_cross_up_today"):
            flags.append("本日25日線を上抜け")
        flag_html = "".join(f'<span class="flag">{esc(f)}</span>' for f in flags)
        ee = effective_earnings(r, cm)
        if ee and ee["days"] <= EARNINGS_WARN_DAYS:
            flags.append("earn")
            flag_html = (f'<span class="flag earn{" hot" if ee["days"] <= 14 else ""}">決算 {esc(earn_text(ee))}</span>') + flag_html
        pe = effective_plan(r, c)
        lines = []
        if pe:
            if pe.get("breakout"):
                lines.append({"price": pe["breakout"]["price"], "kind": "brk", "title": "ブレイク"})
            q = pe.get("pullback") or {}
            if q.get("lo"):
                lines.append({"price": q["hi"], "kind": "pb", "title": "押し目"})
                lines.append({"price": q["lo"], "kind": "pb", "title": ""})
            if q.get("stop"):
                lines.append({"price": q["stop"], "kind": "stop", "title": "損切り"})
        chart_data[code] = {**r["_chart"], "lines": lines}
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
  {plan_html(pe)}
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

    soon = []
    for r in rows:
        ee = effective_earnings(r, cm)
        if ee and ee["days"] <= EARNINGS_WARN_DAYS:
            soon.append((ee["days"], f'<li class="{"hot" if ee["days"] <= 14 else ""}"><a href="#s{r["code"]}">{esc(r["code"])} {esc(r["name"])}</a>　{esc(earn_text(ee))}</li>'))
    earn_block = (f'<section class="block earnblk"><h3>決算が近い銘柄（{EARNINGS_WARN_DAYS}日以内）</h3><ul>{"".join(x for _, x in sorted(soon))}</ul>'
                  f'<p class="note">推定＝前年同期の発表日から算出。会社の発表で確定したら「確定」に切り替わります。決算をまたぐ新規買いは値幅リスクに注意。</p></section>'
                  if soon else "")
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
        "{{EARN}}": earn_block,
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
.flag.earn{border-color:var(--warn);color:var(--warn);background:var(--warn-bg);font-weight:700}
.flag.earn.hot,.ecell.hot,.earnblk li.hot{color:var(--bad)}
.flag.earn.hot{border-color:var(--bad);background:var(--bad-bg)}
.ecell{font-size:12px;color:var(--warn);font-weight:700}
.earnblk ul{margin:0;padding-left:1.2em;font-size:14px}
.earnblk li{font-variant-numeric:tabular-nums}
.plan{border:1px solid var(--rule);border-radius:8px;padding:10px 12px;display:flex;flex-direction:column;gap:6px}
.plan h4{margin:0;font-size:13px;color:var(--ink-2);display:flex;justify-content:space-between;gap:8px;flex-wrap:wrap}
.pl-trend{font-weight:400;color:var(--ink-3)}
.plan dl{margin:0;display:flex;flex-direction:column;gap:6px}
.pl-row{display:grid;grid-template-columns:6.5em minmax(0,1fr);gap:8px;align-items:baseline}
.pl-row dt{font-size:12px;font-weight:700;padding-left:8px;border-left:3px solid var(--rule)}
.pl-brk dt{border-left-color:var(--accent)} .pl-pb dt{border-left-color:var(--good)} .pl-lim dt{border-left-color:var(--ink-3)}
.pl-row dd{margin:0;font-size:14px;font-variant-numeric:tabular-nums}
.pl-row dd b{font-family:var(--f-num);font-size:16px}
.pl-sub{display:block;font-size:12px;color:var(--ink-3)}
.pl-caution{display:block;font-size:12px;color:var(--warn);font-weight:700}
.pl-note{margin:0;font-size:13px}
.pl-lv{margin:0;font-size:12px;color:var(--ink-2);font-variant-numeric:tabular-nums}
.pl-lv span{color:var(--ink-3);margin-right:8px}
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
      <thead><tr><th>コード</th><th>銘柄</th><th class="num">終値</th><th class="num">前日比</th><th class="num">RSI</th><th class="num">25日乖離</th><th class="num">信用倍率</th><th class="num">ブレイク買い</th><th class="num">押し目買い</th><th>決算</th><th>判断</th></tr></thead>
      <tbody>{{ROWS}}</tbody>
    </table></div>
  </section>

  {{EARN}}
  {{EVENTS}}

  {{CARDS}}

  {{CANDIDATES}}
  {{SOURCES}}
  <p class="note">売買の目安は、直近の高値・安値や移動平均（節目）とATR（平均的な1日の値幅）から自動で算出し、分析で補正したもの。チャートの水平線は青破線＝ブレイク買い、緑点線＝押し目買いゾーン、赤破線＝損切り。日足は J-Quants の調整後株価。陽線＝赤、陰線＝青。線は25日移動平均（黄）と75日移動平均（紫）。RSI欄の帯は50〜60（エントリー目安）、点線は30と70。チャートは横にスワイプで期間移動、タップで値を表示。投資判断の材料であり、売買を推奨するものではありません。</p>
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
    const lineObjs = (d.lines||[]).map(l=>candle.createPriceLine({price:l.price,lineWidth:1,lineStyle:l.kind==='pb'?1:2,axisLabelVisible:true,title:l.title}));
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
      const lc={brk:css('--accent'),pb:css('--good'),stop:css('--bad')};
      lineObjs.forEach((o,i)=>o.applyOptions({color:lc[d.lines[i].kind]}));
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
