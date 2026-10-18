"""J-Quants API v2 から監視銘柄のデータを取得して data/ に保存する。

GitHub Actions から毎日実行される。環境変数 JQUANTS_API_KEY が必要。
保存するもの:
  data/bars/<code>.csv     日足（調整後4本値・出来高）約1.5年分
  data/margin/<code>.csv   信用取引残高（買残・売残）約1年分
  data/topix.csv           TOPIX 日足
  data/master.csv          監視銘柄の銘柄名・業種・市場
  data/earnings.csv        決算発表予定（取得できた場合）
  data/meta.json           取得日時・最新日付・エラー
"""
from __future__ import annotations

import csv
import datetime as dt
import json
import os
import sys
import traceback
from pathlib import Path

import pandas as pd
import jquantsapi

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
JST = dt.timezone(dt.timedelta(hours=9))


def load_watchlist() -> list[dict]:
    rows = []
    with open(ROOT / "watchlist.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            code = (r.get("code") or "").strip()
            if code:
                rows.append({"code": code[:4], "group": (r.get("group") or "").strip(),
                             "memo": (r.get("memo") or "").strip()})
    return rows


def main() -> int:
    key = os.environ.get("JQUANTS_API_KEY", "").strip()
    if not key:
        print("JQUANTS_API_KEY が設定されていません", file=sys.stderr)
        return 1
    cli = jquantsapi.ClientV2(api_key=key)

    today = dt.datetime.now(JST).date()
    frm_bars = (today - dt.timedelta(days=550)).strftime("%Y%m%d")
    frm_margin = (today - dt.timedelta(days=400)).strftime("%Y%m%d")
    to = today.strftime("%Y%m%d")

    for sub in ("bars", "margin"):
        (DATA / sub).mkdir(parents=True, exist_ok=True)

    watch = load_watchlist()
    errors: dict[str, str] = {}
    latest: dict[str, str] = {}

    # 銘柄マスタ（銘柄名・業種）
    masters = []
    for w in watch:
        try:
            m = cli.get_eq_master(code=w["code"])
            if not m.empty:
                masters.append(m.iloc[[-1]])
        except Exception as e:  # noqa: BLE001
            errors[f"master:{w['code']}"] = str(e)[:300]
    if masters:
        pd.concat(masters).to_csv(DATA / "master.csv", index=False)

    # 日足
    keep = ["Date", "Code", "AdjO", "AdjH", "AdjL", "AdjC", "AdjVo", "Va", "C"]
    for w in watch:
        c = w["code"]
        try:
            df = cli.get_eq_bars_daily(code=c, from_yyyymmdd=frm_bars, to_yyyymmdd=to)
            if df.empty:
                errors[f"bars:{c}"] = "データなし"
                continue
            df = df[[k for k in keep if k in df.columns]].dropna(subset=["AdjC"])
            df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
            df.to_csv(DATA / "bars" / f"{c}.csv", index=False)
            latest[c] = df["Date"].iloc[-1]
        except Exception as e:  # noqa: BLE001
            errors[f"bars:{c}"] = str(e)[:300]

    # 信用残
    for w in watch:
        c = w["code"]
        try:
            df = cli.get_mkt_margin_interest(code=c, from_yyyymmdd=frm_margin, to_yyyymmdd=to)
            if not df.empty:
                for col in ("Date", "PubDate"):
                    if col in df.columns:
                        df[col] = pd.to_datetime(df[col], errors="coerce").dt.strftime("%Y-%m-%d")
                df.to_csv(DATA / "margin" / f"{c}.csv", index=False)
        except Exception as e:  # noqa: BLE001
            errors[f"margin:{c}"] = str(e)[:300]

    # TOPIX
    try:
        tp = cli.get_idx_bars_daily_topix(from_yyyymmdd=frm_bars, to_yyyymmdd=to)
        if not tp.empty:
            tp["Date"] = pd.to_datetime(tp["Date"]).dt.strftime("%Y-%m-%d")
            tp.to_csv(DATA / "topix.csv", index=False)
    except Exception as e:  # noqa: BLE001
        errors["topix"] = str(e)[:300]

    # 決算短信の開示履歴（次回決算日の推定に使う）
    (DATA / "fins").mkdir(parents=True, exist_ok=True)
    fin_keep = ["DiscDate", "DiscTime", "DocType", "CurPerType", "CurPerEn", "CurFYEn"]
    for w in watch:
        c = w["code"]
        try:
            fs = cli.get_fin_summary(code=c)
            if not fs.empty:
                fs = fs[[k for k in fin_keep if k in fs.columns]].copy()
                fs["DiscDate"] = pd.to_datetime(fs["DiscDate"], errors="coerce").dt.strftime("%Y-%m-%d")
                fs.sort_values("DiscDate").to_csv(DATA / "fins" / f"{c}.csv", index=False)
        except Exception as e:  # noqa: BLE001
            errors[f"fins:{c}"] = str(e)[:300]

    # 決算発表予定（翌営業日分。プランによっては取れない）
    try:
        ec = cli.get_eq_earnings_cal()
        if not ec.empty:
            ec.to_csv(DATA / "earnings.csv", index=False)
    except Exception as e:  # noqa: BLE001
        errors["earnings"] = str(e)[:300]

    meta = {
        "fetched_at": dt.datetime.now(JST).isoformat(timespec="seconds"),
        "latest_bar_date": max(latest.values()) if latest else None,
        "latest_by_code": latest,
        "errors": errors,
    }
    (DATA / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    # 日足が1銘柄も取れなければ失敗扱い（Actionsが赤くなって気づける）
    return 0 if latest else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
