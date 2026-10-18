# stock-charts

毎朝の寄り前チャート分析用のデータ置き場。

## 仕組み
1. GitHub Actions（`.github/workflows/fetch.yml`）が平日 20:05 と 6:20（JST）に `fetch.py` を実行し、J-Quants API v2 から日足・信用残・TOPIX を `data/` に保存してコミットする。
2. 平日 7:30 に Claude のスケジュールタスクがこのリポジトリを読み、`build_report.py` で指標とチャートを作り、分析を書き加えたページを更新する。

## 銘柄の追加・削除
`watchlist.csv` を GitHub の画面で編集するだけ。次回の取得から反映される。

| 列 | 内容 |
|---|---|
| code | 4桁の銘柄コード |
| group | 保有 / 監視 など |
| memo | 売買計画などのメモ（分析時に参照される） |

## 必要な設定
- リポジトリの Settings → Secrets and variables → Actions に `JQUANTS_API_KEY`（J-Quants ダッシュボードで発行した API キー）を登録する。
- 手動で取得したいときは Actions タブ → 「J-Quantsデータ取得」→ Run workflow。

## 手元での実行
```
pip install "jquants-api-client>=2.8,<3" pandas
JQUANTS_API_KEY=... python fetch.py
python build_report.py summary
python build_report.py html commentary.json   # out/report.html
```
