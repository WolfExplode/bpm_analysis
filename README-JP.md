<p align="center">
  <a href="README.md">English</a> |
  <a href="README-JP.md">日本語</a>
</p>

# 心拍数 BPM アナライザー

このツールは、心音図（PCG）解析のためのヒューリスティックベースのアルゴリズムです。
心音のオーディオ録音を解析して心拍を検出し、時間経過に伴う心拍数（BPM）をグラフ化します。

## 概要

```
pcg/
  engine/        アルゴリズム本体：PCG音声 → 心拍 / 状態 / BPM（純粋な計算）
  recording.py   録音の識別（デコード後の音声サンプルのフィンガープリント）、読み込み、チャンネル
  analysis.py    Analysis（1回の解析結果）ファイル。ライブラリフォルダに保存
  annotation.py  Annotation（手で確認した S1 / S2 / ノイズ区間 = 正解データ）
  batch.py       実行キュー、ファイル名のBPMタグ
  cli.py         python -m pcg analyze / inspect / export / rename
  app/           デスクトップアプリ（実行画面 + ワークスペース）
```

用語は [CONTEXT.md](CONTEXT.md)、設計は [docs/workspace-requirements.md](docs/workspace-requirements.md) を参照。

- **Analysis** は `library/`（または `$PCG_LIBRARY`）に、録音の音声サンプルのフィンガープリントをキーとして保存されます。
  ファイル名を変更しても（BPMタグの書き込みを含む）Analysis との対応は失われません。
- `pcg/engine/config.py` またはエンジンのコードが変わると Analysis に「STALE」バッジが付きます。再解析は指示したときだけ（Ctrl+R / Run）。
- **Annotation** は録音の隣の `<録音名>.annotation.json`。保護フォルダ（既定 `inputs/`）内の録音は「名前を付けて保存」（ダウンロードフォルダから開始）になります。

## 設定
エンジンのパラメータはすべて `pcg/engine/config.py` にあり、ここでのみ調整します（アプリにパラメータ編集画面はありません）。
アルゴリズム、自動切替、チャンネル、開始オフセット、開始BPMヒントは実行画面またはCLIで実行ごとに選びます。

## インストール

```bash
pip install -r requirements.txt
```

FFmpeg は libsndfile で読めない形式（m4a や動画ファイルなど）の場合のみ必要です。

## 実行方法

```bash
python -m pcg                      # アプリ（実行画面 + ワークスペース）
python -m pcg app path/to/rec.wav  # 1つの録音をワークスペースで開く
python -m pcg analyze inputs/ -j 8 --rename   # ヘッドレス解析（結果はライブラリへ、ファイル名にBPMタグ）
python -m pcg inspect rec.wav --from 120 --to 126   # Ctrl+Shift+C と同じテキスト
python -m pcg export rec.wav --bpm-csv bpm.csv
```

ワークスペースのキー操作はアプリの Help → Keys を参照してください（Space 再生、T 原音/フィルタ切替、1 / 2 で S1 / S2 配置、
N + ドラッグでノイズ区間、Ctrl+R で再解析、Ctrl+Shift+C で表示範囲をテキストでコピー など）。

## 追加機能:
生成された心拍数グラフをBlenderにインポートして、時間経過に伴うBPMの変化を簡単に計算できます。
Blenderファイルとスクリプトは Blender BPM tool フォルダに配置されています。

<img src="https://github.com/user-attachments/assets/20130a36-d990-43ba-9cb2-c4d4d248d069" alt="BlenderへのインポートAsj3vbrst4v" width="360" />

ジオメトリノードオブジェクトを選択し、編集モードに入ります。これにより以下を計算できます：
- 心拍数回復（HRR）
- 心拍数増加の最大レート

<img src="https://github.com/user-attachments/assets/f41d8e27-f525-4736-b67a-18de4e4b98e5" alt="Place BlenderAsj3zdst4v" width="360" />
<img src="https://github.com/user-attachments/assets/5d033948-f5b8-485f-9ebe-e9b87a6ee94c" alt="Adjust BlenderAsj3zny4v" width="360" />

また、任意の BPM/時間 グラフを作成し、`Export graph data.py` スクリプトを使用してBlenderからエクスポートすることもできます。

フォーマット「Time(Seconds), Beats Per Minute」のCSVファイルをインポートできます