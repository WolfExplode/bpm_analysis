<p align="center">
  <a href="README.md">English</a> |
  <a href="README-JP.md">日本語</a>
</p>

# 心拍数 BPM アナライザー

このツールは、心音図（PCG）解析のためのヒューリスティックベースのアルゴリズムです。
心音のオーディオ録音を解析して心拍を検出し、時間経過に伴う心拍数（BPM）をグラフ化します。

### **GUIインターフェース:**
_心拍数グラフを生成するだけで十分ですが、詳細な情報が必要な場合のために他のオプションも用意されています_

<img width="480" height="380" alt="image" src="https://github.com/user-attachments/assets/d1325e51-4c0c-4eab-bb1a-b2fcc6c17227" />

### [🔗 心拍数グラフの出力:](https://youtu.be/uzc9XESJmb8)
[![動画を見る|857x482](https://github.com/user-attachments/assets/b35ccc4a-dd20-49f6-a21d-64da8c746a92)](https://youtu.be/uzc9XESJmb8)

### **スペクトログラムビュー:**
_このスクリプトにはデバッグ用のスペクトログラムビューが含まれていますが、生成に非常に時間がかかります_
![brave_ykQQ36DQv](https://github.com/user-attachments/assets/7a10acc5-0208-455a-9a3a-0300e5a4d722)

## 設定
`pipeline.py` エンジンのすべての調整可能パラメータは `config.py` に配置されています。
パラメータは論理的なカテゴリに整理され、ナビゲーションと調整が容易になります。
- 複数フォーマットオーディオサポート: WAV、MP3、M4A、MOVなどの一般的なメディアファイルを受け付け、解析用に.wav形式に変換します。

## 動作環境
このスクリプトを実行するには、Pythonと以下のライブラリが必要です：
- **`numpy`**
- **`pandas`**
- **`scipy`**
- **`plotly`**
- **`ttkbootstrap`**
- **`pydub`**
- **`soxr`**
- **`librosa`**（オーディオの読み込みとリサンプリングを担当）
- **`PyWavelets`**（ウェーブレットデノイジングに使用される `pywt` モジュールを提供）
- **`pyPCG-toolbox`**（オプションの調整可能なpyPCGデノイジングを有効にします。この機能は `config.py` で `denoising_method` を設定した場合のみ有効になります）

**FFmpegをインストール:** 公式 [FFmpegウェブサイト](https://ffmpeg.org/download.html "null") からお使いのオペレーティングシステム向けのインストール手順に従ってください。

**FFmpegをインストール:** 公式 [FFmpegウェブサイト](https://ffmpeg.org/download.html "null") からお使いのオペレーティングシステム向けのインストール手順に従ってください。
`pydub` が正しく機能するためには、 **FFmpeg** がシステムのPATHにインストールされてアクセス可能である必要があります。

- [Microsoft Visual C++ Redistributable 最新サポート版 v14](https://aka.ms/vc14/vc_redist.x64.exe) （Visual Studio 2017–2026用）がインストールされていることを確認してください。

## 実行方法
**依存関係のインストール:**

```
pip install numpy pandas scipy plotly ttkbootstrap pydub librosa PyWavelets pyPCG-toolbox
```

**コマンドプロンプトから同じディレクトリでスクリプトを実行:**
```
python main.py
```

ヒント: ファイルを main.pyw にリネームすると、コマンドプロンプトを使用せずに実行できます。ダブルクリックして .exe ファイルのように起動できます。

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