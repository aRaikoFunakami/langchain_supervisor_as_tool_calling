# LangChain Supervisor as Tool Calling

このプロジェクトは、LangChain の `langgraph-supervisor` を使用して、四則演算、単位変換、映画情報検索などのタスクを実行する複数のエージェントを管理するシステムです。スーパーバイザーはエージェント間の調整を行い、タスクが効率的かつ正確に完了するようにします。

さらに、OpenAI Realtime API を使用したリアルタイム音声対話機能も提供しており、WebSocket経由で音声による対話的な操作が可能です。
[langchain_openai_voice](https://github.com/langchain-ai/react-voice-agent/tree/main/server/src/langchain_openai_voice)をテキスト対応可能なように修正して利用しています。

## 特徴

- **四則演算エージェント**: 加算、減算、乗算、除算を処理します。
- **単位変換エージェント**: メートルからフィート、摂氏から華氏、メートルからセンチメートルなどの単位変換を実行します。
- **TMDB検索エージェント**: 映画、テレビ番組、セレブリティ情報を検索します。
- **スーパーバイザー**: エージェント間のタスクの割り当てを管理し、すべての計算が適切なツールを使用して行われることを保証します。
- **リアルタイム音声対話**: OpenAI Realtime APIを使用した音声による対話機能
- **WebSocketサーバー**: ブラウザからのリアルタイム接続をサポート

## 必要条件

- Python 3.13
- 必要なPythonパッケージ（`pyproject.toml`で管理）:
  - `langchain` (>=0.3.27)
  - `langchain-openai` (>=0.3.29)
  - `langgraph-supervisor` (>=0.0.29)
  - `starlette` (>=0.47.2)
  - `uvicorn` (>=0.35.0)
  - `websockets` (>=15.0.1)
  - `langdetect` (>=1.0.9)
  - `tmdb_agent` (TMDBエージェント用)
- OpenAI API キー（環境変数 `OPENAI_API_KEY` に設定）

## インストール

1. リポジトリをクローンします:

   ```bash
   git clone https://github.com/aRaikoFunakami/langchain_supervisor_as_tool_calling.git
   cd langchain_supervisor_as_tool_calling
   ```

2. 依存関係を同期します:

   ```bash
   uv sync
   ```

## 使用方法

### テキストモード（従来の機能）

1. テストモードでサンプルタスクを実行:

   ```bash
   uv run python main.py --test
   ```

2. スーパーバイザーが処理するタスクの例:
   - `2*(3+4)/5` を計算
   - `5メートル` を `フィート` に変換
   - `3フィート` と `2メートル` を `センチメートル` で加算
   - `70kg` と `90kg` の合計体重を `ポンド` で計算
   - `トム・ハンクスが出演した映画を検索`
   - `スターウォーズについて教えてください`
   - `スターウォーズ1の公開年と2の公開年を足すと何年になるか？`

### リアルタイム音声モード（新機能）

1. リアルタイムサーバーを起動:

   ```bash
   uv run python main.py
   ```

2. ブラウザで `http://localhost:3000` にアクセス

3. "Connect" ボタンをクリックしてWebSocket接続を確立

4. 音声で以下のような質問ができます:
   - "What is 15 times 8?"（15×8は？）
   - "Convert 100 fahrenheit to celsius"（100華氏を摂氏に変換）
   - "Tell me about the movie Forrest Gump"（映画フォレストガンプについて教えて）
   - "What movies has Tom Hanks been in?"（トム・ハンクスの出演映画は？）

### コマンドラインオプション

```bash
# ヘルプを表示
uv run python main.py --help

# リアルタイムモード（デフォルト）
OPENAI_VOICE_TEXT_MODE=1 uv run python main.py

# テストモード（サンプルタスクを実行）
uv run python main.py --test
```

## プロジェクト構成

```text
langchain_supervisor_as_tool_calling/
├── main.py                    # メインスクリプト（エージェント、サーバー起動）
├── pyproject.toml            # プロジェクト設定と依存関係
├── uv.lock                   # 依存関係のロックファイル
├── README.md                 # プロジェクト説明書
├── .python-version           # Pythonバージョン指定
└── langchain_openai_voice/   # リアルタイム音声機能
    ├── __init__.py
    └── utils.py
```

### 主要ファイル

- **`main.py`**: エージェント、スーパーバイザー、リアルタイムサーバーを統合したメインスクリプト
- **`pyproject.toml`**: uvによる依存関係管理
- **`langchain_openai_voice/`**: OpenAI Realtime API用のカスタム実装

## 動作の仕組み

### 1. エージェント

- **ArithmeticAgent**: 四則演算を処理するツールを使用
  - `add`, `subtract`, `multiply`, `divide`
- **UnitConversionAgent**: 単位変換を処理するツールを使用
  - `meters_to_feet`, `celsius_to_fahrenheit`, `kilograms_to_pounds` など
- **TMDBSearchAgent**: 映画、テレビ番組、セレブリティ情報を検索
  - TMDB APIを使用した多言語対応検索

### 2. スーパーバイザー

- 適切なエージェントにタスクを自動割り当て
- エージェント間の連携を調整
- すべての計算がツールを使用して行われることを保証
- 複合タスクの順次処理をサポート

### 3. リアルタイム音声機能

- **OpenAI Realtime API**: `gpt-4o-realtime-preview`モデルを使用
- **WebSocketサーバー**: Starletteベースのサーバー実装
- **音声ストリーミング**: リアルタイム音声入出力
- **スーパーバイザー統合**: 音声コマンドを各エージェントに自動ルーティング

### 4. アーキテクチャ

```text
音声入力 → WebSocket → Realtime Agent → Supervisor → 専門エージェント → 結果
                                          ↓
ブラウザ ← WebSocket ← 音声出力 ← レスポンス ← 処理結果 ← ツール実行
```

## 環境設定

### 1. OpenAI API キー

```bash
export OPENAI_API_KEY="your-api-key-here"
```

### 2. オプション環境変数

```bash
# テキストモードでの動作（デバッグ用）
export OPENAI_VOICE_TEXT_MODE=1
```

## トラブルシューティング

### よくある問題

1. **OpenAI API キーが設定されていない**

   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```

2. **ポート3000が使用中**
   - `main.py`の`REALTIME_SERVER_PORT`を変更してください

3. **WebSocket接続エラー**
   - ファイアウォール設定を確認してください
   - ブラウザの開発者ツールでエラーログを確認してください

4. **音声が聞こえない**
   - ブラウザのマイクとスピーカーの許可を確認してください
   - HTTPSではなくHTTP（localhost）でアクセスしてください

## 注意事項

- **OpenAI API使用料**: Realtime APIの利用には課金が発生します
- **ネットワーク**: リアルタイム機能にはインターネット接続が必要です
- **ブラウザ対応**: モダンブラウザでのWebSocket対応が必要です
- **セキュリティ**: 本番環境では適切なセキュリティ設定を行ってください

## 技術仕様

### 使用技術

- **フレームワーク**: LangChain, LangGraph
- **Webサーバー**: Starlette, Uvicorn
- **リアルタイム通信**: WebSocket
- **音声AI**: OpenAI Realtime API (gpt-4o-realtime-preview)
- **パッケージ管理**: uv
- **言語**: Python 3.13

### APIエンドポイント

- **HTTP**: `http://localhost:3000/` - テストページ
- **WebSocket**: `ws://localhost:3000/ws` - リアルタイム音声通信

### 対応言語

- 日本語
- English
- その他（OpenAI Realtime APIがサポートする言語）

## 開発

### 開発環境のセットアップ

```bash
# リポジトリをクローン
git clone https://github.com/aRaikoFunakami/langchain_supervisor_as_tool_calling.git
cd langchain_supervisor_as_tool_calling

# 依存関係をインストール
uv sync

# 開発サーバーを起動
OPENAI_VOICE_TEXT_MODE=1 uv run python main.py
```

### カスタマイズ

- **エージェントの追加**: `main.py`で新しいエージェントを定義
- **サーバー設定**: `REALTIME_SERVER_HOST`, `REALTIME_SERVER_PORT`を変更
- **音声設定**: `REALTIME_AGENT_INSTRUCTIONS`でプロンプトを調整

## ライセンス

このプロジェクトはMITライセンスの下でライセンスされています
