# LangChain Supervisor as Tool Calling

このプロジェクトは、LangChain の `langgraph-supervisor` を使用して、四則演算や単位変換などのタスクを実行する複数のエージェントを管理する仕組みを示しています。スーパーバイザーはエージェント間の調整を行い、タスクが効率的かつ正確に完了するようにします。

## 特徴

- **四則演算エージェント**: 加算、減算、乗算、除算を処理します。
- **単位変換エージェント**: メートルからフィート、摂氏から華氏、メートルからセンチメートルなどの単位変換を実行します。
- **スーパーバイザー**: エージェント間のタスクの割り当てを管理し、すべての計算が適切なツールを使用して行われることを保証します。

## 必要条件

- Python 3.13
- 必要なPythonパッケージ:
  - `langchain`
  - `langgraph-supervisor`
  - `openai`

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

1. メインスクリプトを実行します:

   ```bash
   uv run python main.py
   ```

2. スーパーバイザーが処理するタスクの例:
   - `2*(3+4)/5` を計算
   - `5メートル` を `フィート` に変換
   - `3フィート` と `2メートル` を `センチメートル` で加算
   - `70kg` と `90kg` の合計体重を `ポンド` で計算

## プロジェクト構成

- `main.py`: エージェントとスーパーバイザーを初期化し、タスクの実行を示すメインスクリプト。
- `.python-version`: プロジェクトで使用されるPythonバージョンを指定。

## 動作の仕組み

1. **エージェント**:
   - `ArithmeticAgent`: 四則演算を処理するツールを使用。
   - `UnitConversionAgent`: 単位変換を処理するツールを使用。

2. **スーパーバイザー**:
   - 適切なエージェントにタスクを割り当て。
   - すべての計算がツールを使用して行われることを保証。

3. **ツール**:
   - `@tool` デコレーターを使用して定義。
   - 例: `add`, `subtract`, `meters_to_feet`, `meters_to_centimeters`。

## 注意事項

- LLMベースのエージェントを実行するには、OpenAI APIへのアクセスが必要です。
- `main.py` 内の `tasks` リストを変更して、さまざまなシナリオをテストできます。

## ライセンス

このプロジェクトはMITライセンスの下でライセンスされています
