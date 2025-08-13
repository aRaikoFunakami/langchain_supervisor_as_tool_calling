from langchain_core.tools import tool
from langchain.chat_models import init_chat_model
from langchain_core.messages import convert_to_messages
from langgraph_supervisor import create_supervisor
from langgraph.prebuilt import create_react_agent

from langchain_openai import ChatOpenAI

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../tmdb_agent")))
from tmdb_agent.agent import create_tmdb_agent

# Realtime Server関連のインポート
import uvicorn
from starlette.applications import Starlette
from starlette.responses import HTMLResponse
from starlette.routing import Route, WebSocketRoute

# langchain_openai_voiceのインポート
from langchain_openai_voice import OpenAIVoiceReactAgent

# Configuration constants
REALTIME_SERVER_HOST = "0.0.0.0"
REALTIME_SERVER_PORT = 3000
REALTIME_MODEL = "gpt-4o-realtime-preview"

# Realtime Agent Instructions Template
REALTIME_AGENT_INSTRUCTIONS = """
You are a helpful voice assistant with access to specialized capabilities through a supervisor agent.

CAPABILITIES:
- General conversation and simple questions: Handle directly with your built-in knowledge
- Mathematical calculations: Use supervisor_agent tool
- Unit conversions: Use supervisor_agent tool  
- Movie/TV/celebrity information: Use supervisor_agent tool

ROUTING RULES:
1. For simple greetings, casual conversation, and general knowledge: Respond directly
2. For ANY mathematical calculation (even simple addition): ALWAYS use supervisor_agent tool
3. For ANY unit conversion: ALWAYS use supervisor_agent tool
4. For ANY movie, TV show, or celebrity question: ALWAYS use supervisor_agent tool

EXAMPLES:
- "Hello" → Respond directly
- "How are you?" → Respond directly
- "What's the weather like?" → Respond directly (if you can provide general advice)
- "What is 2+3?" → Use supervisor_agent tool
- "Convert 5 meters to feet" → Use supervisor_agent tool
- "Tell me about Star Wars" → Use supervisor_agent tool
- "Who is Tom Hanks?" → Use supervisor_agent tool

Always be conversational and helpful. When using the supervisor tool, explain what you're doing.
Respond in the same language as the user's input.
"""

# WebSocket utility functions
async def websocket_stream(websocket):
    """WebSocketからのメッセージを非同期で受信するストリーム
    
    Args:
        websocket: StarletteのWebSocketインスタンス
        
    Returns:
        AsyncGenerator: WebSocketメッセージのストリーム
    """
    try:
        while True:
            message = await websocket.receive_text()
            yield message
    except Exception as e:
        print(f"WebSocket stream error: {e}")
        return

    

# 四則演算ツールの定義
@tool("add", description="Adds two numbers.")
def add(a: float, b: float) -> float:
    """Adds two numbers."""
    print(f"Adding {a} and {b}")
    return a + b

@tool("subtract", description="Subtracts the second number from the first.")
def subtract(a: float, b: float) -> float:
    """Subtracts the second number from the first."""
    print(f"Subtracting {b} from {a}")
    return a - b

@tool("multiply", description="Multiplies two numbers.")
def multiply(a: float, b: float) -> float:
    """Multiplies two numbers."""
    print(f"Multiplying {a} and {b}")
    return a * b

@tool("divide", description="Divides the first number by the second.")
def divide(a: float, b: float) -> float:
    """Divides the first number by the second."""
    print(f"Dividing {a} by {b}")
    if b == 0:
        return "Error: Division by zero"
    return a / b

# エージェントの初期化
arithmetic_agent = create_react_agent(
    model=init_chat_model("openai:gpt-4.1-mini"),
    tools=[add, subtract, multiply, divide],
    name="ArithmeticAgent",
    prompt=(
        "You are an Arithmetic Agent.\n\n"
        "INSTRUCTIONS:\n"
        "- Your task is to perform arithmetic operations such as addition, subtraction, multiplication, and division.\n"
        "- Do NOT perform any unit conversions or handle non-arithmetic tasks.\n"
        "- Respond ONLY with the numerical result of the calculation. Do NOT include any explanations or additional text.\n"
        "- If the input is invalid or cannot be calculated, respond with 'Error: Invalid input'."
    ),
)


# 単位変換ツールの定義
@tool("meters_to_feet", description="Converts meters to feet.")
def meters_to_feet(meters: float) -> float:
    """Converts meters to feet."""
    print(f"Converting {meters} meters to feet.")
    return meters * 3.28084

@tool("feet_to_meters", description="Converts feet to meters.")
def feet_to_meters(feet: float) -> float:
    """Converts feet to meters."""
    print(f"Converting {feet} feet to meters.")
    return feet / 3.28084

@tool("fahrenheit_to_celsius", description="Converts Fahrenheit to Celsius.")
def fahrenheit_to_celsius(fahrenheit: float) -> float:
    """Converts Fahrenheit to Celsius."""
    print(f"Converting {fahrenheit} Fahrenheit to Celsius.")
    return (fahrenheit - 32) * 5/9

@tool("celsius_to_fahrenheit", description="Converts Celsius to Fahrenheit.")
def celsius_to_fahrenheit(celsius: float) -> float:
    """Converts Celsius to Fahrenheit."""
    print(f"Converting {celsius} Celsius to Fahrenheit.")
    return (celsius * 9/5) + 32

@tool("kilograms_to_pounds", description="Converts kilograms to pounds.")
def kilograms_to_pounds(kg: float) -> float:
    """Converts kilograms to pounds."""
    print(f"Converting {kg} kilograms to pounds.")
    return kg * 2.20462

@tool("meters_to_centimeters", description="Converts meters to centimeters.")
def meters_to_centimeters(meters: float) -> float:
    """Converts meters to centimeters."""
    print(f"Converting {meters} meters to centimeters.")
    return meters * 100

@tool("centimeters_to_meters", description="Converts centimeters to meters.")
def centimeters_to_meters(centimeters: float) -> float:
    """Converts centimeters to meters."""
    print(f"Converting {centimeters} centimeters to meters.")
    return centimeters / 100

# エージェントの初期化
unit_conversion_agent = create_react_agent(
    model="openai:gpt-4.1-mini",
    tools=[meters_to_feet, celsius_to_fahrenheit, feet_to_meters, kilograms_to_pounds, meters_to_centimeters, centimeters_to_meters],
    name="UnitConversionAgent",
    prompt=(
        "You are an Unit Conversion Agent.\n\n"
        "INSTRUCTIONS:\n"
        "- Your task is to perform unit conversions such as meters to feet, Celsius to Fahrenheit, and so on.\n"
        "- Do NOT perform any arithmetic operations or handle non-unit conversion tasks.\n"
        "- Respond ONLY with the numerical result of the conversion. Do NOT include any explanations or additional text.\n"
        "- If the input is invalid or cannot be calculated, respond with 'Error: Invalid input'."
    ),
)

# 既存のAgentExecutorをsupervisor用に変換する関数
def adapt_agent_executor_for_supervisor(agent_executor, name, debug=False):
    """Unify normal/debug behavior:
    - Always extract user input via extract_user_input_multiple_patterns
    - Emit detailed debug logs only when debug=True
    """
    def supervisor_compatible_agent(input_data, config=None):
        """Supervisor-compatible agent wrapper"""
        try:
            # Debug: high-level shape
            if debug:
                print("=== TMDB Agent Debug Info ===")
                print(f"Input data keys: {list(input_data.keys())}")
                print(f"Input data type: {type(input_data)}")
                if "messages" in input_data:
                    messages_dbg = input_data["messages"]
                    print(f"Messages count: {len(messages_dbg)}")
                    print(f"Messages type: {type(messages_dbg)}")
                    for i, msg in enumerate(messages_dbg):
                        print(f"Message {i}: {type(msg)}")
                        if isinstance(msg, dict):
                            print(f"  - Keys: {list(msg.keys())}")
                            print(f"  - Role: {msg.get('role', 'N/A')}")
                            # Preview up to 100 chars
                            print(f"  - Content preview: {str(msg.get('content', 'N/A'))[:100]}")
                        else:
                            # Generic object preview
                            preview = getattr(msg, "content", str(msg))
                            print(f"  - Role: {getattr(msg, 'role', 'N/A')}")
                            print(f"  - Content preview: {str(preview)[:100]}")
                else:
                    print("No 'messages' key found")
                    print(f"Available keys: {list(input_data.keys())}")
                print("================================")

            # Always use the robust extractor
            user_input = extract_user_input_multiple_patterns(input_data)

            if not user_input:
                if debug:
                    print("=== Extended Debug: Full Input Data ===")
                    print(f"Full input_data: {input_data}")
                raise ValueError("ユーザー入力が見つかりません")

            if debug:
                print(f"抽出されたユーザー入力: {user_input}")

            # Invoke wrapped AgentExecutor
            result = agent_executor.invoke({"input": user_input})
            output = result.get("output", "検索結果を取得できませんでした")

            if debug:
                # Trim preview to avoid huge console output
                print(f"TMDB結果: {str(output)[:200]}...")

            # Return in supervisor message format
            messages = input_data.get("messages", [])
            updated_messages = messages + [{
                "role": "assistant",
                "content": output,
                "name": name
            }]

            return {**input_data, "messages": updated_messages}

        except Exception as e:
            # Error logging
            if debug:
                print(f"TMDBエージェントエラー: {str(e)}")
                import traceback
                traceback.print_exc()

            error_message = f"エラーが発生しました: {str(e)}"
            messages = input_data.get("messages", [])
            updated_messages = messages + [{
                "role": "assistant",
                "content": error_message,
                "name": name,
                "metadata": {"error": True}
            }]
            return {**input_data, "messages": updated_messages}

    # Provide name attribute and .invoke alias for supervisor
    supervisor_compatible_agent.name = name
    supervisor_compatible_agent.invoke = supervisor_compatible_agent
    return supervisor_compatible_agent

# ユーザー入力を抽出する関数
def extract_user_input_multiple_patterns(input_data):
    """input_data からユーザー入力を抽出する関数"""
    messages = input_data.get("messages", [])
    for message in reversed(messages):
        # メッセージが辞書型の場合
        if isinstance(message, dict) and message.get("role") == "user":
            return message.get("content", "")
        # メッセージがLangChainのHumanMessageクラスの場合
        elif hasattr(message, "content") and type(message).__name__ == "HumanMessage":
            return message.content
        # メッセージがクラス型でrole属性がある場合
        elif hasattr(message, "role") and hasattr(message, "content"):
            # HumanMessageの場合、roleは通常"human"または"user"
            if message.role in ["user", "human"]:
                return message.content
    
    # デバッグ情報を追加
    print("=== Extract Debug ===")
    for i, message in enumerate(messages):
        print(f"Message {i}: {type(message)}")
        if hasattr(message, "role"):
            print(f"  Role: {message.role}")
        if hasattr(message, "content"):
            print(f"  Content: {message.content}")
        print(f"  Type name: {type(message).__name__}")
    
    raise ValueError("ユーザー入力が見つかりません")


# TMDBエージェントの初期化
tmdb_agent = create_tmdb_agent(
    llm=ChatOpenAI(model="gpt-4.1-mini", temperature=0.1),
    verbose=True,
)

# アダプターを適用
tmdb_supervisor_compatible = adapt_agent_executor_for_supervisor(
    agent_executor=tmdb_agent.agent_executor,
    name="tmdb_search_agent"
)



# スーパーバイザーの初期化
supervisor = create_supervisor(
    model=init_chat_model("openai:gpt-4.1-mini"),
    agents=[
        arithmetic_agent,
        unit_conversion_agent,
        tmdb_supervisor_compatible,  # TMDBエージェントを追加
    ],
    prompt=(
        "You are a supervisor managing three agents:\n"
        "- ArithmeticAgent: Handles arithmetic operations and mathematical calculations.\n"
        "- UnitConversionAgent: Handles unit conversion tasks (length, weight, temperature, etc.).\n"
        "- TMDBSearchAgent: Handles movie, TV show, and celebrity information searches using TMDB API. "
        "Supports multilingual queries and can search for cast/crew information, plot details, release dates, ratings, etc.\n\n"
        
        "ASSIGNMENT RULES:\n"
        "1. Assign work to ONE agent at a time - do not call agents in parallel.\n"
        "2. Do not perform any work yourself - always delegate to appropriate agents.\n"
        "3. For arithmetic: Always use ArithmeticAgent for ALL calculations, even simple additions.\n"
        "4. For unit conversions: Use UnitConversionAgent.\n"
        "5. For movie/TV/celebrity queries: Use TMDBSearchAgent for any entertainment content questions.\n\n"
        
        "TASK ROUTING:\n"
        "- Movie/TV show information (plot, cast, release date, ratings) → TMDBSearchAgent\n"
        "- Celebrity/actor/director information → TMDBSearchAgent\n"
        "- Entertainment industry questions → TMDBSearchAgent\n"
        "- Mathematical calculations → ArithmeticAgent\n"
        "- Unit conversions → UnitConversionAgent\n\n"
        
        "COMPLEX TASKS:\n"
        "If a task involves multiple domains (e.g., unit conversion + arithmetic), handle in sequence:\n"
        "1. First use UnitConversionAgent for conversions\n"
        "2. Then use ArithmeticAgent for calculations\n"
        "3. Use TMDBSearchAgent if entertainment content is involved\n\n"
        
        "Always respond in the same language as the user's query."
    ),
    add_handoff_back_messages=True,
    output_mode="full_history",
).compile()

# Supervisorをツールとして使用するための関数
@tool("supervisor_agent", description="Handles complex tasks requiring arithmetic operations, unit conversions, or movie/TV/celebrity information searches. Use this tool for mathematical calculations, unit conversions, or entertainment content queries.")
def supervisor_tool(query: str) -> str:
    """
    Supervisorエージェントを呼び出すツール
    Args:
        query: ユーザーのクエリ（計算、単位変換、映画・TV番組情報検索など）
    Returns:
        処理結果の文字列
    """
    try:
        print(f"Supervisor tool called with query: {query}")
        
        # Supervisorに送信
        result_messages = []
        for chunk in supervisor.stream({
            "messages": [{"role": "user", "content": query}]
        }):
            for node_name, node_update in chunk.items():
                messages = convert_to_messages(node_update["messages"])
                if messages:
                    result_messages.extend(messages)
        
        # 最後のアシスタントメッセージを取得
        for message in reversed(result_messages):
            if hasattr(message, 'content') and message.content:
                # assistantまたはエージェント名付きのメッセージを探す
                if (hasattr(message, 'role') and message.role == 'assistant') or \
                   (hasattr(message, 'name') and message.name in ['ArithmeticAgent', 'UnitConversionAgent', 'tmdb_search_agent']):
                    return message.content
        
        return "申し訳ありませんが、処理できませんでした。"
    
    except Exception as e:
        print(f"Supervisor tool error: {str(e)}")
        return f"エラーが発生しました: {str(e)}"

# WebSocket エンドポイント関数
async def websocket_endpoint(websocket):
    """WebSocket経由でRealtime Agentとの通信を処理
    
    Args:
        websocket: StarletteのWebSocketインスタンス
    """
    try:
        await websocket.accept()
        
        # WebSocketストリームを作成
        browser_receive_stream = websocket_stream(websocket)
        
        # Realtime Agentを初期化して接続
        agent = _create_realtime_agent()
        await agent.aconnect(browser_receive_stream, websocket.send_text)
        
    except Exception as e:
        print(f"WebSocket endpoint error: {e}")
        if websocket.client_state.value == 1:  # CONNECTING or CONNECTED
            await websocket.close(code=1011, reason="Internal server error")


def _create_realtime_agent():
    """Realtime Agentを作成する内部関数
    
    Returns:
        OpenAIVoiceReactAgent: 設定済みのエージェントインスタンス
    """
    return OpenAIVoiceReactAgent(
        model=REALTIME_MODEL,
        tools=[supervisor_tool],
        instructions=REALTIME_AGENT_INSTRUCTIONS,
    )

# ホームページ関数
async def homepage(request):
    return HTMLResponse('wscat -c \"ws://127.0.0.1:3000/ws\"')


def pretty_print_message(message, indent=False):
    pretty_message = message.pretty_repr(html=True)
    if not indent:
        print(pretty_message)
        return

    indented = "\n".join("\t" + c for c in pretty_message.split("\n"))
    print(indented)


def pretty_print_messages(update, last_message=False):
    is_subgraph = False
    if isinstance(update, tuple):
        ns, update = update
        # skip parent graph updates in the printouts
        if len(ns) == 0:
            return

        graph_id = ns[-1].split(":")[0]
        print(f"Update from subgraph {graph_id}:")
        print("\n")
        is_subgraph = True

    for node_name, node_update in update.items():
        update_label = f"Update from node {node_name}:"
        if is_subgraph:
            update_label = "\t" + update_label

        print(update_label)
        print("\n")

        messages = convert_to_messages(node_update["messages"])
        if last_message:
            messages = messages[-1:]

        for m in messages:
            pretty_print_message(m, indent=is_subgraph)
        print("\n")

# Realtime Server関数
def start_realtime_server():
    """Realtime ServerをWebSocketサーバーとして起動
    
    Returns:
        bool: サーバー起動の成功/失敗
    """
    _print_server_info()
    
    app = _create_starlette_app()
    
    try:
        uvicorn.run(app, host=REALTIME_SERVER_HOST, port=REALTIME_SERVER_PORT)
        return True
    except Exception as e:
        print(f"Failed to start realtime server: {e}")
        return False


def _print_server_info():
    """サーバー起動情報を表示する内部関数"""
    print("Starting Realtime Server with Supervisor integration...")
    print(f"Server will be available at: http://{REALTIME_SERVER_HOST}:{REALTIME_SERVER_PORT}")
    print(f"WebSocket endpoint: ws://{REALTIME_SERVER_HOST}:{REALTIME_SERVER_PORT}/ws")
    print("\nThe realtime agent will:")
    print("- Handle simple conversations directly")
    print("- Use supervisor for mathematical calculations")
    print("- Use supervisor for unit conversions")
    print("- Use supervisor for movie/TV/celebrity information")
    print("\nPress Ctrl+C to stop the server.")


def _create_starlette_app():
    """Starletteアプリケーションを作成する内部関数
    
    Returns:
        Starlette: 設定済みのアプリケーションインスタンス
    """
    routes = [
        Route("/", homepage),
        WebSocketRoute("/ws", websocket_endpoint)
    ]
    
    return Starlette(debug=True, routes=routes)

def run_test_mode():
    """Test mode でSupervisorの機能をテスト実行"""
    tasks = [
        "Calculate 2*(3+4)/5",
        "Convert 5 meters to feet", 
        "Add 3 feet and 2 meters in centimeters",
        "Convert the sum of 70 kg and 90 kg to pounds",
        "Search for movies starring Tom Hanks",
        "スターウォーズについて教えて下さい",
        "スターウォーズ１の公開された年と２の公開された年を足すと何年になるか？",
    ]
    
    print("Running test mode with sample tasks...")
    for task in tasks:
        print(f"\nTask: {task}")
        for chunk in supervisor.stream({
            "messages": [{"role": "user", "content": task}]
        }):
            pretty_print_messages(chunk, last_message=True)

# メイン実行部分
def main():
    """メイン実行関数"""
    parser = argparse.ArgumentParser(description="Realtime-enabled Supervisor Agent")
    parser.add_argument(
        "--test", 
        action="store_true", 
        help="Run test tasks to verify the supervisor functionality"
    )
    
    args = parser.parse_args()
    
    if args.test:
        print("Running test mode...")
        run_test_mode()
    else:
        print("Starting Realtime mode (WebSocket server)...")
        success = start_realtime_server()
        if not success:
            print("Failed to start realtime server.")


if __name__ == "__main__":
    import argparse
    main()
