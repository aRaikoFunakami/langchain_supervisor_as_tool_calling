from langchain_core.tools import tool
from langchain.chat_models import init_chat_model
from langchain_core.messages import convert_to_messages
from langgraph_supervisor import create_supervisor
from langgraph.prebuilt import create_react_agent


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

# スーパーバイザーの初期化
supervisor = create_supervisor(
    model=init_chat_model("openai:gpt-4.1-mini"),
    agents=[
        arithmetic_agent,
        unit_conversion_agent,
    ],
    prompt=(
        "You are a supervisor managing two agents:\n"
        "- ArithmeticAgent: Handles arithmetic operations.\n"
        "- UnitConversionAgent: Handles unit conversion tasks.\n"
        "Assign work to one agent at a time, do not call agents in parallel.\n"
        "Do not do any work yourself.\n"
        "Ensure that all calculations, including additions, are performed using the ArithmeticAgent.\n"
        "Do not perform any calculations yourself, even if the result seems simple.\n"
        "If a task involves both unit conversion and arithmetic, first use the UnitConversionAgent for conversions,\n"
        "then delegate all arithmetic operations to the ArithmeticAgent."
    ),
    add_handoff_back_messages=True,
    output_mode="full_history",
).compile()



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

# スーパーバイザーの使用例
if __name__ == "__main__":
    tasks = [
        "Calculate 2*(3+4)/5",
        "Convert 5 meters to feet",
        "Add 3 feet and 2 meters in centimeters",
        "Convert the sum of 70 kg and 90 kg to pounds",
    ]
    tasks = tasks[3:4]

    for task in tasks:
        print(f"Task: {task}")
        for chunk in supervisor.stream(
            {
                "messages": [
                    {"role": "user", "content": task}
                ]
            }
        ):
            pretty_print_messages(chunk, last_message=True)
