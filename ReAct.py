from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv  
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

# ---------- TOOLS ----------
@tool
def add(a: int, b: int):
    """Adds two numbers"""
    return a + b

@tool
def subtract(a: int, b: int):
    """Subtracts two numbers"""
    return a - b

@tool
def multiply(a: int, b: int):
    """Multiplies two numbers"""
    return a * b

tools = [add, subtract, multiply]

# ---------- GEMINI MODEL ----------
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",     # or gemini-1.5-flash
    temperature=0.7
).bind_tools(tools)

# ---------- MODEL CALL NODE ----------
def model_call(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(
        content="You are my AI assistant. Answer clearly and use tools when needed."
    )
    result = model.invoke([system_prompt] + state["messages"])
    return {"messages": [result]}

# ---------- CONDITIONAL ROUTING ----------
def should_continue(state: AgentState):
    last = state["messages"][-1]
    if not last.tool_calls:
        return "end"
    return "continue"

# ---------- GRAPH ----------
graph = StateGraph(AgentState)
graph.add_node("agent", model_call)
graph.add_node("tools", ToolNode(tools))
graph.set_entry_point("agent")

graph.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "tools",
        "end": END
    }
)

graph.add_edge("tools", "agent")

app = graph.compile()

# ---------- STREAM OUTPUT ----------
def print_stream(stream):
    for step in stream:
        msg = step["messages"][-1]
        msg.pretty_print()

# ---------- RUN ----------
inputs = {
    "messages": [("user", "Add 40 + 12 and then multiply the result by 6. Also tell me a joke.")]
}

print_stream(app.stream(inputs, stream_mode="values"))
