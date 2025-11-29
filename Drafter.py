from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv  
from langchain_core.messages import (
    BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
)
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# Global document storage
document_content = ""

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


# ---------------------- TOOLS ----------------------
@tool
def update(content: str) -> str:
    """Updates the document with the provided content."""
    global document_content
    document_content = content
    return f"Document updated.\nCurrent content:\n{document_content}"

@tool
def save(filename: str) -> str:
    """Save the current document to a text file."""
    global document_content

    if not filename.endswith(".txt"):
        filename += ".txt"

    try:
        with open(filename, "w") as f:
            f.write(document_content)

        print(f"\n💾 Saved to: {filename}")
        return f"Document has been saved successfully to '{filename}'."
    except Exception as e:
        return f"Error saving file: {str(e)}"


tools = [update, save]


# ---------------------- GEMINI MODEL ----------------------
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",   # or gemini-1.5-flash
    temperature=0.7
).bind_tools(tools)


# ---------------------- AGENT NODE ----------------------
def our_agent(state: AgentState) -> AgentState:
    global document_content

    system_prompt = SystemMessage(content=f"""
You are Drafter, a helpful writing assistant.
- To modify content, ALWAYS call the 'update' tool with the FULL new document.
- To save, call the 'save' tool.
- Always show current document content.
Current document:
{document_content}
""")

    # If first run → automatically greet
    if not state["messages"]:
        user_message = HumanMessage(content="I'm ready to update the document.")
    else:
        user_input = input("\nWhat would you like to do with the document? ")
        print(f"\n👤 USER: {user_input}")
        user_message = HumanMessage(content=user_input)

    all_messages = [system_prompt] + list(state["messages"]) + [user_message]

    response = model.invoke(all_messages)

    print(f"\n🤖 AI: {response.content}")
    if getattr(response, "tool_calls", None):
        print(f"🔧 TOOLS CALLED: {[tc['name'] for tc in response.tool_calls]}")

    return {"messages": list(state["messages"]) + [user_message, response]}



# ---------------------- ROUTING LOGIC ----------------------
def should_continue(state: AgentState) -> str:
    """End only when a save tool call completes."""
    for msg in reversed(state["messages"]):
        if isinstance(msg, ToolMessage) and "saved" in msg.content.lower():
            return "end"
    return "continue"



# ---------------------- GRAPH SETUP ----------------------
graph = StateGraph(AgentState)

graph.add_node("agent", our_agent)
graph.add_node("tools", ToolNode(tools))

graph.set_entry_point("agent")

graph.add_edge("agent", "tools")

graph.add_conditional_edges(
    "tools",
    should_continue,
    {
        "continue": "agent",
        "end": END,
    }
)

app = graph.compile()



# ---------------------- RUNNER ----------------------
def print_messages(messages):
    """Pretty print tool results."""
    for m in messages[-3:]:
        if isinstance(m, ToolMessage):
            print(f"\n🛠️ TOOL RESULT: {m.content}")


def run_document_agent():
    print("\n ===== DRAFTER =====")
    state = {"messages": []}

    for step in app.stream(state, stream_mode="values"):
        if "messages" in step:
            print_messages(step["messages"])

    print("\n ===== DRAFTER FINISHED =====")


if __name__ == "__main__":
    run_document_agent()
