import os
from typing import TypedDict, List, Union
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

# Configure Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
gemini = genai.GenerativeModel("gemini-2.5-flash")  # change to "gemini-1.5-pro" if needed


class AgentState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage]]


def process(state: AgentState) -> AgentState:
    """Process the latest user message with Gemini."""

    # latest user message
    user_msg = state["messages"][-1].content

    # Call Gemini
    response = gemini.generate_content(user_msg)

    # Gemini returns plain text → wrap as AIMessage
    ai_msg = AIMessage(content=response.text)

    # Append to state
    state["messages"].append(ai_msg)

    print(f"\nAI: {ai_msg.content}")
    print("CURRENT STATE:", state["messages"])

    return state


# Build LangGraph
graph = StateGraph(AgentState)
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END)
agent = graph.compile()


conversation_history: List[Union[HumanMessage, AIMessage]] = []

user_input = input("Enter: ")
while user_input != "exit":
    conversation_history.append(HumanMessage(content=user_input))

    result = agent.invoke({"messages": conversation_history})
    conversation_history = result["messages"]

    user_input = input("Enter: ")

# Save conversation log
with open("logging.txt", "w", encoding="utf-8") as file:
    file.write("Your Conversation Log:\n\n")

    for msg in conversation_history:
        if isinstance(msg, HumanMessage):
            file.write(f"You: {msg.content}\n")
        else:
            file.write(f"AI: {msg.content}\n\n")

print("Conversation saved to logging.txt")
