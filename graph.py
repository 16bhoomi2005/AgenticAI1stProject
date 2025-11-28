# graph_setup.py
from typing import TypedDict, Annotated
import os
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()


# --------------------------
# 1. State
# --------------------------
class State(TypedDict):
    messages: Annotated[list, add_messages]
    response: str


# --------------------------
# 2. Tools & LLMs
# --------------------------
search_tool = TavilySearch(max_results=2)
tools = [search_tool]

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=os.getenv("GROQ_API_KEY")
)

# Tool-aware model (VERY IMPORTANT)
llm_with_tools = llm.bind_tools(tools)


# --------------------------
# 3. Classifier
# --------------------------
classifier_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Classify the user message as one word ONLY: FACTUAL or CHAT."),
    ("user", "Message: {message}")
])

classifier_chain = classifier_prompt | llm


def classifier_node(state: State):
    last_user_msg = state["messages"][-1].content
    prediction = classifier_chain.invoke({"message": last_user_msg}).content.strip()
    return {
        "messages": [("assistant", prediction)]
    }


# --------------------------
# 4. Conversational Node
# --------------------------
def chat_node(state: State):
    response = llm.invoke(state["messages"])
    return {
        "messages": [response], 
        "response": response.content # Add this so main.py gets data
    }

# --------------------------
# 5. Tool Node
# --------------------------
tool_node = ToolNode(tools)


# --------------------------
# 6. Routing Logic
# --------------------------
def route(state: State):
    label = state["messages"][-1].content.lower()
    if "factual" in label:
        return "tool"
    return "chat"


# --------------------------
# 7. Final Answer Node
# --------------------------
def answer_node(state: State):
    answer = llm_with_tools.invoke(state["messages"])
    return {
        "messages": [answer],
        "response": answer.content
    }


# --------------------------
# 8. Build Graph
# --------------------------
def create_graph():
    g = StateGraph(State)

    g.add_node("classify", classifier_node)
    g.add_node("chat", chat_node)
    g.add_node("tool", tool_node)
    g.add_node("answer", answer_node)

    g.add_edge(START, "classify")

    g.add_conditional_edges(
        "classify",
        route,
        {
            "tool": "tool", 
            "chat": "chat" 
        }
    )

    g.add_edge("tool", "answer")
    
    # CHANGE: Chat should go directly to END, not answer
    g.add_edge("chat", END) 
    
    g.add_edge("answer", END)

    return g.compile()
graph=create_graph()