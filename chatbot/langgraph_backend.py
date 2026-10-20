from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages


# LLM 
llm = ChatOpenAI(
    base_url='http://127.0.0.1:9931/v1',
    model='local qwen3.5:4b',
    api_key='not needed' # type: ignore
)


# State
class ChatbotState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# Nodes
def chat_node(state: ChatbotState):
    messages = state['messages']
    response = llm.invoke(messages)

    return {'messages': [response]}


# Graph
graph = StateGraph(ChatbotState)

graph.add_node('chat_node', chat_node)

graph.add_edge(START, 'chat_node')
graph.add_edge('chat_node', END)

checkpointer = InMemorySaver()
chatbot = graph.compile(checkpointer=checkpointer)

