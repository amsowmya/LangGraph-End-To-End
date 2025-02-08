import os
from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import AzureChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv

from utils.tasks import read_tasks

load_dotenv()

AZURE_OPENAI_API_KEY = os.getenv('AZURE_OPENAI_API_KEY')
os.environ["AZURE_OPENAI_ENDPOINT"] = os.getenv('AZURE_OPENAI_ENDPOINT')
os.environ["AZURE_OPENAI_API_VERSION"] = "2023-06-01-preview"

class State(TypedDict):
    messages: Annotated[list, add_messages]
    userid: str
    

@tool
def retrieve_tasks(userid: str) -> str:
    """Returns all the tasks for the user"""
    return read_tasks(userid)


tools = [retrieve_tasks]

tool_node = ToolNode(tools)


def agent(state: State):
    llm = AzureChatOpenAI(
        openai_api_version="2023-06-01-preview",
        azure_deployment=os.getenv('AZURE_OPENAI_DEPLOYMENT'),
        azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
        streaming=True
    )
    
    system_messages = SystemMessage(
        f"You are a helpful AI assistant. The user's id is {state['userid']}"
        )
    
    messages = system_messages + state['messages']
    
    return {"messages": [llm.invoke(messages)]}

def create_graph():
    graph_builder = StateGraph(State)
    
    graph_builder.add_node("agent", agent)
    graph_builder.add_node("executor", tool_node)
    
    graph_builder.add_edge(START, "agent")
    graph_builder.add_edge("agent", "executor")
    graph_builder.add_edge("executor", END)
    
    return graph_builder.compile()

def main():
    graph = create_graph()
    
    userid = "YourTechBud"
    
    query = input("Your query: ")
    
    initial_state = {"messages": [HumanMessage(content=query)], "userid": userid}
    
    for event in graph.stream(initial_state):
        for key in event:
            print("\n*******************************************\n")
            print(key + ":")
            print("---------------------\n")
            print(event['messages'][-1].content)
            
            
if __name__ == "__main__":
    main()