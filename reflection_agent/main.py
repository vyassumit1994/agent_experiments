from typing import List, Sequence, Annotated, TypedDict
from dotenv import load_dotenv
load_dotenv()

from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AnyMessage
from chains import generate_chain, reflect_chain

class MessageGraph(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


REFLECT = "reflect"
GENERATE = "generate"

def generation_node(state: MessageGraph):
    print ("state in generate node: ",state)
    ai_generated_content = generate_chain.invoke({"messages": state["messages"]})
    return {"messages": ai_generated_content} # Append newly generated content to messages key

def reflection_node(state: MessageGraph) -> List[BaseMessage]:
    print("state in reflect node: ", state)
    reflection_msg = reflect_chain.invoke({"messages": state["messages"]})
    return {"messages": [HumanMessage(content=reflection_msg.content)]} # Appending Message as Human Message (feedback) to the state


builder = StateGraph(state_schema=MessageGraph)
builder.add_node(GENERATE, generation_node)
builder.add_node(REFLECT, reflection_node)

builder.set_entry_point(GENERATE)


def should_continue(state: MessageGraph):
    if len(state['messages']) > 6:
        return END
    return REFLECT

builder.add_conditional_edges(GENERATE, should_continue, path_map={END: END, REFLECT: REFLECT})
builder.add_edge(REFLECT, GENERATE)

graph = builder.compile()
print (graph.get_graph().draw_mermaid())
graph.get_graph().print_ascii()


if __name__ == '__main__':
    print ("Hello Langgraph")
    inputs = {
        "messages": [HumanMessage(
                content="""Make this tweet better:"
                                    @LangChainAI
            — newly Tool Calling feature is seriously underrated.

            After a long wait, it's  here- making the implementation of agents across different models with function calling - super easy.

            Made a video covering their newest blog post

                                  """
            )]
    }
    response = graph.invoke(inputs)
    print(response)