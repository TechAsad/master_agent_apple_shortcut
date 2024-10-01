from langgraph_branding_agent.state import GraphState
from langgraph.graph import END, StateGraph, START
from langgraph_branding_agent.node import *


workflow = StateGraph(GraphState)

workflow.add_node("google_search", google_search)
workflow.add_node("subreddit_to_search", subreddit_to_search)  # 
workflow.add_node("subreddit_selector",subreddit_selector) 
workflow.add_node("web_summarizer",web_summarizer) # 
workflow.add_node("market_researcher", market_researcher)
workflow.add_node("branding_rag_search", branding_rag_search)

workflow.add_node("strategist", strategist)  # 


workflow.add_node("branding_creator",branding_creator)  # 




workflow.set_entry_point("google_search")
workflow.add_edge("google_search", "subreddit_to_search")

workflow.add_edge("subreddit_to_search","subreddit_selector")


workflow.add_edge("subreddit_selector","web_summarizer")
workflow.add_edge("web_summarizer","market_researcher")
workflow.add_edge("market_researcher","branding_rag_search")

workflow.add_edge("branding_rag_search", "strategist")

workflow.add_edge("strategist", "branding_creator")


workflow.add_edge("branding_creator", END)



# Compile
app = workflow.compile()