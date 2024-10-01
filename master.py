from rag_pinecone.branding_rag import RAGbot
from langgraph_branding_agent.main import branding_agent
from langgraph_product_researcher.main import market_researcher

from web_scrape import get_links_and_text
from google_serper import serper_search
import os

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate

from langchain.tools import tool

from langchain_openai import ChatOpenAI
from langchain.chains.conversation.memory import ConversationBufferWindowMemory

from langchain.agents import initialize_agent

from datetime import datetime


from langchain_community.tools.tavily_search import TavilySearchResults



from dotenv import load_dotenv
load_dotenv()


os.environ['OPENAI_API_KEY'] =os.getenv("OPENAI_API_KEY")

os.environ['PINECONE_API_KEY'] = os.getenv("PINECONE_API_KEY")

os.environ['TAVILY_API_KEY'] = os.getenv("TAVILY_API_KEY")

# initialize LLM (we use ChatOpenAI because we'll later define a `chat` agent)
llm = ChatOpenAI(
        
        temperature=0.4,
        model_name='gpt-4o-mini'
)

# initialize conversational memory
conversational_memory = ConversationBufferWindowMemory(
        memory_key='chat_history',
        k=10,
        return_messages=True
)



# branding_agent(), market_researcher(), 
#google_search_results= serper_search(search_query)
@tool
def vector_store(query: str, namespace: str) -> str:
    """ 
    Pinecone Vectorstore
    This tool is used too retrieve contents of a specific course with input query and namespace. 
    
    courses and their namespaces are: 
    
    Alex Hormozi - $100M Leads - How to Get Strangers To Want To Buy Your Stuff: 'hormozicourse'
    
    Branding Strategies Course oon How To Write Effective Branding: 'brandingcourse'

    How to write effective AI sales letters: 'aisaleslettercourse' 
    Avatar Course: 'aiavatarcourse'  
    Positioning Course on ways to identify your best customer: 'postioningcourse'    
    
    """
    docs=  RAGbot.run(query, namespace)
   
    return docs
    

@tool
def website_scraper(url: str) -> str:
    """ 
    Website Scraper
    use this tool when you need to retrieve contents of a website to answer the question. 
    
    """
    website_contents=  get_links_and_text(url)
   
    return website_contents
    


@tool
def google_searcher(search_query: str) -> str:
    
  
    """ 
    Google Searcher
    use this tool when you need to retrieve knowledge from google.     
   
    
    """
    
    google_search_tool = TavilySearchResults(k=2, max_results=4)
    google_search_results= google_search_tool.invoke({"query": search_query})

    return google_search_results
    





  
tools = [vector_store, website_scraper, google_searcher ]


def master_agent(query:str):
    date_today = datetime.today()
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", f"""
            Imagine you are chatting with an individual sitting in front of you.\n
            You are very intelligent Business Developer Assistant, a versatile AI assistant that adapts to different mindsets—from analytical to creative.\n
            
            Pleease Answer shortly and to the point in conversational style. Always provide toop three best results. Be very concise. 
            

            Your role is to assist with business development, product research, market analysis, providing clear, concise, and actionable responses in a conversational style.\n

            You should use available tools for research and information gathering, including Google, web scraper and different courses to conduct market research, perform tasks, and product development.
            When reflecting on previous answers, adapting the reflection based on context or user input.
            Do not answer with your training knowledge.
        \n
            date today : {date_today}
        \n
            ## Output

            Keep responses short and clear, optimized for voice delivery. Don't hallucinate, if you don't know the answer, say you don't know. 

            """),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )


    agent = create_tool_calling_agent(llm, tools, prompt)

    agent_executor = AgentExecutor(agent=agent, tools=tools, memory=conversational_memory, verbose=True)

    result = agent_executor.invoke({"input": query})
    conversational_memory.save_context({"Human": query}, {"AI": result['output']})
    
    return result['output']



if __name__ == "__main__":
      print("## Welcome to the RAG chatbot")
      print('-------------------------------')
      
      while True:
        query = input("You: ")
        if query.lower() == 'exit':
            break
       
        
        result = master_agent( query)
        print("Bot:", result)