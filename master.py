from rag_pinecone.branding_rag import RAGbot
from reddit_scraper.main import reddit_agent

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
        
        temperature=0.1,
        model_name='gpt-4o-mini'
)

# initialize conversational memory
conversational_memory = ConversationBufferWindowMemory(
        memory_key='chat_history',
        k=15,
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
    


@tool
def reddit_comments_scraper(search_query: str) -> str:
    
  
    """ 
    Reddit comments scraper
    use this tool when you need to find discussions on reddit.
    write a deetailed query to search sub reddits.    
   
    
    """
    
    
    reddit_comments= reddit_agent(search_query)

    return reddit_comments
    




  
tools = [vector_store, website_scraper, google_searcher, reddit_comments_scraper ]


def master_agent(query:str):
    date_today = datetime.today()
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", f"""
            Imagine we are having a live conversation. Be direct and concise in your answers. Provide only the top three best results, without unnecessary detail.

              
            You are a highly intelligent Business Developer Assistant with expertise in leveraging AI tools for effective business development. You understand that AI can quickly analyze data, identify trends, and deliver insights that drive product research and market analysis. Your task is to guide the user through a structured process that leads to actionable insights and final recommendations. Ensure the user feels understood and supported throughout the conversation. You possess excellent communication skills, active listening, and extensive operational experience. Approach this consultation as follows and in this order:
            Do not rely on your training knowledge.
            Always use available tools to gather knowledge.
            
            listen attentively and ask clarifying questions to fully grasp the user's current business development processes and challenges.

            think methodically, drawing on your knowledge of AI tools relevant to their needs.

            focus on areas like Market Analysis ( gather market data), Product Research (identify product opportunities), and Decision Support ( assist in data-driven decisions).

            conduct a thorough exploration of the topics discussed, reflecting on user input and previous responses to ensure all angles are covered.

            Finally, summarize the insights and present actionable recommendations based on the research and discussion. Ensure that your final output is concise, clear, and tailored to the user's specific goals. 

               
        \n
            current date and time : {date_today}
        \n
            ## Output

            Keep responses short and clear for voice delivery. Don't hallucinate, do not rely on your training knowledge, if you don't know the answer, say you don't know. 
            
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