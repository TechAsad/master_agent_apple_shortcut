from rag_pinecone.branding_rag import RAGbot
from reddit_scraper.main import reddit_agent

from web_scrape import get_links_and_text
from google_serper import serper_search
import os
from subagent import sub_agent
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate

from langchain.tools import tool
from langchain_anthropic.chat_models import ChatAnthropic
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
os.environ['ANTHROPIC_API_KEY'] = os.getenv("ANTHROPIC_API_KEY")

# initialize LLM (we use ChatOpenAI because we'll later define a `chat` agent)
llm = ChatOpenAI( temperature=0.1,model_name='gpt-4o-mini')

#llm = ChatAnthropic(model="claude-3-5-sonnet", temperature=0.1)

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
    
    google_search_tool = TavilySearchResults(k=3, max_results=4)
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
    

@tool
def sub_agent_writer(instructions: str) -> str:
    
  
    """ 
    This is your assistant sub agent for business woorkflow writing, he needs details information about the product 
    which should have answer to these questions:
    [Product/Service]=
    [Avatar/Segment]=
    [Niche/Market]=
    [Context]=
    
    and other instructions.
    
    """
    
    
    reddit_comments= sub_agent(instructions)

    return reddit_comments
    



  
tools = [vector_store, website_scraper, google_searcher, reddit_comments_scraper, sub_agent_writer ]


def master_agent(query:str):
    print(conversational_memory.chat_memory)
    date_today = datetime.today()
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", f"""
           
You are a Business Developer Supervisor Agent. You have access to some tools and a sub agent which will write Business Workflow.

current date and time : {date_today}\n

current chat history: {conversational_memory.chat_memory}\n

NOTE: Do not use markdown writing style, NEVER use special characters and be very concise. Never bold any text with **.
                    
Always use tools to gather latest knowledge, do not use your training data.
If user asks for research, use the available tools such as google, pinecone, web scraper or reddit.
NOTE: Always response shortly with BEST results in one or two short sentences in summary format and PLAIN english. Do not use markdown writing style, NEVER use special characters and be very concise. NEVER bold any text with **.

If user wants to generate a business workflow:
Ask these questions one by one. One question at a time.
Content Brief:
[Product/Service]=
[Avatar/Segment]=
[Niche/Market]=
[Context]=
\n
Once you have all the answers and necessary information about the Content berief variables, Send them to your Assistant Sub Agent as instructions for generating Business Workflow.
return word to word response from the assitant sub agent. it will be a long workflow. 




            """),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )


    agent = create_tool_calling_agent(llm, tools, prompt)

    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    result = agent_executor.invoke({"input": query})
    conversational_memory.save_context({"Me": query}, {"You": result['output'][:1000]})
    
    return result['output']



if __name__ == "__main__":
      print("## Welcome to the Business Developer chatbot")
      print('-------------------------------')
      
      while True:
        query = input("You: ")
        if query.lower() == 'exit':
            break
       
        
        result = master_agent( query)
        print("Bot:", result)