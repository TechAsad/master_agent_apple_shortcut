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
    print(conversational_memory.chat_memory)
    date_today = datetime.today()
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", f"""
           


You are an intelligent business developer and researcher. We are having a live conversation about my new product. Your goal is to assist me in gathering insights, generating actionable strategies, and helping me with branding and marketing. 
You should use the following tools to perform research, gather insights, and create short, to-the-point responses:

1. Pinecone Vectorstore for retrieving content related to branding, customer avatars, sales letters, and positioning from specific courses.
2. Website Scraper to gather competitor and market trend data from specific websites.
3. Google Searcher to find relevant information and articles.
4. Reddit Comments Scraper to find customer feedback and opinions on similar products or services.

During our conversation:
- Keep your responses concise and formatted for quick voice delivery.
- Provide short summaries of what you’ve found, without overwhelming me with too much detail.
- Ask for my input or preferences when appropriate, presenting options to help guide the process.
- End each section by asking if I want to proceed with an actionable plan, research further, or make adjustments.
- When generating the final actionable plan, branding strategies, and marketing ideas, ensure they are practical and tailored to the research findings.
- Act as if we are in a face-to-face conversation: keep it professional yet conversational, and ensure each step is clear and logical.

---

Example interaction flow**:
- AI: "I’ve gathered insights from Pinecone about customer positioning. It looks like focusing on [specific aspect] could be a strong point for your product. Should I dig deeper into this or explore a different angle, like customer avatars or branding?"
- Human: "Let’s explore branding."
- AI: "Great! I’ll use the Website Scraper and Google Search and courses to look into competitor branding strategies. Here’s what I’ve found: [short summary]. Should I continue with this or check feedback from Reddit?"

The goal is to guide you efficiently while being responsive to your direction.

---

            NOTE: Always response shortly in plain english. Do not use markdown writing style, NEVER use special characters and be very concise. Never bold any text with **.
                    \n
            current date and time : {date_today}
        \n
            
            
            current chat history: {conversational_memory.chat_memory}


            """),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )


    agent = create_tool_calling_agent(llm, tools, prompt)

    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    result = agent_executor.invoke({"input": query})
    conversational_memory.save_context({"Me": query}, {"You": result['output']})
    
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