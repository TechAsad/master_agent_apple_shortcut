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

prompt_business= """
“Identify the key demographics of our target audience for [product/service].”
Tip: Specify age, gender, income level, and other relevant factors.
2. “Analyze the market size and growth potential for [industry/market] in [location].”

Tip: Include current market trends and future projections.
3. “Create a competitor analysis report for [product/service] in [industry].”

Tip: List main competitors, their strengths, weaknesses, and market position.
4. “Develop a SWOT analysis for [company name] in the [industry] sector.”

Tip: Identify internal strengths and weaknesses, and external opportunities and threats.
5. “Research the latest trends in the [industry] and how they impact [company name].”

Tip: Focus on technological advancements, consumer behavior, and regulatory changes.
Understanding Your Audience
Knowing your audience is key to tailoring your products and marketing strategies.

“Survey potential customers to understand their needs and preferences regarding [product/service].”
Tip: Include questions about pain points, desired features, and buying behavior.
2. “Create customer personas for [product/service] based on market research data.”

Tip: Detail the personas’ demographics, motivations, challenges, and buying habits.
3. “Analyze customer feedback and reviews for [product/service] to identify areas for improvement.”

Tip: Categorize feedback into themes such as quality, usability, and customer service.
4. “Identify the primary channels through which our target audience discovers new products in [industry].”

Tip: Consider social media, search engines, online marketplaces, and word-of-mouth.
5. “Research the purchasing journey of our target audience for [product/service].”

Tip: Map out the stages from awareness to decision-making and post-purchase behavior.
Competitor Analysis
Understanding your competitors’ strategies can give you a competitive edge.

“Evaluate the marketing strategies of our top three competitors in the [industry].”
Tip: Analyze their advertising campaigns, content marketing, and social media presence.
2. “Compare the pricing strategies of competitors for similar products/services in [industry].”

Tip: Highlight the differences in pricing models, discounts, and value propositions.
3. “Analyze the product features and benefits offered by competitors in the [industry].”

Tip: Identify unique selling points and potential gaps in the market.
4. “Research the customer service and support strategies of competitors in [industry].”

Tip: Consider response times, support channels, and customer satisfaction levels.
5. “Evaluate the brand positioning and messaging of competitors in [industry].”

Tip: Assess how they differentiate themselves and communicate their value.
Market Trends and Opportunities
Staying updated on market trends helps you anticipate changes and capitalize on new opportunities.

“Identify emerging trends in the [industry] that could impact our business in the next 5 years.”
Tip: Focus on technological, economic, and cultural shifts.
2. “Analyze the impact of current economic conditions on the [industry].”

Tip: Include factors such as inflation, unemployment rates, and consumer spending.
3. “Research regulatory changes affecting the [industry] and how they could impact [company name].”

Tip: Highlight both opportunities and challenges posed by new regulations.
4. “Identify potential market gaps in the [industry] that [company name] could exploit.”

Tip: Look for unmet needs, underserved segments, and niche markets.
5. “Evaluate the potential for [product/service] expansion into new geographic markets.”

Tip: Assess market demand, competition, and entry barriers in the new regions.

"""



linkedin_ideas_prompts= """

1. To Write a Catchy Headline
Your LinkedIn headline is often the first thing people see, acting as a brief introduction to who you are and what you bring to the table. But many people are struggling to write a short, impactful sentence. If you're one of them.

Act as an experienced LinkedIn copywriter. Craft a headline for my LinkedIn profile that effectively showcases my [expertise], grabs attention, and communicates my unique value proposition. The headline should be professional, concise, and tailored to my target audience or industry. It needs to set me apart from others in the field while encapsulating the essence of my professional journey and aspirations. Consider SEO optimization to ensure it reaches the right audience on LinkedIn searches.

2. To Draft a Profile Summary
The next prompt is to write a profile summary. Because a good profile summary is just as important as a headline. So if you want to convey your professional journey and aspirations clearly.

Act as an experienced LinkedIn copywriter. Draft a compelling LinkedIn profile summary for me. The summary should accurately represent my professional background, skills, and accomplishments while also showcasing my unique value proposition. It should be tailored to appeal to [target industry], positioning me as a thought leader or expert in their field. Ensure that the tone is professional yet approachable and that it encourages networking and connection opportunities. Incorporate relevant keywords to enhance search visibility and alignment with industry trends.

3. To Suggest Relevant Skills
When it comes to LinkedIn, showcasing the right skills can make all the difference between being overlooked and being headhunted. But sometimes, we underestimate or even forget to list down some of our most valuable assets.


Act as a LinkedIn power user. Identify and suggest relevant skills that align with my [profession] and [expertise], enhancing my online presence and attractiveness to potential employers or clients. The suggested skills should be pertinent, and trend-aware, and should increase the chances of my profile being discovered by recruiters or industry professionals in search results.

4. To Identify Important Keywords
To stand out and get noticed, it's also important to have the right keywords mixed throughout your profile. But knowing which ones to use, is not so obvious. With the next ChatGPT prompt, you can easily find them and include them in your profile.

Act as a LinkedIn SEO expert. Conduct comprehensive research to identify SEO keywords that are relevant for [desired profession/industry/role] on LinkedIn. Your goal is to optimize my LinkedIn profile to increase visibility, attract potential employers or clients, and position me as a top professional in my field. Provide a list of at least 10 high-impact keywords along with their search volume and relevance. Additionally, offer suggestions on how to organically integrate these keywords into my LinkedIn profile sections, ensuring a natural flow of content while maximizing SEO benefits. Ensure that all recommendations adhere to LinkedIn's best practices and guidelines.

"""



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
    TThi tool will retrieve contents of a website and linkedin pages.
    scrape any url for information
    
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
    
    and other instructions and latest research.
    
    """
    
    
    reddit_comments= sub_agent(instructions)

    return reddit_comments
    

@tool
def business_workflow_instructions(say_anything: str) -> str:
    
  
    """ 
    this will give you the instruction on how to do the market analysis. 
    
    """
    
    
    business_prompt= prompt_business

    return business_prompt
    

@tool
def linkedin_ideas(say_anything: str) -> str:
    
  
    """ 
    this will give you the instruction on how to write linkedin profiles. 
    
    """
    
    
    linkedin_prompts= linkedin_ideas_prompts

    return linkedin_prompts
    


  
tools = [vector_store, website_scraper, google_searcher, reddit_comments_scraper, linkedin_ideas ]


def master_agent(query:str):
    print(conversational_memory.chat_memory)
    date_today = datetime.today()
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", f"""
           
You are an Experienced Business Developer with 20 years of experience in business development and Market research. You have access to some tools and instructions for how to do the research for the given product and business.

current date and time: {date_today}\n

current chat history:\n {conversational_memory.chat_memory}\n
       
Always use tools to gather latest up to date knowledge.
For research, use the available tools such as google_searcher, reddit_comments_scraper, website_scraper, and vector_store.
NOTE: Always response shortly with BEST results in few short sentences in summary format and PLAIN english. Do not use markdown writing style, NEVER use special characters and be very concise. NEVER bold any text with **.


INSTRUCTIONS for Market and competitor analysis:

“Identify the key demographics of our target audience for [product/service].”
Tip: Specify age, gender, income level, and other relevant factors.
2. “Analyze the market size and growth potential for [industry/market] in [location].”

Tip: Include current market trends and future projections.
3. “Create a competitor analysis report for [product/service] in [industry].”

Tip: List main competitors, their strengths, weaknesses, and market position.
4. “Develop a SWOT analysis for [company name] in the [industry] sector.”

Tip: Identify internal strengths and weaknesses, and external opportunities and threats.
5. “Research the latest trends in the [industry] and how they impact [company name].”

Tip: Focus on technological advancements, consumer behavior, and regulatory changes.
Understanding Your Audience
Knowing your audience is key to tailoring your products and marketing strategies.

“Survey potential customers to understand their needs and preferences regarding [product/service].”
Tip: Include questions about pain points, desired features, and buying behavior.
2. “Create customer personas for [product/service] based on market research data.”

Tip: Detail the personas’ demographics, motivations, challenges, and buying habits.
3. “Analyze customer feedback and reviews for [product/service] to identify areas for improvement.”

Tip: Categorize feedback into themes such as quality, usability, and customer service.
4. “Identify the primary channels through which our target audience discovers new products in [industry].”

Tip: Consider social media, search engines, online marketplaces, and word-of-mouth.
5. “Research the purchasing journey of our target audience for [product/service].”

Tip: Map out the stages from awareness to decision-making and post-purchase behavior.
Competitor Analysis
Understanding your competitors’ strategies can give you a competitive edge.

“Evaluate the marketing strategies of our top three competitors in the [industry].”
Tip: Analyze their advertising campaigns, content marketing, and social media presence.
2. “Compare the pricing strategies of competitors for similar products/services in [industry].”

Tip: Highlight the differences in pricing models, discounts, and value propositions.
3. “Analyze the product features and benefits offered by competitors in the [industry].”

Tip: Identify unique selling points and potential gaps in the market.
4. “Research the customer service and support strategies of competitors in [industry].”

Tip: Consider response times, support channels, and customer satisfaction levels.
5. “Evaluate the brand positioning and messaging of competitors in [industry].”

Tip: Assess how they differentiate themselves and communicate their value.
Market Trends and Opportunities
Staying updated on market trends helps you anticipate changes and capitalize on new opportunities.

“Identify emerging trends in the [industry] that could impact our business in the next 5 years.”
Tip: Focus on technological, economic, and cultural shifts.
2. “Analyze the impact of current economic conditions on the [industry].”

Tip: Include factors such as inflation, unemployment rates, and consumer spending.
3. “Research regulatory changes affecting the [industry] and how they could impact [company name].”

Tip: Highlight both opportunities and challenges posed by new regulations.
4. “Identify potential market gaps in the [industry] that [company name] could exploit.”

Tip: Look for unmet needs, underserved segments, and niche markets.
5. “Evaluate the potential for [product/service] expansion into new geographic markets.”

Tip: Assess market demand, competition, and entry barriers in the new regions.




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