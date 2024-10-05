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
from langchain_anthropic.chat_models import ChatAnthropic
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
#llm = ChatOpenAI(temperature=0.1,      model_name='gpt-4o-mini')

llm = ChatAnthropic(model="claude-3-5-sonnet-20240620", temperature=0.1, max_tokens_to_sample=5000)
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


def sub_agent(query:str):
    #print(conversational_memory.chat_memory)
    date_today = datetime.today()
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", f"""
           
current date and time : {date_today}\n

Content Brief

[Product/Service]=
[Avatar/Segment]=
[Niche/Market]=
[Context]=

Countinue to the following once you have got the answers. If needed, Gather additional knowledge using tools provided.

\n
Section 1
Buyer’s Brief
“Start Here”
[PART 1]=
Buyer’s Brief
> Act as a world class marketer and customer researcher. You understand the human brain and marketing psychology behind driving buying decision. Identify Main Market [Segments] for [Product/Service] based on [Context] above. Analyze [Segment].
To begin, start the resort with [Product/Service] Buyer’s Brief for [Segement] in [Niche]
Next, start with analyzing the segmented audience, and understand their challenges and pain points, interests, and behaviors. This includes the core [Demographic], [Psychographic], [Needs], [Wants]. [Desires], [Behaviors], and [Pain Points]. You will go deep on the psychosocial elements of what drives this [Segment] of the market.
Develop a thorough understanding of your prospects' pain-points by creating comprehensive prompts surrounding their primary and secondary goals and complaints related to [Segment] and [Niche] based on [Context].
1. Primary Goal: Create (1) [Primary Goal] for [Segment] related to [Niche]. This should be in reference to the prior understanding of the analysis conducted already.
2. Primary Complaint: What are the top 5 [Primary Complaints] of prospects related to segment's industry and niche?
3. Primary Complaints Topic: What is the [Topic] for the [Primary Complaints] of Segment’s market?
4. Secondary Complaints: What are the top 5 [Secondary Complaints] of prospects related to segment's niche?
5. Secondary Complaints Topic: What is the [Topic] for the [Secondary Complaints] of Segments market ?
6. Goals: Secondary Goals - What are the 5 important [Secondary Goals] that [Segment] may have in relation to their [Primary Goal]? These goals should be closely tied to the primary goal and provide additional context and objectives.
7. Using the provided format, conduct a deep dive into your target audience's [Dreams], [Objections], [Negative Feelings], [Ultimate Fear], [Promises], [Mistaken Beliefs], and [Causes] related to[Segment] in [Niche].
Be sure to provide a comprehensive overview, focusing on each aspect, and providing specific and actionable insights. Remember to use the provided list formats, including the top 5 for each category, and provide examples that will resonate with the reader.
8. Goals: Goal Descriptors - As a marketer, what are the 5 most compelling [Goal Descriptors] to describe [segment]'s journey towards their primary goal using [adverbs]? These should be brief and impactful, promising progress and results.
Finally, include a detailed explanation of the [Primary Cause Defined] and why it is a roadblock to your segment's success.
9. Solutions: Expensive Alternatives
Generate a list of 10 expensive alternatives that segment may have considered once to reach the primary goal. Also, identify their drawbacks, starting with 1.
10. Offers: Offer Benefits
List down 10 benefits that segment will get from using [Offer Name] in a with/without format.
3. Offers: Offer Reframe
Create 10 Offer Reframes for [Offer Name] using the format: "It isn't just ____, it's _____."
11. Offers: Offer Target
Develop a list of 10 [Offer target] for [Offer Name] and explain why it's the right target. Focus on the target audience and how it will benefit from the offer. Include any variables in square brackets, for example, [product
> COMMAND: Reference the top of the recipe "Start Here". Fill out [Content Brief]. Execute [Part 1] then each section to complete this content. After the sequence stops, if not complete, continue on to the next Part of the sequence in order, building off the prior content to establish a better, more accurate offer for [segment] in [Niche] with [Context]. Only execute the command from the Marketer to give direct information related to each Part. Use the insights to drill down to the core of this offer using visceral emotional language. No fluff. Just actionable marketing insights and data:
Section 2
$100M Offer
Part 1: Picking The Right Market/Niche
To identify the best market/niche for [product], I would start by outlining criteria such as:
●	Target prospects experiencing intense pain points or unmet desires
●	Prospects have high buying power and ability to pay for solutions
●	Easy to reach and serve prospects at scale
●	Rapidly growing market with increasing demand
Based on this, I would identify [specific audience] as the ideal target market for [product] because they are struggling with [list specific pain points] and are desperately seeking solutions. This audience has sufficient income and budgets to invest in [price range] solutions. I can effectively target them by [list channels] where they currently spend time. Additionally, this is a rapidly expanding market expected to grow [XX%] over the next 5 years as [explain growth drivers].
To better understand this audience, I would aim to uncover:
●	Their deepest pain points and unmet desires: [list 2-3]
●	What they hope to achieve or avoid with a solution: [list 2-3]
●	The problems a solution must solve: [list 2-3]
●	Their decision making motivators: [list 2-3]
This data will allow me to position [product] as the ideal solution that directly speaks to their needs and achieves the outcomes they desire most.
Part 2: Picking a niche [Market] → [Niche 1] → [Sub-niche 1] → [Sub-niche 2]
For this niche, I would offer [type of solutions] tailored to [sub-niche 2] who are dealing with [key problem]. This is a subset of [sub-niche 1] within [niche 1] in the broader [market]. The audience profile is [describe target audience details].
Some potential solutions for their needs could include:
●	[Solution 1]
●	[Solution 2]
●	[Solution 3]
These types of hyper-targeted solutions would resonate strongly by addressing their precise pain points. The marketing could emphasize how it solves their specific problems better than any alternatives.
Part 3: Pricing & Maximizing Value
To determine ideal pricing, first identify the core value and benefits delivered by [product], such as:
●	[Key benefit 1]
●	[Key benefit 2]
●	[Key benefit 3]
Then list out all the possible ways [product] could be used to provide value:
1.	[Use case 1]
2.	[Use case 2]
3.	[Use case 3]
4.	[Use case 4]
5.	[Use case 5]
Rank those use cases by importance to determine the most persuasive benefits to emphasize. Finally, research competitors' pricing to find the sweet spot that maximizes perceived value.
Part 4: Maximizing Value Equation
To maximize the value equation, first identify desirable outcomes like: [list 2-3 outcomes]. These are worth solving because [explain importance].
Then showcase success stories and testimonials that demonstrate high likelihood of achievement. Provide specific examples like [example case study].
Next, highlight the fast time-to-results through [explain quick wins] that show visible progress.
Finally, make the product as easy and convenient to use as possible by [explain frictionless features]. This plan will amplify perceived value and lifetime value.
Part 5: Create Your Offer Components
The dream outcome clients want is to [describe aspirational goal]. They want to be seen as [explain desired perception].
Potential problems blocking this are:
●	[Problem 1]
●	[Problem 2]
●	[Problem 3]
Solutions could include:
●	[Creative solution 1]
●	[Creative solution 2]
●	[Creative solution 3]
Potential names for these bonus products are:
●	[Catchy name 1]
●	[Catchy name 2]
●	[Catchy name 3]
Part 6: Trim & Stack
High value solutions to keep are:
●	[Solution 1]
●	[Solution 2]
●	[Solution 3]
Low value solutions to cut are:
●	[Solution 4]
●	[Solution 5]
High cost solutions are:
●	[Solution 1]
Low cost solutions are:
●	[Solution 2]
●	[Solution 3]
Start with [Solution 1] as the high cost, high value core offer. Then maximize satisfaction with additional low cost, high value solutions [Solution 2] and [Solution 3].
Part 7: Add Scarcity
Here are 3 scarcity strategies:
1.	Only accepting the first [X] customers who sign up
2.	Closing enrollment after [X] spots are filled
3.	Limiting access to [X] customers per month
Communicate via urgency messaging on website/sales pages.
Part 8: Add Urgency
Here are 4 urgency messages:
1.	Next class starts soon! Enroll by [date] to secure your spot.
2.	Special pricing ends [date]! Get [discount] if you sign up now.
3.	Only [X] spots remain at this price! Claim yours today.
4.	Don't miss this chance! Opportunity shrinking fast as seats fill. Sign up ASAP!
Part 9: Enhancing Offer (Bonuses)
Bonuses:
1.	[Bonus 1] - [Name 1]
2.	[Bonus 2] - [Name 2]
3.	[Bonus 3] - [Name 3]
4.	[Bonus 4] - [Name 4]
5.	[Bonus 5] - [Name 5]
Part 10: Guarantee
No questions asked refund guarantee for peace of mind.
Conditional guarantee: Complete [requirement] to get [reward]
Other conditional guarantee ideas: [ideas]
Anti-guarantee for high-level mastermind emphasizes exclusivity.
Implied guarantee: [example]
Part 11: MAGIC Headline Formulas
[Product]
Bonus 1: [Name] - [Avatar] will [Goal] in just [Time] with [Container]
Bonus 2: [Name] - [Avatar] can now [Goal] in [Time] with [Container]
Bonus 3: [Name] - [Avatar] can finally [Goal] in [Time] with [Container]

> Execute [Part 1] - [Part 11] of Section 2 in sequence based off Section 1, each part building off the last to complete the execution


Section 3
Creative Brief: Boiling it all down

THE PROMISE 
What truth can you promise your client or customer?

THE PROBLEM 
Define what problem your product, service, or business solves
What problem does your client or customer have that keeps them up at night? 

THE SOLUTION 
Define a clear & simple answer to how you solve the problem. This should be different than the promise and involves the “how” 
Elaborate on the problem. Who has it? What are the existing inferior solutions? 
Elaborate on the solution in detail.

THE AUDIENCE 
List the folks that need your product. Name their top 3 problems next to them. Then name the three benefits they need the most ranking them from most important to them to least. Then next to that benefit place the feature that delivers that benefit. This should help you in your thinking about Google Ad Words as well. Who are we targeting and what are associated keywords and categories our customers might self-identify with.
> Who are our top 3 customer avatars? 

CUSTOMER 1: 
TOP 3 PROBLEMS 
> List the top 3 problems our customer is experiencing 
FEATURES
> List the features and benefits for this customer 
BENEFITS

CUSTOMER 2: 
TOP 3 PROBLEMS 
> List the top 3 problems our customer is experiencing 
FEATURES
> List the features and benefits for this customer 
BENEFITS

CUSTOMER 3: 
TOP 3 PROBLEMS 
> List the top 3 problems our customer is experiencing 
FEATURES
> List the features and benefits for this customer 
BENEFITS

SALES ARGUMENT 
Write a long-form essay as if you had unlimited time to speak to all of these customers. Describe in detail their problems, your solution to those problems and why/how you can service that need. Get your entire pitch out here in detail. 
> Write a long-form essay as if you had unlimited time to speak to all of these customers.
>elaborate on the above
>elaborate on how [Product/Service] helps each of our customer avatars in detail 

DEFINE YOUR USP (UNIQUE SELLING PROPOSITION) 
This is what will make you a BRAND. All of your customers should fit under the umbrella of your USP. Do not overcomplicate this. 

Questions to Consider when coming up with your USP: 
> What is the New Category we have created? 
> What does that mean?  
> How does this help my customers or clients? 
> What are the results my customers or clients can expect? 
> What is the “secret sauce” that makes my offer unique? 
> How can we support that claim? (Demo, science, testimonials, outside experts, etc).

REASONS FOR
SUPPORT

OFFERS 
What are 3 ways you can offer your product? The word risk here is relating to the customers risk. What can you offer that will make them say yes! Which consumer segment should respond to which offer. (TEST THEM ALL).
LOW RISK 
- Trial? (time or price)
- Contest Entry? 
- Sample for Shipping?
MEDIUM RISK 
- Payments? (monthly vs annual)
- Low / Discounted Price? 
- Money Back Guarantee?
HIGH RISK 
- Full Price? 
- Deluxe Upgraded Offer? 
- 2-Product Offer? 
- Continuity? 
- Subscription?

Accessories-Related Products (add-ons that fit the original purchase). 
What are other like items that would improve the experience of purchasing your initial offer? They should as a guide be less expensive....but not as a rule. 
Other accessories that would improve the experience of purchasing your initial offer include: 

3 Ideas for campaigns. 

> Create a list of 10 creative unique marketing ideas to promote our product
> Create a list of short online marketing campaign ideas for how we can market this product to get new customers 

=> Execute [Creative Brief] and all prompts inside to create a comprehensive Creative Brief for the business/offer in Section 1 and Section 2

Section 4
Creative Brief: Brand Style Guide

Act as a brand strategist. Your task is to create a comprehensive brand style guide based on the provided content while incorporating the brand's core avatar, niche, and unique selling proposition. Start by selecting the approved fonts, logos, and Pantone colors which best align with your brand values, and create a set of guidelines for your creative team to work under. These guidelines should include using A.I. or Tiff files instead of jpegs, maintaining a consistent visual format when displayed on black or white backgrounds. Additionally, you should provide directives for graphic layouts, packaging, and communications to ensure the consistent use of brand elements.

Once you have established the visual guidelines, it's time to define the tone and voice of the brand. Consider your brand as a person - how would you describe its personality and character? Think about the emotions and feelings that you want to evoke in your target audience and reflect this in your brand's messaging. Use the Character Diamond framework to help create a personality for your brand.

=> Execute Section 4 in line with Section 1, Section 2 and Section 3

Section 5
$100M Leads Workflow
[Content Brief]=
[business_name]=[The name of your company]
[offering]=[A brief description of the product or service you offer.]
[avatar]=[A short profile of your ideal target customer.]
[niche]=[The niche/market you are targeting.]
[pain_points]=[Describe the current pain or challenges your Avatar is experiencing.]
[resources]=[An overview of your team size and budget.]
[goals]=[Your revenue or growth goals.]
[current_stage]=[Where you're at today in terms of customers and revenue.]
[platforms]=[Marketing platforms you currently use.]
[metrics]=[Key metrics you track.]
[pricing]=[How your product/service is priced.]
[formats]=[How it's packaged and sold.]
[delivery]=[Your distribution method.]
[brand_voice]=[Describe your brand voice.]
=> Execute [Content Brief] for this Offer based on Sections 1-4



=>command1="Act as an advertising consultant for [business_name], which offers [offering]. Recommend 2-3 effective lead magnets to attract high-value [avatar]. Consider ways AI can be incorporated into lead magnets to create more meaningful and unique offerings to help [business_name] stand out and help [avatar] overcome [pain_points] while leading them to [offering]."
=>command2="Draft an outreach message for [business_name] in the format of [desired format] to [avatar]. Use the background info [sample background] to ensure it's personalized, complimentary, and value-driven."
=>command3="List 3-4 messaging statements that [business_name] can use to quickly build trust with [avatar], emphasizing their core value."
=>command4="Design a 3-step follow-up sequence for [business_name] to engage [avatar] across different contact methods. The content should aim to start a conversation without coming across as too salesy."
=>command5="Craft a statement for [business_name] to request referrals from warm connections regarding their [offering]. Generate 3 variations."
=>command6="Suggest 3-5 content formats and styles for [business_name] to maximize engagement from [avatar] on [platforms], including a brief strategy for each."
=>command7="Generate 5 engaging headlines for [business_name] targeting [avatar], using tactics like curiosity triggers, numbers, and celebrity."
=>command8="Outline 3 structures for long-form content for [business_name] to reward [avatar] using tactics like lists and stories."
=>command9="Create 5 promotional statements for [business_name] to integrate into a content piece about [content topic], directing [avatar] to their [offering]."
=>command10="Identify 3 sources for [business_name] to build targeted contact lists for [avatar]. Provide criteria for each source."
=>command11="Advise on how [business_name] can automate their cold outreach to [avatar]. Suggest 2-3 components for automation."
=>command12="Develop an ad copy framework tailored to [business_name]'s [offering] for [avatar], incorporating PAS and AIDA frameworks. Expand on each of these frameworks to create 3 ads for each related to [avatar] in [niche] with [pain_points] for [business_name] with  [offering]"
=>command13="Propose 3-5 offer structures for [business_name] to achieve client-financed acquisition via paid ads."
=>command14="Provide 3-4 language examples for [business_name] to set expectations with customers, aiming to overdeliver."
=>command15="Suggest 2 sources of post-purchase value that [business_name] can offer customers to motivate referrals."
=>command16="Detail 2-3 incentive ideas for [business_name] to stimulate referrals from existing customers."
=>command17="Based on the book's Roadmap 7 Levels to Scale: 
Roadmap 7 Levels to Scale. 
1.	Friends know about your offer
2.	Max out personal outreach
3.	Hire team to scale outreach
4.	Focus on product for referrals
5.	Add new platforms and methods
6.	Hire executive team per method
7.	$100M+ revenue. Profitable paid ads at scale across methods and platforms
Level 1 Prompt: "Summarize key focus areas at Level 1 for businesses with initial product-market fit. Recommend 2-3 goals and their measurement metrics." 
Level 2 Prompt: "Guide [business_name] on maximizing personal outreach at Level 2, including volume, metrics, team hires, and progression indicators." 
Level 3 Prompt: "For Level 3, delineate strategies for [business_name] focusing on team hiring for outbound lead gen." 
Level 4 Prompt: "Advise on product enhancements at Level 4 for [business_name]'s [offering] to boost organic referrals and measure readiness for scaling." 
Level 5 Prompt: "Strategize for [business_name] on Level 5 completion, focusing on platform expansion, placements, and audience metrics." 
Level 6 Prompt: "Suggest executive team hiring and structuring for [business_name] aiming for $100M+ revenue by Level 6."
=>command18="Recommend enhancements to [offering] for [business_name] at Level 4 to boost organic referrals."
=>command19="Strategize for [business_name] on completing Level 5, focusing on expanding platforms and audiences."
=>command20="Detail executive team hiring recommendations for [business_name] tailored to their [offering] as they approach Level 6."
⇒ [EXECUTE] [REFERENCE] to [result]
[result]=<command1>=><command2>=><command3>=><command4>=><command5>=><command6>=><command7>=><command8>=><command9>=><command10>=><command11>=><command12>=><command13>=><command14>=><command15>=><command16>=><command17>=><command18>=><command19>=><command20>.
For each command, execute the prompt as it related to the [business_name] and [offering] aligned with [avatar] and [goals] from [CONTEXT FRAMEWORK]. Execute each in sequential order, each one building off the last and everything addressing the key [variables] related to [avatar] in [niche] to help [company] reach more [avatar] and hit [goals]. Apply [brand_voice] across <commands>. Write out the full expected final outcome of the copy for each section, expanding on the prompt to create the desired end result. IMPORTANT: When you execute, do NOT include the <command> for each section. Do NOT include the word <command> in any of the output response. Instead, produce a headers for each <command> that identifies that as a new section. The beginning of the output should begin with $100M Leads Workflow for [business_name]. Format each section with blog or whitepaper formatting with no syntax and execute the format so it is clean and easy to read and understand what each section entails. 
When you get to the Roadmap 7 Levels to Scale. 
List out each level in accordance with what has been produced in the workflow above to draw on the same context and apply it for each prompt.
Level 1: Friends know about your offer
Level 2: Max out personal outreach
Level 3: Hire team to scale outreach
Level 4: Focus on product for referrals
Level 5: Add new platforms and methods
Level 6: Hire executive team per method
Level 7: $100M+ revenue. Profitable paid ads at scale across methods and platforms
⇒ Execute [result] 
Section 6
[GO TO MARKET ADS CONTENT BRIEF]
[Client/Company Name]= 
[Offer/Product Being Advertised]= 
[Campaign Objective]= 
[Target Success Metrics]=

[Audience Age Range]= 
[Audience Gender]= 
[Audience Location]=
[Audience Interests/Behaviors]= 
[Audience Goals/Needs]= 
[Audience Pain Points]=

[Ad Platform]= 
[Ad Placement Type]= 
[Ad Format]=
[Ad Tone]= 

[Brand Voice]= 
[Brand Colors]= 
[Other Brand Guidelines]=

[Campaign Timeline]= 
[Campaign Budget]=

[Special Offers/Promotions]= 
[Competitor Examples]=

Run the prompt to be dynamically populated with the provided variables:

=> Welcome [Client/Company Name]! Let's create effective ads to promote your [Offer/Product] to your target audience.

Your campaign goal is to [Campaign Objective] and you will measure success by [Target Metrics].

I understand your target audience is [Audience Age Range] [Audience Gender] located in [Locations]. They are interested in [Interests/Behaviors], hoping to achieve [Goals/Needs], but struggling with [Pain Points].
For ad placement, we will use [Platform] and target [Ad Placements] like [Placement Types]. The preferred ad [Formats] are [Creative Formats].

Based on your brand guidelines, the ads should have a [Tone] tone of voice and align with your [Brand Voice] standards. Your brand [Colors] and other [Guidelines] will inform the visuals.

For context, the campaign [Timeline/Duration] is [Duration], with a budget of [Budget]. We should highlight [Promotions/Discounts] and compare against [Competitor Examples].

With this information, I can now generate tailored ad concepts optimized for [Client/Company Name]'s campaign goals, audience, brand, and platform. Please confirm if I have understood the variables correctly. I'm ready to start creating ads whenever you are!

=> Act as an expert marketer and [Platform] advertiser. Review the copy on this website - [Client/Company Website]

The goal is to create 5 possible [Platform] ads promoting [Offer/Product]. The tone should be [Tone] and align with [Brand Voice].

The target audience is [Audience Age Range] [Audience Gender] interested in [Interests/Behaviors]. Their primary goal and reason for engaging is to [Campaign Objective].

Provide 5 ad options including copy, headlines, CTAs and details of possible creative.
Format: Include emojis in the copy and headlines, as appropriate. Use [Brand Colors] and [Brand Guidelines] in the visual designs.

The [Timeline/Duration] is [Duration] with a budget of [Budget]. Highlight any [Promotions/Discounts] being offered.

Success will be measured by [Target Metrics] such as [Metric Types]. Compare approaches to [Competitor Examples].

With this additional context tailored to [Client/Company Name]'s campaign details, generate 10 optimized ads for [Platform] to promote [Offer/Product] to the target [Audience Gender] interested in [Interests/Behaviors] in order to [Campaign Objective].

=> Execute [GO TO MARKET ADS CONTENT BRIEF] based on the context above then [Result]

Section 7
AFFILIATE PROGRAM BRIEF

> Act as a marketing strategist. Create an optimized process to develop a comprehensive affiliate program for the [product]. Start by creating key mission statements and messaging to ensure consistency in all promotional materials. Identify potential target audiences who are most likely to engage in the program. Set commission rates, select promotional offers, and establish program terms that provide incentives, but also protect the interests of the [product] brand. Write extensive product descriptions that communicate the unique benefits of the product to inspire customer interest and demand. Develop and provide marketing materials such as email swipe files, banners, and social media posts templates that affiliates can easily implement. Ensure that everything produced is aligned with your target personas and conveys a clear message. Finally, track progress by setting up a plan to monitor metrics that include click-through rates, conversion rate, earnings per click and other relevant performance metrics that can help calibrate, evaluate and improve the performance of the affiliate program.

AutoScript:

command1=>"Create key mission statements and messaging for the [product] affiliate program."
command2=>"Identify potential target audiences."
command3=>"Set commission rates, select promotional offers, and establish program terms."
command4=>"Write extensive product descriptions."
command5=>"Develop marketing materials such as email swipe files, banners, and social media posts templates."
command6=>"Ensure alignment with target personas and convey a clear message."
command7=>"Track progress by monitoring metrics such as click-through rates, conversion rate, earnings per click."

[result]=fully built out copy and content with strategy to launch an affiliate program
Execute command1
Execute command2
Execute command3
Execute command4
Execute command5
Execute command6
Execute command7

⇒ Execute [result] starting with Section 1 then working all the way down, each section building off the last
 


            """),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )


    agent = create_tool_calling_agent(llm, tools, prompt)

    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    result = agent_executor.invoke({"input": query})
    #conversational_memory.save_context({"Me": query}, {"You": result['output']})
    
    return result['output'][0]['text']



if __name__ == "__main__":
      print("## Welcome to the RAG chatbot")
      print('-------------------------------')
      
      while True:
        query = input("You: ")
        if query.lower() == 'exit':
            break
       
        
        result = sub_agent( query)
        print("Bot:", result)