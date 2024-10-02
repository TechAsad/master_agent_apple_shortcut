from termcolor import colored
from typing_extensions import TypedDict
from reddit_scraper.chains import *
from reddit_scraper.tools import *  # reddit scraper and comment cleaner
from reddit_scraper.graph import app
from reddit_scraper.node import *
# Define the GraphState class






def reddit_agent(query):
    


    # Use session state to maintain input values across reruns

    output = app.invoke({"query":query})  # Replace with actual invoke logic
    comments = output['comments']





# Run the apps
if __name__ == "__main__":
    query = "AI Researcher Agent"
  
    reddit_agent(query)
        
            
