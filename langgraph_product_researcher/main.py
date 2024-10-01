from termcolor import colored
from typing_extensions import TypedDict
from langgraph_product_researcher.chains import *
from langgraph_product_researcher.tools import *  # reddit scraper and comment cleaner
from langgraph_product_researcher.graph import app
from langgraph_product_researcher.node import *
# Define the GraphState class





def market_researcher(product_info):
    


    # Use session state to maintain input values across reruns

    output = app.invoke(product_info)  # Replace with actual invoke logic
    generated_branding = output['branding']





# Run the app
if __name__ == "__main__":
    product_info = "AI Researcher Agent"
    market_researcher(product_info)
        
     