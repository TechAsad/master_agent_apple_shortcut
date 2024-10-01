from termcolor import colored
from typing_extensions import TypedDict
from langgraph_branding_agent.chains import *
from langgraph_branding_agent.tools import *  # reddit scraper and comment cleaner
from langgraph_branding_agent.graph import app
from langgraph_branding_agent.node import *
# Define the GraphState class





def branding_agent(product_info):
    


    # Use session state to maintain input values across reruns

    output = app.invoke(product_info)  # Replace with actual invoke logic
    generated_branding = output['branding']





# Run the app
if __name__ == "__main__":
    product_info = "AI Researcher Agent"
    branding_agent(product_info)
        
            
