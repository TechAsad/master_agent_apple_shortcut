import os
import langchain
from textwrap import dedent
import pandas as pd
from langchain.tools import BaseTool
from langchain_openai import ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains.question_answering import load_qa_chain
import pinecone
from langchain_openai import OpenAIEmbeddings

from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.prompts import (
    ChatPromptTemplate
)
import openai


from langchain_core.messages import HumanMessage, SystemMessage

from dotenv import load_dotenv
load_dotenv()


# Set API keys
os.environ['OPENAI_API_KEY'] =os.getenv("OPENAI_API_KEY")

os.environ['PINECONE_API_KEY'] = os.getenv("PINECONE_API_KEY")

#pinecone.init(api_key=os.environ['PINECONE_API_KEY'], environment="us-east-1")

# Initialize embeddings and vector store
index_name = "courses"
embedding = OpenAIEmbeddings()
#index = pinecone.Index(index_name=index_name, api_key=os.environ['PINECONE_API_KEY'])
vector_store = PineconeVectorStore(index_name=index_name, embedding=embedding)


memory = ConversationBufferWindowMemory(memory_key="chat_history", input_key="question", human_prefix= "User", ai_prefix= "Assistant")
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)




class RAGbot(BaseTool):
      
    name = "Pinecone Vectorstore"
    description = "use this tool when you need to retrieve contents of a specific course with input query and namespace. namespace of courses are: 'hormozicourse', 'brandingcourse', 'aisaleslettercourse', 'aiavatarcourse',  'postioningcourse' "

    
    def _run(prompt): #need to pass user query and memory variable.
       
        #try:  
               

      
            prompt_template =dedent(r"""
            You are a helpful assistant who has knowledge abouT  "product branding course"
        
            \n
            User has asked question about this course and must answer humbly and respectfully
            to help the user. Your response should be focused on the course context given to you. Please also answer  relevant to the current conversation between you (AI) and the user.
            if asked, you must generate effective offers about the provided product/service.
            Do not answer from your training knowledge.
            \n
            this is the course context :
              ---------
              {context}
              
              ---------
               

              chat history: 
              ---------
              {chat_history}
              ---------

              Question: 
              {question}

              Helpful Answer: 
              """)
              
              

            PROMPT = PromptTemplate(
                    template=prompt_template, input_variables=[ "context", "question", "chat_history"]
                )

                
            chain = load_qa_chain(llm=llm, verbose= True, prompt = PROMPT,memory=memory, chain_type="stuff")
                    
        
        
         
            
            docs =vector_store.similarity_search(prompt, namespace="brandingcourse", k=4)
            
            
            response = chain.run(input_documents=docs, question=prompt)
            
            
            memory.save_context({"question": prompt}, {"output": response})
            
            
                
            return response
            
        #except Exception as e:
            
         #   "Sorry, the question is irrelevant or the bot crashed"
    


if __name__ == "__main__":
      print("## Welcome to the RAG chatbot")
      print('-------------------------------')
      
      while True:
        query = input("You: ")
        if query.lower() == 'exit':
            break
        chatbot= RAGbot
        #namspace = "hormozicourse"
        result = chatbot.run(query)
        print("Bot:", result)