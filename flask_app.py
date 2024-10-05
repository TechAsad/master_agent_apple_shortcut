from flask import Flask, request, jsonify
from master import master_agent

from langchain.chains.conversation.memory import ConversationBufferWindowMemory
import os
memory = ConversationBufferWindowMemory(
        memory_key='chat_history',
        k=10,
        return_messages=True
)


app = Flask(__name__)

@app.route('/assistant', methods=['POST'])
def assistant():
    # Get the JSON data from the Apple Shortcut request
    data = request.get_json()

    # Debug: Print the incoming data
    print(f"Received data: {data}")
    
    # Check if 'message' is present
    message = data.get('input', '') if data else ''
    
    # Debug: Print the received message
    print(f"Received message: {message}")
    
    # Process the message with your master_agent function
    response = master_agent(message)
    
    #memory.save_context({"Human": message}, {"AI": response})
    
    # Return the processed response as JSON
    return jsonify(response)

if __name__ == '__main__':
    # Get the PORT from the environment, with a fallback to 5000
    port = int(os.environ.get('PORT', 5000))
    # Run the app on 0.0.0.0 to allow external connections
    app.run(host='0.0.0.0', port=port)