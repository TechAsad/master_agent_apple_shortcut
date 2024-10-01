from flask import Flask, request, jsonify
from master import master_agent
from langchain.chains.conversation.memory import ConversationBufferWindowMemory

memory = ConversationBufferWindowMemory(
        memory_key='chat_history',
        k=10,
        return_messages=True
)


app = Flask(__name__)




@app.route('/webhook-test/assistant', methods=['POST'])
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
    # Run the app on port 5000
    app.run(host='0.0.0.0', port=5001)
