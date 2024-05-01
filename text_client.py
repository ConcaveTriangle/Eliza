import requests

def send_to_chatbot(user_input, password):
    url = 'https://upright-jolly-troll.ngrok-free.app/chat'
    data = {'input': user_input, 'password': password}
    response = requests.post(url, json=data)
    return response.json()['response']

# Example usage
for x in range(100): 
    chatbot_response = send_to_chatbot(input("Jerry: "), "ConcaveTriangle")
    print(chatbot_response)