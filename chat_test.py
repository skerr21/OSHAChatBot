import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BertTokenizer

chatbot_params = {
    "name": "OSHA Safety Advisor",
    "description": "OSHA Buddy is here to talk to you about anything related to OSHA.",
    "intents": [
        "General OSHA information",
    ],
        "model_path": r"F:\ConversationalOsha\model"  # the directory should contain pytorch_model.bin and config.json files
}



class OSHAChatBot:
    def __init__(self, params):
        self.params = params
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(params["model_path"])
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(params["model_path"]).to(self.device)
    
    def respond_to(self, input_text):
        # Add the chatbot's description to the input text to provide context
        context = self.params["description"]
        input_text = context + " " + input_text
        
        # Encode the input text
        input_ids = self.tokenizer.encode(input_text + self.tokenizer.eos_token, return_tensors='pt').to(self.device)

        # Generate a response
        chatbot_response = self.model.generate(
            input_ids, 
            max_length=150, 
            temperature=0.3,
            top_k=6,
            top_p=0.7,
            num_beams=7,
            no_repeat_ngram_size=3,
            early_stopping=True
        )

        # Decode the response
        chatbot_response = self.tokenizer.decode(chatbot_response[:, input_ids.shape[-1]:][0], skip_special_tokens=True)

        return chatbot_response

# Initialize the chatbot with the name of your trained model
chatbot = OSHAChatBot(chatbot_params)

# Print the bot's role
print("Hello, I am your OSHA Buddy. How can I assist you today?")

while True:
    # Get user input
    user_input = input("User: ")

    # End conversation if the user says 'quit'
    if user_input.lower() == 'quit':
        break

    # Generate a response
    chatbot_response = chatbot.respond_to(user_input)

    # Print the response
    print(f"{chatbot.params['name']}: {chatbot_response}")
