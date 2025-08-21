# mychatbot
The chatbot is developed using hugging face with small model for responding input queries and maintaing memory of previous history
# 🧠 Terminal Chatbot using Hugging Face

This is a lightweight command-line chatbot powered by a Hugging Face model. It maintains short-term memory for multi-turn conversations and runs entirely on your local machine—no virtual environment required.

## 📂 Files Included

- `model_loader.py` – Loads the Hugging Face model and tokenizer  
- `chat_memory.py` – Manages conversation history using a sliding window  
- `interface.py` – Handles user input/output in the terminal  
- `main.py` – Entry point that ties everything together

## 🚀 How to Run

1. **Install required packages**  
   Make sure you have Python 3 installed. Then run:
   ```bash
   pip install transformers torch
2.How to run:
git clone https://github.com/SunithaMudiraj70/mychatbot.git
cd mychatbot
python main.py

python main.py
3.sample interaction:
User: What's the capital of Japan?
Bot: The capital of Japan is Tokyo.
User: What about france?
Bot: The capital of france is paris.

User: /exit
Exiting chatbot. Goodbye!
#I have also included my chatbot to answer other questions also.Here down i have prepared a colab version that also works well
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from collections import deque

# -------- Model Loader --------
class ModelLoader:
    def __init__(self, model_name="google/flan-t5-base"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None

    def load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        print(f"✅ Model {self.model_name} loaded successfully!")

    def generate_response(self, prompt, max_new_tokens=128):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


# -------- Sliding Memory --------
class ChatMemory:
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.history = deque(maxlen=window_size * 2)

    def add_message(self, role, message):
        self.history.append(f"{role}: {message}")

    def get_context(self):
        return "\n".join(self.history)


# -------- Chat Interface --------
def run_chat():
    loader = ModelLoader("google/flan-t5-base")
    loader.load_model()
    memory = ChatMemory(window_size=5)

    print("🤖 Chatbot ready! Type '/exit' to quit.\n")
    memory.add_message("Bot", "You are a helpful assistant.")

    while True:
        user_input = input("User: ")
        if user_input.strip().lower() == "/exit":
            print("Bot: Exiting chatbot. Goodbye!")
            break

        memory.add_message("User", user_input)

        # Few-shot examples + memory context
        prompt = f"""
The following is a factual conversation between a User and an AI assistant.
The assistant always answers with the correct factual answer.

Examples:
User: What is the capital of France?
Bot: The capital of France is Paris.

User: What is the capital of Italy?
Bot: The capital of Italy is Rome.

Now continue the conversation:

{memory.get_context()}

Bot:"""

        bot_reply = loader.generate_response(prompt, max_new_tokens=128)

        # Clean reply
        if "Bot:" in bot_reply:
            bot_reply = bot_reply.split("Bot:")[-1].strip()

        print(f"Bot: {bot_reply}")
        memory.add_message("Bot", bot_reply)


# -------- Run --------
if __name__ == "__main__":
    run_chat()
