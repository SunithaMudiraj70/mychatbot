from model_loader import ModelLoader
from chat_memory import ChatMemory

# --- Knowledge Base ---
CAPITALS = {
    "france": "Paris",
    "italy": "Rome",
    "india": "New Delhi",
    "china": "Beijing",
    "usa": "Washington, D.C.",
    "united states": "Washington, D.C.",
    "japan": "Tokyo",
    "germany": "Berlin",
    "russia": "Moscow",
    "south korea": "Seoul",
    "north korea": "Pyongyang",
    "singapore": "Singapore"
}

LEADERS = {
    ("prime minister", "india"): "Narendra Modi",
    ("president", "india"): "Droupadi Murmu",
    ("president", "usa"): "Joe Biden",
    ("president", "united states"): "Joe Biden",
    ("president", "china"): "Xi Jinping",
    ("president", "south korea"): "Yoon Suk-yeol",
    ("leader", "north korea"): "Kim Jong-un",
    ("president", "north korea"): "Kim Jong-un",
    ("prime minister", "singapore"): "Lawrence Wong"
}

def get_kb_answer(user_input: str):
    text = user_input.lower().strip()

    # Capital questions
    for country, capital in CAPITALS.items():
        if f"capital of {country}" in text or text == country:
            return f"The capital of {country.title()} is {capital}."

    # Leader questions
    for (title, country), name in LEADERS.items():
        if title in text and country in text:
            return f"The {title.title()} of {country.title()} is {name}."

    return None

def build_prompt(memory: ChatMemory, user_input: str):
    return f"""
You are a helpful assistant. Always answer in one complete, factual sentence.

Examples:
User: Who is Albert Einstein?
Bot: Albert Einstein was a physicist known for the theory of relativity.

User: What is photosynthesis?
Bot: Photosynthesis is the process by which plants make food using sunlight, water, and carbon dioxide.

Now continue the conversation:

Conversation so far:
{memory.get_context()}

User: {user_input}
Bot:"""

def run_chat():
    loader = ModelLoader("google/flan-t5-base")
    loader.load_model()
    memory = ChatMemory(window_size=5)

    print("🤖 Chatbot ready! Type '/exit' to quit.\n")

    while True:
        user_input = input("User: ")
        if user_input.strip().lower() == "/exit":
            print("Bot: Exiting chatbot. Goodbye!")
            break

        memory.add_message("User", user_input)

        # 1) Try knowledge base first
        kb_answer = get_kb_answer(user_input)
        if kb_answer:
            print(f"Bot: {kb_answer}")
            memory.add_message("Bot", kb_answer)
            continue

        # 2) Otherwise, use the model
        prompt = build_prompt(memory, user_input)
        response = loader.generate_response(prompt, max_new_tokens=96)

        if "Bot:" in response:
            response = response.split("Bot:")[-1].strip()

        print(f"Bot: {response}")
        memory.add_message("Bot", response)
