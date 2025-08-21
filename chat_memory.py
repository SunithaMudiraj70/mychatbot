from collections import deque

class ChatMemory:
    def __init__(self, window_size=5):
        self.history = deque(maxlen=window_size * 2)

    def add_message(self, role, message):
        self.history.append(f"{role}: {message}")

    def get_context(self):
        return "\n".join(self.history)
