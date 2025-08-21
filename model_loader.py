# model_loader.py
from transformers import pipeline
import torch

class ModelLoader:
    def __init__(self, model_name="google/flan-t5-base"):
        self.model_name = model_name
        self.generator = None
        self.device = 0 if torch.cuda.is_available() else -1  # GPU if available

    def load_model(self):
        # Hugging Face pipeline for text generation
        self.generator = pipeline(
            "text2text-generation",
            model=self.model_name,
            device=self.device
        )
        device_name = "GPU" if self.device >= 0 else "CPU"
        print(f"✅ Model {self.model_name} loaded on {device_name}!")

    def generate_response(self, prompt, max_new_tokens=128):
        result = self.generator(
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=False
        )
        return result[0]['generated_text'].strip()
