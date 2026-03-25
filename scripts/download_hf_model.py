"""
Download a HuggingFace language model for word prediction.
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import os

MODEL_NAME = "gpt2"


SAVE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "word_predictor", MODEL_NAME.split("/")[-1])

def download_model():
    print(f"Downloading model: {MODEL_NAME}")
    print(f"Saving to: {SAVE_DIR}")
    print("This might take a few minutes on first run...\n")

    os.makedirs(SAVE_DIR, exist_ok=True)

    print("Downloading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.save_pretrained(SAVE_DIR)
    print("Tokenizer saved!\n")

    print("Downloading model weights...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.save_pretrained(SAVE_DIR)
    print("Model saved!\n")

    print("Running quick test...")
    input_text = "Hello my name"
    inputs = tokenizer(input_text, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_new_tokens=5,
        num_return_sequences=1,
        do_sample=True,
        temperature=0.7
    )
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Input: '{input_text}'")
    print(f"Model completed: '{result}'")
    print("\nModel downloaded and verified successfully!")
    print(f"Location: {SAVE_DIR}")

if __name__ == "__main__":
    download_model()