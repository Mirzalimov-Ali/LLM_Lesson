from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.save_pretrained("./gpt2-tokenizer")

text = "My name is Ali. I live in Korea"

print("Text:", text)
print("Tokens:", tokenizer.tokenize(text))
print("Token IDs:", tokenizer.encode(text))