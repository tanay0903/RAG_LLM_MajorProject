# MODEL_ID = "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ"
# MODEL_ID = "TheBloke/Mistral-7B-Instruct-v0.2-GGUF"
# MODEL_BASENAME = "mistral-7b-instruct-v0.2.Q8_0.gguf"
from transformers import AutoModelForCausalLM, AutoTokenizer

# Specify the model name
model_name = "meta-llama/Llama-2-7b-hf"

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load the model
model = AutoModelForCausalLM.from_pretrained(model_name)

# Example usage
input_text = "Once upon a time"
inputs = tokenizer(input_text, return_tensors="pt")
output = model.generate(**inputs)

# Decode the generated text
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)
