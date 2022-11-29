import torch
from transformers import AutoTokenizer, GPTJForCausalLM


# hugging face gpt-j-6b model
class GPTJ:
    def __init__(self, device):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
        if self.device == "cpu":
            self.model = GPTJForCausalLM.from_pretrained("EleutherAI/gpt-j-6B")
        elif self.device == "cuda":
            self.model = GPTJForCausalLM.from_pretrained(
                "EleutherAI/gpt-j-6B", revision="float16", torch_dtype=torch.float16
            )
        else:
            print(f"Invalid device {self.device}!")
        self.model.to(self.device)
        self.model.eval()

    def generate(self, text):
        input_ids = self.tokenizer(text, return_tensors="pt").input_ids.to(self.device)
        output = self.model.generate(
            input_ids, max_new_tokens=20, do_sample=True, temperature=0.95, num_return_sequences=1
        )
        return self.tokenizer.decode(output[0])
