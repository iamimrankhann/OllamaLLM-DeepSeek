from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from typing import Optional

class HuggingFaceModel:
    def __init__(self, model_name: str, api_token: Optional[str] = None):
        """
        Initialize the model with Hugging Face credentials
        
        Args:
            model_name: Name of the model on Hugging Face Hub
            api_token: Your Hugging Face API token
        """
        self.model_name = model_name
        self.api_token = api_token
        
        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            token=api_token,
            trust_remote_code=True
        )
        
        # Load model with CPU-specific configurations
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            token=api_token,
            torch_dtype=torch.float32,  # Use float32 for CPU
            device_map="cpu",  # Force CPU usage
            trust_remote_code=True,
            low_cpu_mem_usage=True,  # Optimize for lower memory usage
            load_in_8bit=False  # Disable 8-bit quantization for CPU
        )
        
    def generate_text(
        self,
        prompt: str,
        max_length: int = 256,  # Reduced max length for faster generation
        temperature: float = 0.7,
        top_p: float = 0.95,
        num_return_sequences: int = 1
    ) -> list[str]:
        """
        Generate text based on the input prompt
        
        Args:
            prompt: Input text to generate from
            max_length: Maximum length of generated text
            temperature: Controls randomness (higher = more random)
            top_p: Controls diversity via nucleus sampling
            num_return_sequences: Number of different sequences to generate
            
        Returns:
            List of generated text sequences
        """
        try:
            # Encode the input prompt
            inputs = self.tokenizer(prompt, return_tensors="pt")
            
            # Generate sequences
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                num_return_sequences=num_return_sequences,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=True
            )
            
            # Decode and return the generated sequences
            return [
                self.tokenizer.decode(output, skip_special_tokens=True)
                for output in outputs
            ]
            
        except Exception as e:
            print(f"Error generating text: {str(e)}")
            return []

# Example usage
if __name__ == "__main__":
    # Replace with your actual API token
    API_TOKEN = "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"
    
    # Initialize with a smaller model
    model = HuggingFaceModel(
        model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",  # Much smaller model
        api_token=API_TOKEN
    )
    
    # Generate text
    prompt = "I want to generate even numbers from 1 to 1000 using python"
    generated_texts = model.generate_text(
        prompt=prompt,
        max_length=256,
        temperature=0.7
    )
    
    # Print results
    for i, text in enumerate(generated_texts, 1):
        print(f"\nGenerated text {i}:")
        print(text)