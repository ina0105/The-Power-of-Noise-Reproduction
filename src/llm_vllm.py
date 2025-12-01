import os
from typing import List, Tuple, Optional
from vllm import LLM as VLLMEngine, SamplingParams
from transformers import AutoTokenizer

# Try to import huggingface_hub for login
try:
    from huggingface_hub import login as hf_login
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False
    hf_login = None

class LLM:
    """
    A class for loading and generating text using a Language Model (LM) with vLLM backend.
    
    Attributes:
        model_id (str): Identifier for the model to load.
        quantization (Optional[str]): Quantization method ('awq', 'gptq', 'fp8', None for no quantization).
        stop_list (Optional[List[str]]): List of tokens where generation should stop.
        model_max_length (int): Maximum length of the model inputs.
        tensor_parallel_size (int): Number of GPUs for tensor parallelism.
    """
    def __init__(
        self, 
        model_id: str, 
        quantization: Optional[str] = None,
        stop_list: Optional[List[str]] = None, 
        model_max_length: int = 4096,
        tensor_parallel_size: int = 1,
        dtype: str = "bfloat16",
        max_model_len: Optional[int] = None,
    ):
        self.model_id = model_id
        self.model_max_length = model_max_length
        self.max_model_len = max_model_len or model_max_length

        self.stop_list = stop_list
        if stop_list is None:
            # Default stop tokens for Llama models
            # Llama 3 uses <|end_of_text|> and <|eot_id|>
            # Also stop on new questions or repeated prompts
            if 'llama-3' in model_id.lower() or 'meta-llama-3' in model_id.lower():
                self.stop_list = ['<|end_of_text|>', '<|eot_id|>', '<|eot|>', '\n\nQuestion:', '\nQuestion:', 'Question:', '\n\nDocuments:', '\nDocuments:', 'Documents:']
            else:
                # For Llama 2 and other models
                self.stop_list = ['<|end_of_text|>', '<|eot_id|>', '<|eot|>', '\n\nQuestion:', '\nQuestion:', 'Question:', '\n\n']
        
        # Get HF token from environment if available
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        
        # Try to login to HuggingFace if token is available
        if hf_token and HF_HUB_AVAILABLE:
            try:
                hf_login(token=hf_token, add_to_git_credential=False)
                print("Successfully logged in to HuggingFace Hub")
            except Exception as e:
                print(f"Warning: HuggingFace login failed: {e}")
        elif hf_token:
            print(f"Using HF token from environment (token length: {len(hf_token)} chars)")
        else:
            print("WARNING: No HF token found in environment variables!")
            print("Available env vars with 'HF' or 'HUGGING':", [k for k in os.environ.keys() if 'HF' in k.upper() or 'HUGGING' in k.upper()])
        
        # Load tokenizer separately for prompt processing
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            padding_side="left",
            truncation_side="left",
            model_max_length=model_max_length,
            trust_remote_code=True,
            token=hf_token
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Configure vLLM quantization
        quantization_config = None
        if quantization and quantization.strip():
            quant_lower = quantization.lower().strip()
            if quant_lower == 'awq':
                quantization_config = "awq"
            elif quant_lower == 'gptq':
                quantization_config = "gptq"
            elif quant_lower == 'fp8':
                quantization_config = "fp8"
            else:
                print(f"Warning: Unknown quantization method '{quantization}', using None")
        
        # Initialize vLLM engine
        print(f"Initializing vLLM engine with model: {model_id}")
        print(f"Quantization: {quantization_config or 'None'}")
        print(f"Tensor parallel size: {tensor_parallel_size}")
        print(f"Max model length: {self.max_model_len}")
        
        # vLLM uses environment variables or HuggingFace cache for authentication
        # Since we've already logged in via huggingface_hub.login(), vLLM will use that
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        
        if hf_token:
            print(f"HF token available in environment (token length: {len(hf_token)} chars)")
            print("vLLM will use environment variable or HuggingFace cache for authentication")
        else:
            print("WARNING: No HF token found in environment!")
        
        # Note: vLLM doesn't accept 'token' parameter - it uses environment variables
        # or the HuggingFace cache (which we've set via huggingface_hub.login())
        self.engine = VLLMEngine(
            model=model_id,
            quantization=quantization_config,
            tensor_parallel_size=tensor_parallel_size,
            dtype=dtype,
            max_model_len=self.max_model_len,
            trust_remote_code=True,
        )
        print("vLLM engine initialized successfully")
        
    def generate(self, prompts: List[str], max_new_tokens: int = 15, temperature: float = 0.0, 
                 top_p: float = 1.0, repetition_penalty: float = 1.1) -> List[str]:
        """
        Generates text based on the given prompts.
        
        Args:
            prompts (List[str]): Input text prompts for generation.
            max_new_tokens (int): Maximum number of tokens to generate.
            temperature (float): Sampling temperature (0.0 for greedy).
            top_p (float): Top-p sampling parameter.
            repetition_penalty (float): Repetition penalty.
        
        Returns:
            List[str]: The generated text responses.
        """
        # Check if we should use chat template for Llama 3 models
        use_chat_template = False
        if self.tokenizer.chat_template is not None and ('llama-3' in self.model_id.lower() or 'meta-llama-3' in self.model_id.lower()):
            use_chat_template = True
        
        # Apply chat template if needed
        formatted_prompts = []
        if use_chat_template:
            for prompt in prompts:
                # Parse the prompt to extract system instruction and user message
                # Format: "{task_instruction}\nDocuments:\n{documents_str}\nQuestion: {query}\nAnswer:"
                lines = prompt.split('\n')
                
                # Find where "Documents:" starts
                docs_idx = -1
                question_idx = -1
                for i, line in enumerate(lines):
                    if line.strip().startswith('Documents:'):
                        docs_idx = i
                    if line.strip().startswith('Question:'):
                        question_idx = i
                
                if docs_idx >= 0 and question_idx >= 0:
                    # Extract system instruction (everything before "Documents:")
                    system_message = '\n'.join(lines[:docs_idx]).strip()
                    # Extract user message (Documents + Question, excluding "Answer:" line)
                    # lines[docs_idx:question_idx+1] includes Documents through Question line
                    # The "Answer:" line is at question_idx+1, which is excluded by the slice
                    user_message = '\n'.join(lines[docs_idx:question_idx+1]).strip()
                    
                    # Apply chat template
                    messages = [
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": user_message}
                    ]
                    formatted_prompt = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                    formatted_prompts.append(formatted_prompt)
                else:
                    # Fallback: use prompt as-is
                    formatted_prompts.append(prompt)
        else:
            formatted_prompts = prompts
        
        # Configure sampling parameters
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_new_tokens,
            stop=self.stop_list,
            repetition_penalty=repetition_penalty,
        )
        
        # Generate using vLLM
        outputs = self.engine.generate(formatted_prompts, sampling_params)
        
        # Extract generated text
        generated_texts = []
        for output in outputs:
            # Get the generated text from the first (and only) completion
            generated_text = output.outputs[0].text
            generated_texts.append(generated_text)
        
        return generated_texts

