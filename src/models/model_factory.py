"""
🌙 Moon Dev's Model Factory
Simplified model factory for Swarm Agent
"""

import os
from typing import Optional, Any


class ModelResponse:
    """Simple model response wrapper"""
    def __init__(self, content: str):
        self.content = content
    
    def __str__(self):
        return self.content


class MockModel:
    """Mock model for demonstration purposes"""
    def __init__(self, model_name: str):
        self.model_name = model_name
    
    def generate_response(self, system_prompt: str, user_content: str, 
                         temperature: float = 0.7, max_tokens: int = 2048) -> ModelResponse:
        """Generate a mock response"""
        return ModelResponse(
            f"[{self.model_name}] Mock response to: {user_content[:100]}...\n\n"
            f"This is a demonstration response. To enable real AI models, configure API keys "
            f"in the sidebar and ensure the model providers are properly set up."
        )


class ModelFactory:
    """Factory for creating AI model instances"""
    
    def __init__(self):
        self.models = {}
    
    def get_model(self, model_type: str, model_name: str, **kwargs) -> Optional[Any]:
        """
        Get a model instance
        
        Args:
            model_type: Type of model (claude, openai, deepseek, etc.)
            model_name: Specific model name
            **kwargs: Additional model configuration (e.g., api_key)
        
        Returns:
            Model instance or None
        """
        try:
            # For now, return mock models
            # In production, this would instantiate real AI model clients
            
            if model_type == "claude":
                return self._get_claude_model(model_name, **kwargs)
            elif model_type == "openai":
                return self._get_openai_model(model_name, **kwargs)
            elif model_type == "deepseek":
                return self._get_deepseek_model(model_name, **kwargs)
            elif model_type == "xai":
                return self._get_xai_model(model_name, **kwargs)
            elif model_type == "openrouter":
                return self._get_openrouter_model(model_name, **kwargs)
            elif model_type == "ollama":
                return self._get_ollama_model(model_name, **kwargs)
            else:
                print(f"⚠️ Unknown model type: {model_type}")
                return MockModel(f"{model_type}/{model_name}")
                
        except Exception as e:
            print(f"❌ Error creating {model_type} model: {e}")
            return None
    
    def _get_claude_model(self, model_name: str, **kwargs):
        """Get Claude model instance"""
        api_key = kwargs.get('api_key') or os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            print(f"⚠️ No API key for Claude - using mock model")
            return MockModel(f"claude/{model_name}")
        
        try:
            from anthropic import Anthropic
            
            class ClaudeModel:
                def __init__(self, client, model_name):
                    self.client = client
                    self.model_name = model_name
                
                def generate_response(self, system_prompt: str, user_content: str,
                                    temperature: float = 0.7, max_tokens: int = 2048):
                    response = self.client.messages.create(
                        model=self.model_name,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        system=system_prompt,
                        messages=[{"role": "user", "content": user_content}]
                    )
                    return ModelResponse(response.content[0].text)
            
            client = Anthropic(api_key=api_key)
            return ClaudeModel(client, model_name)
        except ImportError:
            print("⚠️ anthropic package not installed - using mock model")
            return MockModel(f"claude/{model_name}")
    
    def _get_openai_model(self, model_name: str, **kwargs):
        """Get OpenAI model instance"""
        api_key = kwargs.get('api_key') or os.getenv('OPENAI_API_KEY')
        if not api_key:
            print(f"⚠️ No API key for OpenAI - using mock model")
            return MockModel(f"openai/{model_name}")
        
        try:
            from openai import OpenAI
            
            class OpenAIModel:
                def __init__(self, client, model_name):
                    self.client = client
                    self.model_name = model_name
                
                def generate_response(self, system_prompt: str, user_content: str,
                                    temperature: float = 0.7, max_tokens: int = 2048):
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_content}
                        ],
                        temperature=temperature,
                        max_tokens=max_tokens
                    )
                    return ModelResponse(response.choices[0].message.content)
            
            client = OpenAI(api_key=api_key)
            return OpenAIModel(client, model_name)
        except ImportError:
            print("⚠️ openai package not installed - using mock model")
            return MockModel(f"openai/{model_name}")
    
    def _get_deepseek_model(self, model_name: str, **kwargs):
        """Get DeepSeek model instance"""
        api_key = kwargs.get('api_key') or os.getenv('DEEPSEEK_API_KEY')
        if not api_key:
            print(f"⚠️ No API key for DeepSeek - using mock model")
            return MockModel(f"deepseek/{model_name}")
        
        try:
            from openai import OpenAI
            
            class DeepSeekModel:
                def __init__(self, api_key, model_name):
                    self.client = OpenAI(
                        api_key=api_key,
                        base_url="https://api.deepseek.com"
                    )
                    self.model_name = model_name
                
                def generate_response(self, system_prompt: str, user_content: str,
                                    temperature: float = 0.7, max_tokens: int = 2048):
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_content}
                        ],
                        temperature=temperature,
                        max_tokens=max_tokens
                    )
                    return ModelResponse(response.choices[0].message.content)
            
            return DeepSeekModel(api_key, model_name)
        except ImportError:
            print("⚠️ openai package not installed - using mock model")
            return MockModel(f"deepseek/{model_name}")
    
    def _get_xai_model(self, model_name: str, **kwargs):
        """Get XAI (Grok) model instance"""
        api_key = kwargs.get('api_key') or os.getenv('XAI_API_KEY')
        if not api_key:
            print(f"⚠️ No API key for XAI - using mock model")
            return MockModel(f"xai/{model_name}")
        
        try:
            from openai import OpenAI
            
            class XAIModel:
                def __init__(self, api_key, model_name):
                    self.client = OpenAI(
                        api_key=api_key,
                        base_url="https://api.x.ai/v1"
                    )
                    self.model_name = model_name
                
                def generate_response(self, system_prompt: str, user_content: str,
                                    temperature: float = 0.7, max_tokens: int = 2048):
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_content}
                        ],
                        temperature=temperature,
                        max_tokens=max_tokens
                    )
                    return ModelResponse(response.choices[0].message.content)
            
            return XAIModel(api_key, model_name)
        except ImportError:
            print("⚠️ openai package not installed - using mock model")
            return MockModel(f"xai/{model_name}")
    
    def _get_openrouter_model(self, model_name: str, **kwargs):
        """Get OpenRouter model instance"""
        api_key = kwargs.get('api_key') or os.getenv('OPENROUTER_API_KEY')
        if not api_key:
            print(f"⚠️ No API key for OpenRouter - using mock model")
            return MockModel(f"openrouter/{model_name}")
        
        try:
            from openai import OpenAI
            
            class OpenRouterModel:
                def __init__(self, api_key, model_name):
                    self.client = OpenAI(
                        api_key=api_key,
                        base_url="https://openrouter.ai/api/v1"
                    )
                    self.model_name = model_name
                
                def generate_response(self, system_prompt: str, user_content: str,
                                    temperature: float = 0.7, max_tokens: int = 2048):
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_content}
                        ],
                        temperature=temperature,
                        max_tokens=max_tokens,
                        extra_headers={
                            "HTTP-Referer": "https://github.com/moondev",
                            "X-Title": "Moon Dev Swarm Agent"
                        }
                    )
                    return ModelResponse(response.choices[0].message.content)
            
            return OpenRouterModel(api_key, model_name)
        except ImportError:
            print("⚠️ openai package not installed - using mock model")
            return MockModel(f"openrouter/{model_name}")
    
    def _get_ollama_model(self, model_name: str, **kwargs):
        """Get Ollama model instance"""
        try:
            import ollama
            
            class OllamaModel:
                def __init__(self, model_name):
                    self.model_name = model_name
                
                def generate_response(self, system_prompt: str, user_content: str,
                                    temperature: float = 0.7, max_tokens: int = 2048):
                    response = ollama.chat(
                        model=self.model_name,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_content}
                        ],
                        options={
                            "temperature": temperature,
                            "num_predict": max_tokens
                        }
                    )
                    return ModelResponse(response['message']['content'])
            
            return OllamaModel(model_name)
        except ImportError:
            print("⚠️ ollama package not installed - using mock model")
            return MockModel(f"ollama/{model_name}")


# Global factory instance
model_factory = ModelFactory()
