#!/usr/bin/env python3
"""
🌙 Moon Dev's Swarm Agent 🌙

Queries multiple AI models in parallel and returns:
- Clean individual responses from each model
- AI-generated consensus summary (Claude 4.5 synthesizes all responses)

Perfect for getting diverse AI perspectives on trading decisions,
validating strategies, or any decision that benefits from multiple viewpoints.

Usage:
    # Run directly (asks for prompt interactively)
    python src/agents/swarm_agent.py

    # Run Streamlit Web UI
    streamlit run src/agents/swarm_agent.py -- --web-ui

    # Import and use in other agents
    from src.agents.swarm_agent import SwarmAgent

    swarm = SwarmAgent()
    result = swarm.query("Should I buy Bitcoin now?")

    # Access consensus summary
    print(result["consensus_summary"])  # 3-sentence synthesis by Claude 4.5

    # Access individual responses
    for provider, data in result["responses"].items():
        if data["success"]:
            print(f"{provider}: {data['response']}")

    # Check model mapping (AI #1 = CLAUDE, AI #2 = OPENAI, etc.)
    print(result["model_mapping"])

Built with love by Moon Dev 🚀
"""

import os
import sys
import json
import time
import re
import pandas as pd
import streamlit as st
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from termcolor import colored, cprint
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Add project root to path
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

# Import Moon Dev's model factory
from src.models.model_factory import model_factory

# ============================================
# 🎯 SWARM CONFIGURATION - EDIT THIS SECTION
# ============================================

# Configure which models to use in the swarm (set to True to enable)
SWARM_MODELS = {
    # 🌙 Moon Dev's Active Swarm Models - 7 Model Configuration
    "deepseek": (True, "deepseek", "deepseek-chat"),  # DeepSeek Chat - Fast chat model (API)
    "xai": (True, "xai", "grok-4-fast-reasoning"),  # Grok-4 fast reasoning ($0.20-$0.50/1M tokens)
    "openrouter_qwen": (True, "openrouter", "qwen/qwen3-max"),  # Qwen 3 Max - Powerful reasoning ($1.00/$1.00 per 1M tokens)

    # 🔇 Disabled Models (uncomment to enable)
    "claude": (True, "claude", "claude-sonnet-4-5"),  # Claude 4.5 Sonnet - Latest & Greatest!
    "opus": (True, "claude", "claude-opus-4-5-20251101"),  # 🌙 Moon Dev - Claude Opus 4.5 - Most powerful!
    #"openai": (True, "openai", "gpt-5"),  # GPT-5 - Most advanced model!
    #"ollama_qwen": (True, "ollama", "qwen3:8b"),  # Qwen3 8B via Ollama - Fast local reasoning!
    #"ollama": (True, "ollama", "DeepSeek-R1:latest"),  # DeepSeek-R1 local model via Ollama
    #"openrouter_qwen": (True, "openrouter", "qwen/qwen3-max"),  # Qwen 3 Max - Powerful reasoning ($1.00/$1.00 per 1M tokens)

    # 🌙 OpenRouter Models - Access 200+ models through one API!
    # Uncomment any of these to add them to your swarm:
    #"openrouter_gemini": (True, "openrouter", "google/gemini-2.5-flash"),  # Gemini 2.5 Flash - Fast & cheap! ($0.10/$0.40 per 1M tokens)
    "openrouter_glm": (True, "openrouter", "z-ai/glm-4.6"),  # GLM 4.6 - Zhipu AI reasoning ($0.50/$0.50 per 1M tokens)
    #"openrouter_deepseek_r1": (True, "openrouter", "deepseek/deepseek-r1-0528"),  # DeepSeek R1 - Advanced reasoning ($0.55/$2.19 per 1M tokens)
    #"openrouter_claude_opus": (True, "openrouter", "anthropic/claude-opus-4.1"),  # Claude Opus 4.1 via OpenRouter
    "openrouter_gpt5_mini": (True, "openrouter", "openai/gpt-5-mini"),  # GPT-5 Mini via OpenRouter

    # 💡 See all 200+ models at: https://openrouter.ai/docs
    # 💡 Any model from openrouter_model.py can be used here!
}

# Default parameters for model queries
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 2048  # Increased for model compatibility (Gemini/Groq/Qwen need 2048+ minimum)

# Timeout for each model (seconds)
MODEL_TIMEOUT = 120  # 🌙 Moon Dev - Increased to 120s for more reliability

# Consensus Reviewer - Synthesizes all responses into a clean summary
CONSENSUS_REVIEWER_MODEL = ("deepseek", "deepseek-chat")  # Using DeepSeek Chat API (fast)
CONSENSUS_REVIEWER_PROMPT = """You are a consensus analyzer reviewing multiple AI responses.

Below are responses from {num_models} different AI models to the same question.

{responses}

Your task: Provide a clear, concise 3-sentence maximum consensus response that:
1. Synthesizes the common themes across all responses
2. Highlights any notable agreements or disagreements
3. Gives a balanced, actionable summary

Keep it under 3 sentences. Be direct and clear."""

# Save results to file
SAVE_RESULTS = True
RESULTS_DIR = Path(project_root) / "src" / "data" / "swarm_agent"

# Streamlit UI settings
STREAMLIT_TITLE = "🌙 Moon Dev's AI Swarm Agent"
STREAMLIT_DESCRIPTION = """
Query multiple AI models simultaneously and get a consensus summary. Perfect for trading decisions, research, and analysis.
"""

# ============================================
# END CONFIGURATION
# ============================================

class SwarmAgent:
    """Moon Dev's Swarm Agent for multi-model consensus"""

    def __init__(self, custom_models: Optional[Dict] = None, api_keys: Optional[Dict] = None):
        """
        Initialize the Swarm Agent

        Args:
            custom_models: Optional dict to override SWARM_MODELS configuration
            api_keys: Optional dict of API keys for different providers
        """
        self.models_config = custom_models or SWARM_MODELS
        self.active_models = {}
        self.results_dir = RESULTS_DIR
        self.api_keys = api_keys or {}

        # Create results directory if saving is enabled
        if SAVE_RESULTS:
            self.results_dir.mkdir(parents=True, exist_ok=True)

        # Initialize models
        self._initialize_models()

        if not self._is_streamlit():
            cprint("\n" + "="*60, "cyan")
            cprint("🌙 Moon Dev's Swarm Agent Initialized 🌙", "cyan", attrs=['bold'])
            cprint("="*60, "cyan")
            cprint(f"\n🤖 Active Models in Swarm: {len(self.active_models)}", "green")
            for name in self.active_models.keys():
                cprint(f"   ✅ {name}", "green")

    def _is_streamlit(self):
        """Check if running in Streamlit context"""
        try:
            from streamlit.runtime.scriptrunner import get_script_run_ctx
            return get_script_run_ctx() is not None
        except:
            return False

    def _initialize_models(self):
        """Initialize all enabled models"""
        for provider, (enabled, model_type, model_name) in self.models_config.items():
            if not enabled:
                continue

            try:
                # Get model from factory with API keys if provided
                model_kwargs = {}
                if model_type in self.api_keys:
                    model_kwargs['api_key'] = self.api_keys[model_type]
                
                model = model_factory.get_model(model_type, model_name, **model_kwargs)
                if model:
                    self.active_models[provider] = {
                        "model": model,
                        "type": model_type,
                        "name": model_name
                    }
                    if not self._is_streamlit():
                        cprint(f"✅ Initialized {provider}: {model_name}", "green")
                else:
                    if not self._is_streamlit():
                        cprint(f"⚠️ Could not initialize {provider}: {model_name}", "yellow")
            except Exception as e:
                if not self._is_streamlit():
                    cprint(f"❌ Error initializing {provider}: {e}", "red")

    def _query_single_model(self, provider: str, model_info: Dict, prompt: str,
                          system_prompt: Optional[str] = None) -> Tuple[str, Dict]:
        """
        Query a single model

        Returns:
            Tuple of (provider_name, response_dict)
        """
        start_time = time.time()

        try:
            # Default system prompt if none provided
            if system_prompt is None:
                system_prompt = "You are a helpful AI assistant providing thoughtful analysis."

            # Query the model
            response = model_info["model"].generate_response(
                system_prompt=system_prompt,
                user_content=prompt,
                temperature=DEFAULT_TEMPERATURE,
                max_tokens=DEFAULT_MAX_TOKENS
            )

            elapsed = time.time() - start_time

            return provider, {
                "provider": provider,
                "model": model_info["name"],
                "response": response,
                "success": True,
                "error": None,
                "response_time": round(elapsed, 2)
            }

        except Exception as e:
            elapsed = time.time() - start_time
            if not self._is_streamlit():
                cprint(f"❌ Error querying {provider}: {e}", "red")

            return provider, {
                "provider": provider,
                "model": model_info["name"],
                "response": None,
                "success": False,
                "error": str(e),
                "response_time": round(elapsed, 2)
            }

    def query(self, prompt: str, system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """
        Query all models in the swarm in parallel

        Args:
            prompt: The prompt to send to all models
            system_prompt: Optional system prompt (uses default if None)

        Returns:
            Dict containing individual responses and metadata
        """
        if not self._is_streamlit():
            cprint(f"\n🌊 Initiating Swarm Query with {len(self.active_models)} models...", "cyan", attrs=['bold'])
            cprint(f"📝 Prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}", "blue")

            # Show which models are being called in parallel
            cprint(f"\n🚀 Calling models in parallel:", "yellow", attrs=['bold'])
            for provider, model_info in self.active_models.items():
                cprint(f"   → {provider.upper()}: {model_info['name']}", "cyan")

        start_time = time.time()
        all_responses = {}

        # Use ThreadPoolExecutor for parallel queries
        with ThreadPoolExecutor(max_workers=len(self.active_models)) as executor:
            # Submit all queries
            futures = {
                executor.submit(
                    self._query_single_model,
                    provider,
                    model_info,
                    prompt,
                    system_prompt
                ): provider
                for provider, model_info in self.active_models.items()
            }

            # Track which models are still pending
            completed_count = 0
            total_models = len(futures)

            # Collect results as they complete (with timeout handling)
            try:
                for future in as_completed(futures, timeout=MODEL_TIMEOUT + 10):
                    provider = futures[future]
                    completed_count += 1

                    if not self._is_streamlit():
                        cprint(f"\n⏳ Waiting for responses... ({completed_count}/{total_models} completed)", "yellow")
                        cprint(f"🔄 Processing: {provider}...", "cyan")

                    try:
                        provider, response = future.result(timeout=5)  # 5 second timeout per result
                        all_responses[provider] = response

                        if not self._is_streamlit():
                            if response["success"]:
                                cprint(f"   ✅ {provider} responded ({response['response_time']}s)", "green")
                            else:
                                cprint(f"   ❌ {provider} failed: {response['error']}", "red")

                    except TimeoutError:
                        if not self._is_streamlit():
                            cprint(f"   ⏰ {provider} timed out (>{MODEL_TIMEOUT}s) - skipping", "yellow")
                        all_responses[provider] = {
                            "provider": provider,
                            "model": "timeout",
                            "response": None,
                            "success": False,
                            "error": f"Timeout after {MODEL_TIMEOUT}s",
                            "response_time": MODEL_TIMEOUT
                        }
                    except Exception as e:
                        if not self._is_streamlit():
                            cprint(f"   💥 {provider} error: {str(e)}", "red")
                        all_responses[provider] = {
                            "provider": provider,
                            "model": "error",
                            "response": None,
                            "success": False,
                            "error": str(e),
                            "response_time": 0
                        }

            except TimeoutError as timeout_error:
                # as_completed timed out waiting for all futures
                if not self._is_streamlit():
                    cprint(f"\n⏰ Overall timeout reached - some models didn't respond", "yellow")
                    cprint(f"⚠️ {str(timeout_error)}", "yellow")
                # Mark any remaining futures as timed out
                for future, provider in futures.items():
                    if provider not in all_responses:
                        if not self._is_streamlit():
                            cprint(f"   ⏰ {provider} never responded - marking as timeout", "red")
                        all_responses[provider] = {
                            "provider": provider,
                            "model": "timeout",
                            "response": None,
                            "success": False,
                            "error": f"Global timeout - no response received",
                            "response_time": MODEL_TIMEOUT
                        }
                # 🌙 Moon Dev - Don't raise, continue with partial results
                if not self._is_streamlit():
                    cprint(f"✅ Continuing with {len([r for r in all_responses.values() if r['success']])} successful responses", "green")

        # Generate consensus review summary (with model mapping)
        consensus_summary, model_mapping = self._generate_consensus_review(all_responses, prompt)

        # Clean up responses for easy parsing (extract just the text content)
        clean_responses = {}
        for provider, data in all_responses.items():
            if data["success"]:
                # Extract clean text from ModelResponse
                if hasattr(data['response'], 'content'):
                    response_text = data['response'].content
                else:
                    response_text = str(data['response'])

                # Strip out <think> tags from Ollama responses
                response_text = self._strip_think_tags(response_text)

                clean_responses[provider] = {
                    "response": response_text,
                    "response_time": data["response_time"],
                    "success": True
                }
            else:
                clean_responses[provider] = {
                    "response": None,
                    "error": data["error"],
                    "response_time": data["response_time"],
                    "success": False
                }

        # Prepare results
        total_time = round(time.time() - start_time, 2)

        result = {
            "timestamp": datetime.now().isoformat(),
            "prompt": prompt,
            "system_prompt": system_prompt,
            "consensus_summary": consensus_summary,  # Clean 3-sentence AI review
            "model_mapping": model_mapping,  # Which AI # corresponds to which provider
            "responses": clean_responses,  # Clean, easy-to-parse responses
            "metadata": {
                "total_models": len(self.active_models),
                "successful_responses": sum(1 for r in all_responses.values() if r["success"]),
                "failed_responses": sum(1 for r in all_responses.values() if not r["success"]),
                "total_time": total_time
            }
        }

        # Save results if enabled
        if SAVE_RESULTS:
            self._save_results(result)

        return result

    def _strip_think_tags(self, text: str) -> str:
        """Remove <think>...</think> tags from response text"""
        # Remove <think>...</think> blocks (multiline)
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        # Clean up extra whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()

    def _generate_consensus_review(self, responses: Dict[str, Dict], original_prompt: str) -> Tuple[str, Dict]:
        """
        Generate a consensus review summary using the consensus reviewer AI

        Args:
            responses: All responses from the swarm
            original_prompt: The original user prompt

        Returns:
            Tuple of (consensus_summary, model_mapping)
            - consensus_summary: Clean 3-sentence consensus summary
            - model_mapping: Dict mapping AI numbers to provider names
        """
        try:
            # Get successful responses only
            successful_responses = [
                (provider, r["response"]) for provider, r in responses.items()
                if r["success"] and r["response"]
            ]

            if not successful_responses:
                return "No successful responses to analyze.", {}

            # Build model mapping (AI #1 = claude, AI #2 = openai, etc.)
            model_mapping = {}
            formatted_responses = []
            for i, (provider, response) in enumerate(successful_responses, 1):
                model_mapping[f"AI #{i}"] = provider.upper()

                # Extract clean text
                if hasattr(response, 'content'):
                    response_text = response.content
                else:
                    response_text = str(response)

                # Strip <think> tags before sending to consensus reviewer
                response_text = self._strip_think_tags(response_text)

                # Truncate long responses for the reviewer
                if len(response_text) > 1000:
                    response_text = response_text[:1000] + "..."

                # Don't include provider name in prompt to avoid bias - just use numbers
                formatted_responses.append(f"AI #{i}:\n{response_text}\n")

            # Build the full prompt for consensus reviewer
            responses_text = "\n".join(formatted_responses)
            full_prompt = CONSENSUS_REVIEWER_PROMPT.format(
                num_models=len(successful_responses),
                responses=responses_text
            )

            # Get consensus reviewer model
            model_type, model_name = CONSENSUS_REVIEWER_MODEL
            reviewer_model = model_factory.get_model(model_type, model_name)

            if not reviewer_model:
                return "Consensus reviewer model not available.", model_mapping

            if not self._is_streamlit():
                cprint(f"\n🧠 Generating consensus summary with {model_name}...", "magenta")

            # Generate consensus review
            review_response = reviewer_model.generate_response(
                system_prompt="You are a consensus analyzer. Provide clear, concise 3-sentence summaries.",
                user_content=f"Original Question: {original_prompt}\n\n{full_prompt}",
                temperature=0.3,  # Lower temperature for more focused summary
                max_tokens=200  # Short and concise
            )

            # Extract clean text
            if hasattr(review_response, 'content'):
                consensus_summary = review_response.content.strip()
            else:
                consensus_summary = str(review_response).strip()

            if not self._is_streamlit():
                cprint(f"✅ Consensus summary generated!", "green")

            return consensus_summary, model_mapping

        except Exception as e:
            if not self._is_streamlit():
                cprint(f"❌ Error generating consensus review: {e}", "red")
            return f"Error generating consensus summary: {str(e)}", {}

    def _save_results(self, result: Dict):
        """Save results to JSON file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.results_dir / f"swarm_result_{timestamp}.json"

        with open(filename, 'w') as f:
            json.dump(result, f, indent=2, default=str)

        if not self._is_streamlit():
            cprint(f"\n💾 Results saved to: {filename.relative_to(Path(project_root))}", "blue")

    def _print_summary(self, result: Dict):
        """Print a summary of the swarm results"""
        metadata = result["metadata"]

        cprint("\n" + "="*60, "green")
        cprint("🎯 SWARM CONSENSUS", "green", attrs=['bold'])
        cprint("="*60, "green")

        # Show model mapping first
        if "model_mapping" in result and result["model_mapping"]:
            cprint("\n🔢 Model Key:", "blue")
            for ai_num, provider in result["model_mapping"].items():
                cprint(f"   {ai_num} = {provider}", "white")

        # Show AI-generated consensus summary
        if "consensus_summary" in result:
            cprint("\n🧠 AI CONSENSUS SUMMARY:", "magenta", attrs=['bold'])
            cprint(f"{result['consensus_summary']}\n", "white")

        cprint(f"⚡ Performance:", "cyan")
        cprint(f"   Total Time: {metadata['total_time']}s", "white")
        cprint(f"   Success Rate: {metadata['successful_responses']}/{metadata['total_models']}", "white")

    def query_dataframe(self, prompt: str, system_prompt: Optional[str] = None) -> pd.DataFrame:
        """
        Query swarm and return results as a DataFrame

        Returns:
            DataFrame with columns: provider, response, success, error, response_time
        """
        result = self.query(prompt, system_prompt)

        # Convert responses to DataFrame
        data = []
        for provider, response_data in result["responses"].items():
            data.append({
                "provider": provider,
                "response": response_data["response"][:500] if response_data["response"] else None,
                "success": response_data["success"],
                "error": response_data.get("error"),
                "response_time": response_data["response_time"]
            })

        return pd.DataFrame(data)
    
    # ============================================
    # 🚀 ADVANCED FUNCTIONS
    # ============================================
    
    def add_conversation_context(self, context_type: str, data: Dict):
        """Add persistent context for future queries (trading positions, portfolio, market conditions)"""
        if not hasattr(self, 'context'):
            self.context = {}
        self.context[context_type] = data
        if not self._is_streamlit():
            cprint(f"✅ Added context: {context_type}", "green")
    
    def get_enriched_prompt(self, base_prompt: str) -> str:
        """Automatically enrich prompts with relevant context"""
        enriched = base_prompt
        if hasattr(self, 'context') and self.context:
            enriched += "\n\nContext:\n"
            for ctx_type, ctx_data in self.context.items():
                enriched += f"\n{ctx_type}: {json.dumps(ctx_data, indent=2)}"
        return enriched
    
    def query_with_fallback(self, prompt: str, max_retries: int = 3) -> Dict:
        """Query with automatic retry and fallback to different models on failure"""
        attempts = 0
        last_error = None
        
        while attempts < max_retries:
            try:
                result = self.query(prompt)
                if result['metadata']['successful_responses'] > 0:
                    return result
                last_error = "No successful responses"
            except Exception as e:
                last_error = str(e)
                attempts += 1
                if attempts < max_retries:
                    if not self._is_streamlit():
                        cprint(f"⚠️ Attempt {attempts} failed: {last_error}, retrying...", "yellow")
                    time.sleep(1)  # Brief delay before retry
                else:
                    raise Exception(f"All {max_retries} attempts failed. Last error: {last_error}")
        
        raise Exception(f"Query failed after {max_retries} attempts: {last_error}")
    
    def vote_on_decision(self, question: str, options: List[str]) -> Dict:
        """Have models vote on predefined options (e.g., Buy/Sell/Hold)"""
        voting_prompt = f"""{question}
        
        Vote for ONE option only: {', '.join(options)}
        
        Respond in this exact format:
        VOTE: [your choice]
        REASONING: [brief explanation]
        CONFIDENCE: [0-100]"""
        
        result = self.query(voting_prompt)
        
        votes = {option: 0 for option in options}
        reasoning_by_vote = {option: [] for option in options}
        
        for provider, data in result['responses'].items():
            if data['success']:
                vote, reasoning, confidence = self._parse_vote(data['response'], options)
                if vote:
                    votes[vote] += 1
                    reasoning_by_vote[vote].append({
                        'model': provider,
                        'reasoning': reasoning,
                        'confidence': confidence
                    })
        
        winner = max(votes.items(), key=lambda x: x[1]) if votes else (None, 0)
        
        return {
            'winner': winner[0],
            'vote_count': winner[1],
            'total_votes': sum(votes.values()),
            'vote_breakdown': votes,
            'reasoning': reasoning_by_vote,
            'consensus': result['consensus_summary']
        }
    
    def _parse_vote(self, response: str, options: List[str]) -> Tuple[Optional[str], str, int]:
        """Parse vote, reasoning, and confidence from response"""
        vote = None
        reasoning = ""
        confidence = 50
        
        lines = response.split('\n')
        for line in lines:
            line_upper = line.upper()
            if 'VOTE:' in line_upper:
                for option in options:
                    if option.upper() in line_upper:
                        vote = option
                        break
            elif 'REASONING:' in line_upper:
                reasoning = line.split(':', 1)[1].strip() if ':' in line else ""
            elif 'CONFIDENCE:' in line_upper:
                try:
                    confidence = int(''.join(filter(str.isdigit, line)))
                except:
                    confidence = 50
        
        return vote, reasoning, confidence
    
    def fact_check(self, statement: str) -> Dict:
        """Cross-reference a statement across multiple models for verification"""
        fact_check_prompt = f"""Fact-check this statement. Respond with:
        - VERDICT: True/False/Uncertain
        - CONFIDENCE: 0-100%
        - EVIDENCE: Supporting evidence or counterexamples
        
        Statement: {statement}"""
        
        result = self.query(fact_check_prompt)
        
        verdicts = {'True': 0, 'False': 0, 'Uncertain': 0}
        
        for provider, data in result['responses'].items():
            if data['success']:
                verdict = self._extract_verdict(data['response'])
                verdicts[verdict] += 1
        
        total = sum(verdicts.values())
        agreement_level = max(verdicts.values()) / total if total > 0 else 0
        
        return {
            'verdicts': verdicts,
            'agreement_level': agreement_level,
            'majority_verdict': max(verdicts.items(), key=lambda x: x[1])[0],
            'consensus': result['consensus_summary'],
            'full_result': result
        }
    
    def _extract_verdict(self, response: str) -> str:
        """Extract verdict from response"""
        response_upper = response.upper()
        if 'VERDICT: TRUE' in response_upper or 'IS TRUE' in response_upper:
            return 'True'
        elif 'VERDICT: FALSE' in response_upper or 'IS FALSE' in response_upper:
            return 'False'
        else:
            return 'Uncertain'
    
    def analyze_market_conditions(self, symbol: str, exchange_data: Optional[Dict] = None) -> Dict:
        """Analyze market conditions using exchange data + AI swarm"""
        if exchange_data is None:
            exchange_data = {
                'price': 'N/A',
                'change_24h': 'N/A',
                'volume': 'N/A',
                'rsi': 'N/A',
                'macd': 'N/A'
            }
        
        analysis_prompt = f"""Analyze the current market conditions for {symbol}:
        
        Price: ${exchange_data.get('price', 'N/A')}
        24h Change: {exchange_data.get('change_24h', 'N/A')}%
        Volume: ${exchange_data.get('volume', 'N/A')}
        RSI: {exchange_data.get('rsi', 'N/A')}
        MACD: {exchange_data.get('macd', 'N/A')}
        
        Provide:
        1. Market sentiment (Bullish/Bearish/Neutral)
        2. Key support/resistance levels
        3. Trading recommendation
        4. Risk assessment"""
        
        result = self.query(analysis_prompt)
        
        return {
            'symbol': symbol,
            'market_data': exchange_data,
            'ai_analysis': result['consensus_summary'],
            'individual_analyses': result['responses'],
            'timestamp': datetime.now().isoformat()
        }
    
    def calculate_consensus_strength(self, result: Dict) -> float:
        """Calculate how similar the responses are (0-1 score)"""
        responses = [
            data['response'] for data in result['responses'].values() 
            if data['success'] and data['response']
        ]
        
        if len(responses) < 2:
            return 1.0
        
        # Use simple word overlap for similarity
        word_sets = [set(resp.lower().split()) for resp in responses]
        
        # Calculate pairwise similarity
        similarities = []
        for i in range(len(word_sets)):
            for j in range(i + 1, len(word_sets)):
                intersection = len(word_sets[i] & word_sets[j])
                union = len(word_sets[i] | word_sets[j])
                similarity = intersection / union if union > 0 else 0
                similarities.append(similarity)
        
        return sum(similarities) / len(similarities) if similarities else 0
    
    def get_response_confidence(self, result: Dict) -> Dict:
        """Analyze confidence levels in responses"""
        confidence_scores = {}
        
        for provider, data in result['responses'].items():
            if data['success'] and data['response']:
                response_text = data['response'].lower()
                
                # Simple heuristic scoring
                confidence = 0.5  # default
                
                if any(word in response_text for word in ['definitely', 'certainly', 'absolutely', 'confirmed']):
                    confidence = 0.9
                elif any(word in response_text for word in ['likely', 'probably', 'should', 'expect']):
                    confidence = 0.7
                elif any(word in response_text for word in ['maybe', 'possibly', 'might', 'could']):
                    confidence = 0.4
                elif any(word in response_text for word in ['unlikely', 'doubtful', 'uncertain', 'unclear']):
                    confidence = 0.2
                
                confidence_scores[provider] = confidence
        
        avg_confidence = sum(confidence_scores.values()) / len(confidence_scores) if confidence_scores else 0
        
        return {
            'individual_scores': confidence_scores,
            'average_confidence': avg_confidence,
            'confidence_level': 'High' if avg_confidence > 0.7 else 'Medium' if avg_confidence > 0.4 else 'Low'
        }
    
    def benchmark_models(self, test_prompts: List[str]) -> pd.DataFrame:
        """Benchmark all models on a set of test prompts"""
        benchmark_results = []
        
        for i, prompt in enumerate(test_prompts):
            if not self._is_streamlit():
                cprint(f"\n📊 Benchmarking prompt {i+1}/{len(test_prompts)}...", "cyan")
            
            result = self.query(prompt)
            
            for provider, data in result['responses'].items():
                benchmark_results.append({
                    'Model': provider,
                    'Prompt': prompt[:50] + '...',
                    'Success': data['success'],
                    'Response Time': data['response_time'],
                    'Response Length': len(data['response']) if data['success'] and data['response'] else 0
                })
        
        df = pd.DataFrame(benchmark_results)
        
        # Calculate aggregates
        summary = df.groupby('Model').agg({
            'Success': 'mean',
            'Response Time': 'mean',
            'Response Length': 'mean'
        }).round(2)
        
        summary.columns = ['Success Rate', 'Avg Response Time (s)', 'Avg Response Length']
        
        return summary


class SwarmUI:
    """Streamlit Web Interface for Swarm Agent"""
    
    def __init__(self):
        self.setup_page()
        self.initialize_session_state()
        
    def setup_page(self):
        """Configure Streamlit page settings"""
        st.set_page_config(
            page_title="Moon Dev's AI Swarm",
            page_icon="🌙",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Custom CSS
        st.markdown("""
        <style>
        .typing-indicator {
            display: inline-block;
        }
        .typing-indicator span {
            height: 10px;
            width: 10px;
            background-color: #667eea;
            border-radius: 50%;
            display: inline-block;
            margin: 0 3px;
            animation: bounce 1.4s infinite ease-in-out both;
        }
        
        .typing-indicator span:nth-child(1) { animation-delay: -0.32s; }
        .typing-indicator span:nth-child(2) { animation-delay: -0.16s; }
        
        @keyframes bounce {
            0%, 80%, 100% { transform: scale(0); }
            40% { transform: scale(1); }
        }
        
        /* Toast Notifications */
        .toast {
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 15px 25px;
            border-radius: 10px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.3);
            animation: slideIn 0.5s, slideOut 0.5s 2.5s;
            z-index: 1000;
            color: white;
            font-weight: bold;
        }
        
        .toast.success {
            background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
        }
        
        .toast.error {
            background: linear-gradient(135deg, #f44336 0%, #da190b 100%);
        }
        
        @keyframes slideIn {
            from { transform: translateX(400px); opacity: 0; }
            to { transform: translateX(0); opacity: 1; }
        }
        
        @keyframes slideOut {
            from { transform: translateX(0); opacity: 1; }
            to { transform: translateX(400px); opacity: 0; }
        }
        </style>
        """, unsafe_allow_html=True)
        
    def initialize_session_state(self):
        """Initialize session state variables"""
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        if 'api_keys' not in st.session_state:
            st.session_state.api_keys = {}
        if 'exchange_keys' not in st.session_state:
            st.session_state.exchange_keys = {}
        if 'uploaded_files' not in st.session_state:
            st.session_state.uploaded_files = []
        if 'active_swarm' not in st.session_state:
            st.session_state.active_swarm = None
        if 'query_in_progress' not in st.session_state:
            st.session_state.query_in_progress = False
            
    def render_sidebar(self):
        """Render the sidebar with API key management and model selection"""
        with st.sidebar:
            st.title("🔧 Configuration")
            
            # API Key Management
            with st.expander("🔑 API Keys", expanded=True):
                st.markdown("Enter your API keys to enable models:")
                
                api_providers = {
                    "OpenAI": "openai",
                    "Anthropic (Claude)": "anthropic",
                    "Google (Gemini)": "google",
                    "DeepSeek": "deepseek",
                    "OpenRouter": "openrouter",
                    "XAI (Grok)": "xai"
                }
                
                for display_name, provider_key in api_providers.items():
                    key = st.text_input(
                        f"{display_name} API Key",
                        type="password",
                        value=st.session_state.api_keys.get(provider_key, ""),
                        help=f"Enter your {display_name} API key"
                    )
                    if key:
                        st.session_state.api_keys[provider_key] = key
            
            # Exchange API Management
            with st.expander("📈 Exchange APIs", expanded=False):
                st.markdown("Configure your exchange API credentials for trading analysis:")
                
                exchange_providers = {
                    "Binance": "binance",
                    "Coinbase": "coinbase",
                    "Kraken": "kraken",
                    "Bybit": "bybit",
                    "OKX": "okx",
                    "Bitfinex": "bitfinex"
                }
                
                for display_name, exchange_key in exchange_providers.items():
                    st.markdown(f"**{display_name}**")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        api_key = st.text_input(
                            f"{display_name} API Key",
                            type="password",
                            value=st.session_state.exchange_keys.get(f"{exchange_key}_api_key", ""),
                            help=f"Enter your {display_name} API key",
                            key=f"input_{exchange_key}_api_key"
                        )
                        if api_key:
                            st.session_state.exchange_keys[f"{exchange_key}_api_key"] = api_key
                    
                    with col2:
                        api_secret = st.text_input(
                            f"{display_name} Secret",
                            type="password",
                            value=st.session_state.exchange_keys.get(f"{exchange_key}_secret", ""),
                            help=f"Enter your {display_name} API secret",
                            key=f"input_{exchange_key}_secret"
                        )
                        if api_secret:
                            st.session_state.exchange_keys[f"{exchange_key}_secret"] = api_secret
                    
                    # Optional: Passphrase for exchanges that need it
                    if exchange_key in ['coinbase', 'okx']:
                        passphrase = st.text_input(
                            f"{display_name} Passphrase",
                            type="password",
                            value=st.session_state.exchange_keys.get(f"{exchange_key}_passphrase", ""),
                            help=f"Enter your {display_name} API passphrase (if required)",
                            key=f"input_{exchange_key}_passphrase"
                        )
                        if passphrase:
                            st.session_state.exchange_keys[f"{exchange_key}_passphrase"] = passphrase
                
                # Connection status
                if st.session_state.exchange_keys:
                    st.markdown("---")
                    st.markdown("**📊 Configured Exchanges:**")
                    configured_exchanges = set()
                    for key in st.session_state.exchange_keys.keys():
                        exchange = key.split('_')[0]
                        configured_exchanges.add(exchange)
                    
                    for exchange in sorted(configured_exchanges):
                        st.success(f"✅ {exchange.capitalize()}")
            
            # Model Selection
            with st.expander("🤖 Model Selection", expanded=True):
                st.markdown("Select which models to include in the swarm:")
                
                # Model checkboxes
                model_groups = {
                    "Claude Models": [
                        ("Claude 4.5 Sonnet", "claude", "claude-sonnet-4-5"),
                        ("Claude Opus 4.5", "claude", "claude-opus-4-5-20251101"),
                    ],
                    "OpenAI Models": [
                        ("GPT-5", "openai", "gpt-5"),
                        ("GPT-4 Turbo", "openai", "gpt-4-turbo"),
                    ],
                    "OpenRouter Models": [
                        ("Qwen 3 Max", "openrouter", "qwen/qwen3-max"),
                        ("GPT-5 Mini", "openrouter", "openai/gpt-5-mini"),
                        ("GLM 4.6", "openrouter", "z-ai/glm-4.6"),
                    ],
                    "Other Models": [
                        ("DeepSeek Chat", "deepseek", "deepseek-chat"),
                        ("Grok-4 Fast", "xai", "grok-4-fast-reasoning"),
                    ]
                }
                
                selected_models = {}
                for group_name, models in model_groups.items():
                    st.subheader(group_name)
                    for model_display, provider_type, model_name in models:
                        # Check if API key is available for this provider
                        api_key_available = any(
                            key in provider_type.lower() 
                            for key in st.session_state.api_keys.keys()
                        ) or provider_type in ['deepseek', 'xai']  # Some models might not need explicit keys
                        
                        if api_key_available:
                            enabled = st.checkbox(
                                model_display, 
                                value=True,
                                help=f"{provider_type}: {model_name}"
                            )
                            if enabled:
                                key = f"{provider_type}_{model_name}".replace("/", "_").replace(":", "_")
                                selected_models[key] = (True, provider_type, model_name)
                        else:
                            st.warning(f"⚠️ {model_display} requires API key")
                
                if selected_models:
                    st.session_state.selected_models = selected_models
            
            # File Upload
            with st.expander("📁 Upload Files", expanded=True):
                uploaded_files = st.file_uploader(
                    "Upload files for analysis",
                    type=['txt', 'pdf', 'csv', 'json', 'py', 'md'],
                    accept_multiple_files=True
                )
                
                if uploaded_files:
                    for file in uploaded_files:
                        if file.name not in [f['name'] for f in st.session_state.uploaded_files]:
                            file_content = file.getvalue().decode('utf-8', errors='ignore')
                            st.session_state.uploaded_files.append({
                                'name': file.name,
                                'content': file_content,
                                'type': file.type
                            })
                            st.success(f"✅ {file.name} uploaded")
            
            # Display uploaded files
            if st.session_state.uploaded_files:
                st.subheader("📄 Uploaded Files")
                for file_info in st.session_state.uploaded_files:
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.text(file_info['name'])
                    with col2:
                        if st.button("❌", key=f"remove_{file_info['name']}"):
                            st.session_state.uploaded_files = [
                                f for f in st.session_state.uploaded_files 
                                if f['name'] != file_info['name']
                            ]
                            st.rerun()
            
            # Clear chat button
            if st.button("🗑️ Clear Chat History", type="secondary"):
                st.session_state.chat_history = []
                st.rerun()
                
            # Export results button
            if st.session_state.chat_history:
                if st.button("💾 Export Chat History", type="primary"):
                    self.export_chat_history()
    
    def export_chat_history(self):
        """Export chat history to JSON file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"swarm_chat_history_{timestamp}.json"
        
        export_data = {
            "timestamp": datetime.now().isoformat(),
            "chat_history": st.session_state.chat_history
        }
        
        st.download_button(
            label="📥 Download Chat History",
            data=json.dumps(export_data, indent=2),
            file_name=filename,
            mime="application/json"
        )
    
    def render_swarm_animation(self):
        """Animated visualization of AI models working in parallel"""
        st.markdown("""
        <div class="swarm-container">
            <div class="ai-node active">🤖</div>
            <div class="ai-node active">🧠</div>
            <div class="ai-node active">💡</div>
            <div class="ai-node active">⚡</div>
            <div class="ai-node active">🔮</div>
        </div>
        """, unsafe_allow_html=True)
    
    def show_typing_indicator(self):
        """Show AI is thinking animation"""
        return st.markdown("""
        <div class="typing-indicator">
            <span></span>
            <span></span>
            <span></span>
        </div>
        """, unsafe_allow_html=True)
    
    def show_toast(self, message, toast_type="success"):
        """Animated toast notifications"""
        icon = "✅" if toast_type == "success" else "❌"
        st.markdown(f"""
        <div class="toast {toast_type}">
            {icon} {message}
        </div>
        """, unsafe_allow_html=True)
    
    def render_live_metrics(self, result):
        """Real-time response dashboard with metrics"""
        if not result or 'responses' not in result:
            return
        
        # Collect response data
        response_times = []
        providers = []
        statuses = []
        
        for provider, data in result['responses'].items():
            providers.append(provider.upper())
            response_times.append(data.get('response_time', 0))
            statuses.append("✅" if data.get('success') else "❌")
        
        if not providers:
            return
        
        # Create metrics columns
        st.markdown("### 📊 Response Metrics Dashboard")
        
        cols = st.columns(len(providers))
        for i, (col, provider, time, status) in enumerate(zip(cols, providers, response_times, statuses)):
            with col:
                st.metric(
                    label=f"{status} {provider}",
                    value=f"{time:.2f}s",
                    delta=None
                )
        
        # Bar chart visualization
        try:
            import plotly.graph_objects as go
            
            fig = go.Figure(data=[
                go.Bar(
                    x=providers,
                    y=response_times,
                    marker_color='#667eea',
                    text=[f"{t:.2f}s" for t in response_times],
                    textposition='auto',
                )
            ])
            
            fig.update_layout(
                title="⏱️ Response Times by Model",
                xaxis_title="AI Model",
                yaxis_title="Time (seconds)",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                height=300,
            )
            
            st.plotly_chart(fig, use_container_width=True)
        except ImportError:
            # Fallback if plotly not installed
            st.bar_chart(pd.DataFrame({
                'Model': providers,
                'Response Time (s)': response_times
            }).set_index('Model'))
    
    def render_consensus_strength_meter(self, consensus_score: float):
        """Visual meter showing how much models agree"""
        percentage = int(consensus_score * 100)
        color = "#4CAF50" if percentage > 70 else "#FFA500" if percentage > 40 else "#f44336"
        
        st.markdown(f"""
        <div style="margin: 20px 0;">
            <h4>🎯 Model Agreement Level</h4>
            <div style="width: 100%; height: 30px; background: #e0e0e0; border-radius: 15px; overflow: hidden;">
                <div style="
                    width: {percentage}%;
                    height: 100%;
                    background: linear-gradient(90deg, {color} 0%, {color}dd 100%);
                    transition: width 1s;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    color: white;
                    font-weight: bold;
                ">
                    {percentage}% Agreement
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    def render_response_comparison(self, result):
        """Side-by-side comparison of model responses"""
        comparison_data = []
        for provider, data in result['responses'].items():
            if data['success']:
                comparison_data.append({
                    'Model': provider.upper(),
                    'Response Preview': data['response'][:200] + '...' if data['response'] else '',
                    'Response Time': f"{data['response_time']}s",
                    'Status': '✅'
                })
            else:
                comparison_data.append({
                    'Model': provider.upper(),
                    'Response Preview': f"Error: {data['error']}",
                    'Response Time': f"{data['response_time']}s",
                    'Status': '❌'
                })
        
        df = pd.DataFrame(comparison_data)
        st.dataframe(df, use_container_width=True)
    
    def render_query_templates(self):
        """UI for managing and using query templates"""
        if 'query_templates' not in st.session_state:
            st.session_state.query_templates = {
                'Market Analysis': {
                    'template': 'Analyze {symbol} market conditions. Current price: {price}. Should I {action}?',
                    'variables': ['symbol', 'price', 'action'],
                    'created': datetime.now().isoformat()
                },
                'Risk Assessment': {
                    'template': 'Assess the risk of {trade_type} {amount} {symbol} at {price}. Portfolio size: {portfolio}',
                    'variables': ['trade_type', 'amount', 'symbol', 'price', 'portfolio'],
                    'created': datetime.now().isoformat()
                }
            }
        
        st.markdown("### 📝 Query Templates")
        
        template_name = st.selectbox(
            "Select Template",
            options=["None"] + list(st.session_state.query_templates.keys())
        )
        
        if template_name != "None":
            template = st.session_state.query_templates[template_name]
            st.code(template['template'])
            
            # Show variable inputs
            variables = {}
            cols = st.columns(len(template['variables']))
            for i, var in enumerate(template['variables']):
                with cols[i]:
                    variables[var] = st.text_input(f"{var}:", key=f"template_var_{var}")
            
            if st.button("🚀 Apply Template") and all(variables.values()):
                prompt = template['template']
                for var, value in variables.items():
                    prompt = prompt.replace(f"{{{var}}}", value)
                return prompt
        
        return None
    
    def render_advanced_features(self, swarm):
        """Render advanced feature tabs"""
        tab1, tab2, tab3, tab4 = st.tabs(["🗳️ Voting", "✅ Fact Check", "📈 Market Analysis", "📊 Benchmark"])
        
        with tab1:
            st.markdown("### 🗳️ AI Voting System")
            question = st.text_area("Question:", "Should I buy Bitcoin now?")
            options = st.text_input("Options (comma-separated):", "Buy,Sell,Hold")
            
            if st.button("🗳️ Start Vote"):
                with st.spinner("Collecting votes from AI models..."):
                    options_list = [opt.strip() for opt in options.split(',')]
                    vote_result = swarm.vote_on_decision(question, options_list)
                    
                    st.success(f"🏆 Winner: **{vote_result['winner']}** ({vote_result['vote_count']}/{vote_result['total_votes']} votes)")
                    
                    # Vote breakdown
                    st.markdown("#### Vote Breakdown")
                    cols = st.columns(len(options_list))
                    for i, option in enumerate(options_list):
                        with cols[i]:
                            st.metric(option, vote_result['vote_breakdown'][option])
                    
                    st.markdown("#### Reasoning by Vote")
                    for option, reasonings in vote_result['reasoning'].items():
                        if reasonings:
                            with st.expander(f"{option} - {len(reasonings)} vote(s)"):
                                for r in reasonings:
                                    st.markdown(f"**{r['model']}** (Confidence: {r['confidence']}%)")
                                    st.write(r['reasoning'])
        
        with tab2:
            st.markdown("### ✅ Fact Checker")
            statement = st.text_area("Statement to fact-check:", "Bitcoin is the first cryptocurrency.")
            
            if st.button("✅ Check Facts"):
                with st.spinner("Cross-referencing across AI models..."):
                    fact_result = swarm.fact_check(statement)
                    
                    # Verdict display
                    verdict_color = "green" if fact_result['majority_verdict'] == 'True' else "red" if fact_result['majority_verdict'] == 'False' else "orange"
                    st.markdown(f"### Verdict: :{verdict_color}[{fact_result['majority_verdict']}]")
                    st.progress(fact_result['agreement_level'])
                    st.caption(f"Agreement Level: {fact_result['agreement_level']*100:.1f}%")
                    
                    # Breakdown
                    cols = st.columns(3)
                    for i, (verdict, count) in enumerate(fact_result['verdicts'].items()):
                        with cols[i]:
                            st.metric(verdict, count)
                    
                    st.markdown("#### AI Consensus")
                    st.info(fact_result['consensus'])
        
        with tab3:
            st.markdown("### 📈 Market Analysis")
            col1, col2 = st.columns(2)
            with col1:
                symbol = st.text_input("Symbol:", "BTC/USD")
            with col2:
                price = st.number_input("Current Price:", value=50000.0)
            
            if st.button("📈 Analyze Market"):
                with st.spinner("Analyzing market conditions..."):
                    market_data = {
                        'price': price,
                        'change_24h': 'N/A',
                        'volume': 'N/A',
                        'rsi': 'N/A',
                        'macd': 'N/A'
                    }
                    analysis = swarm.analyze_market_conditions(symbol, market_data)
                    
                    st.markdown("#### 🧠 AI Analysis")
                    st.success(analysis['ai_analysis'])
                    
                    st.markdown("#### 📊 Market Data")
                    st.json(analysis['market_data'])
        
        with tab4:
            st.markdown("### 📊 Model Benchmarking")
            test_prompts_input = st.text_area(
                "Test prompts (one per line):",
                "What is 2+2?\nExplain quantum computing\nShould I invest in tech stocks?"
            )
            
            if st.button("🏃 Run Benchmark"):
                test_prompts = [p.strip() for p in test_prompts_input.split('\n') if p.strip()]
                
                with st.spinner(f"Benchmarking {len(test_prompts)} prompts..."):
                    benchmark_df = swarm.benchmark_models(test_prompts)
                    
                    st.markdown("#### Benchmark Results")
                    st.dataframe(benchmark_df, use_container_width=True)
                    
                    # Highlight best performer
                    best_time = benchmark_df['Avg Response Time (s)'].idxmin()
                    best_success = benchmark_df['Success Rate'].idxmax()
                    
                    st.success(f"🏆 Fastest: **{best_time}** | Most Reliable: **{best_success}**")
    
    def render_chat_interface(self):
        """Render the main chat interface"""
        # Header
        col1, col2 = st.columns([3, 1])
        with col1:
            st.title("🌙 Moon Dev's AI Swarm Agent")
            st.markdown(STREAMLIT_DESCRIPTION)
        
        # Show swarm animation
        self.render_swarm_animation()
        
        # Query templates
        template_prompt = self.render_query_templates()
        if template_prompt:
            st.info(f"📝 Template applied: {template_prompt}")
        
        # Chat history display
        chat_container = st.container()
        
        with chat_container:
            for message in st.session_state.chat_history:
                if message['role'] == 'user':
                    with st.chat_message("user"):
                        st.markdown(f"**You:** {message['content']}")
                        if 'files' in message and message['files']:
                            st.caption(f"📎 Attached files: {', '.join(message['files'])}")
                else:
                    with st.chat_message("assistant"):
                        st.markdown(f"**AI Swarm Consensus:**")
                        st.markdown(
                            f'<div class="consensus-box">{message["consensus_summary"]}</div>',
                            unsafe_allow_html=True
                        )
                        
                        # Show live metrics dashboard
                        if 'responses' in message:
                            self.render_live_metrics(message)
                            
                            # Show consensus strength meter
                            if 'metadata' in message:
                                swarm_temp = SwarmAgent()
                                consensus_score = swarm_temp.calculate_consensus_strength(message)
                                self.render_consensus_strength_meter(consensus_score)
                            
                            # Show response comparison table
                            with st.expander("📊 Response Comparison Table"):
                                self.render_response_comparison(message)
                        
                        # Show individual responses in expanders
                        with st.expander("👁️ View Individual Model Responses"):
                            for provider, data in message.get('responses', {}).items():
                                if data['success']:
                                    st.markdown(f"**{provider.upper()}** (⏱️ {data['response_time']}s)")
                                    st.markdown(
                                        f'<div class="success-response">{data["response"][:500]}...</div>',
                                        unsafe_allow_html=True
                                    )
                                else:
                                    st.markdown(f"**{provider.upper()}** ❌")
                                    st.markdown(
                                        f'<div class="error-response">Error: {data["error"]}</div>',
                                        unsafe_allow_html=True
                                    )
        
        # Chat input
        prompt = st.chat_input("Ask the AI swarm...", disabled=st.session_state.query_in_progress)
        
        if prompt:
            # Add user message to chat
            st.session_state.chat_history.append({
                'role': 'user',
                'content': prompt,
                'timestamp': datetime.now().isoformat(),
                'files': [f['name'] for f in st.session_state.uploaded_files]
            })
            
            # Initialize swarm with selected models and API keys
            if 'selected_models' in st.session_state:
                swarm = SwarmAgent(
                    custom_models=st.session_state.selected_models,
                    api_keys=st.session_state.api_keys
                )
                
                # Prepare prompt with file content if uploaded
                full_prompt = prompt
                if st.session_state.uploaded_files:
                    file_context = "\n\nAttached Files:\n"
                    for file_info in st.session_state.uploaded_files:
                        file_context += f"\n--- {file_info['name']} ---\n"
                        file_context += file_info['content'][:5000] + ("..." if len(file_info['content']) > 5000 else "")
                    full_prompt += file_context
                
                # Query the swarm
                with st.spinner("🤖 Querying AI swarm..."):
                    st.session_state.query_in_progress = True
                    
                    # Show typing indicator
                    typing_placeholder = st.empty()
                    with typing_placeholder:
                        self.show_typing_indicator()
                    
                    # Create progress bar
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Simulate progress updates
                    for i in range(5):
                        progress_bar.progress((i + 1) * 20)
                        status_text.text(f"Querying AI models... {20*(i+1)}%")
                        time.sleep(0.5)
                    
                    try:
                        result = swarm.query(full_prompt)
                        
                        # Clear typing indicator
                        typing_placeholder.empty()
                        
                        # Show success toast
                        self.show_toast("✨ Swarm query completed successfully!", "success")
                        
                        # Add AI response to chat
                        st.session_state.chat_history.append({
                            'role': 'assistant',
                            'content': result['consensus_summary'],
                            'consensus_summary': result['consensus_summary'],
                            'responses': result['responses'],
                            'metadata': result['metadata'],
                            'timestamp': datetime.now().isoformat()
                        })
                        
                        # Clear uploaded files after query
                        st.session_state.uploaded_files = []
                        
                        st.session_state.query_in_progress = False
                        st.rerun()
                        
                    except Exception as e:
                        typing_placeholder.empty()
                        self.show_toast(f"Error: {str(e)}", "error")
                        st.error(f"Error querying swarm: {str(e)}")
                        st.session_state.query_in_progress = False
            else:
                st.warning("⚠️ Please select at least one model in the sidebar configuration.")
    
    def render_dashboard(self):
        """Render dashboard with swarm statistics"""
        if st.session_state.chat_history:
            st.subheader("📊 Swarm Statistics")
            
            # Calculate statistics
            ai_responses = [msg for msg in st.session_state.chat_history if msg['role'] == 'assistant']
            
            if ai_responses:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    total_queries = len(ai_responses)
                    st.metric("Total Queries", total_queries)
                
                with col2:
                    avg_response_time = sum(
                        msg['metadata']['total_time'] 
                        for msg in ai_responses 
                        if 'metadata' in msg
                    ) / len(ai_responses)
                    st.metric("Avg Response Time", f"{avg_response_time:.1f}s")
                
                with col3:
                    total_models = sum(
                        msg['metadata']['total_models'] 
                        for msg in ai_responses 
                        if 'metadata' in msg
                    )
                    st.metric("Total Models Used", total_models)
    
    def run(self):
        """Run the Streamlit UI"""
        self.render_sidebar()
        
        # Create tabs for main interface
        tab1, tab2 = st.tabs(["💬 Chat", "🚀 Advanced Features"])
        
        with tab1:
            self.render_chat_interface()
            self.render_dashboard()
        
        with tab2:
            # Initialize swarm for advanced features
            if 'selected_models' in st.session_state and st.session_state.selected_models:
                swarm = SwarmAgent(
                    custom_models=st.session_state.selected_models,
                    api_keys=st.session_state.api_keys
                )
                self.render_advanced_features(swarm)
            else:
                st.warning("⚠️ Please configure models in the sidebar first.")


def main():
    """Entry point for the application"""
    
    # Check if running with web UI flag
    if len(sys.argv) > 1 and sys.argv[1] == "--web-ui":
        # Run Streamlit UI
        ui = SwarmUI()
        ui.run()
    else:
        # Run terminal interface
        cprint("\n" + "="*60, "cyan")
        cprint("🌙 Moon Dev's Swarm Agent 🌙", "cyan", attrs=['bold'])
        cprint("="*60, "cyan")
        
        # Show usage options
        cprint("\nUsage Options:", "yellow")
        cprint("  1. Terminal mode (current)", "green")
        cprint("  2. Web UI mode: python swarm_agent.py --web-ui", "green")
        
        choice = input("\nSelect mode (1 or 2): ").strip()
        
        if choice == "2":
            cprint("\n🚀 Starting Web UI...", "cyan")
            cprint("👉 Run: streamlit run src/agents/swarm_agent.py -- --web-ui", "yellow")
            sys.exit(0)
        
        # Continue with terminal mode
        swarm = SwarmAgent()
        
        # Ask for prompt
        cprint("\n💭 What would you like to ask the AI swarm?", "yellow")
        prompt = input("🌙 Prompt > ").strip()
        
        if not prompt:
            cprint("❌ No prompt provided. Exiting.", "red")
            return
        
        # Query the swarm
        result = swarm.query(prompt)
        
        # Show individual responses
        cprint("\n" + "="*60, "cyan")
        cprint("📋 AI RESPONSES", "cyan", attrs=['bold'])
        cprint("="*60, "cyan")
        
        # Create reverse mapping to show AI numbers
        reverse_mapping = {}
        if "model_mapping" in result:
            for ai_num, provider in result["model_mapping"].items():
                reverse_mapping[provider.lower()] = ai_num
        
        for provider, data in result["responses"].items():
            if data["success"]:
                # Get AI number if available
                ai_label = reverse_mapping.get(provider, "")
                if ai_label:
                    cprint(f"\n🤖 {ai_label} ({provider.upper()}):", "yellow", attrs=['bold'])
                else:
                    cprint(f"\n🤖 {provider.upper()}:", "yellow", attrs=['bold'])
                
                response_text = data['response']
                
                # Truncate if too long (show first 800 chars)
                if len(response_text) > 800:
                    cprint(f"{response_text[:800]}...\n", "white")
                    cprint("[Response truncated - see full output in saved JSON]", "cyan")
                else:
                    cprint(f"{response_text}", "white")
                
                cprint(f"⏱️  Response time: {data['response_time']}s", "cyan")
            else:
                cprint(f"\n❌ {provider.upper()}: Failed - {data['error']}", "red")
        
        # Show summary
        swarm._print_summary(result)
        
        cprint("\n✨ Swarm query complete! 🌙", "cyan", attrs=['bold'])


if __name__ == "__main__":
    main()