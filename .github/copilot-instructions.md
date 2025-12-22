# Moon Dev's Swarm Agent - AI Instructions

## Project Overview
This is a **multi-AI swarm agent** that queries multiple language models in parallel (Claude, GPT, DeepSeek, Grok, Qwen, etc.) and generates a consensus summary. The project offers both terminal CLI and Streamlit web UI interfaces for trading analysis, decision-making, and multi-perspective AI queries.

## Architecture

### Core Components
1. **SwarmAgent** ([swarm_agent.py](swarm_agent.py)) - Main orchestrator that:
   - Queries 5-7 AI models concurrently using `ThreadPoolExecutor`
   - Strips `<think>` tags from reasoning models (Ollama)
   - Generates AI consensus using DeepSeek as reviewer
   - Returns structured results with model mapping (AI #1, AI #2, etc.)

2. **ModelFactory** ([src/models/model_factory.py](src/models/model_factory.py)) - Factory pattern for model creation:
   - Uses OpenAI-compatible clients for DeepSeek, XAI, OpenRouter
   - Falls back to `MockModel` when API keys are missing
   - Each model returns `ModelResponse` wrapper with `.content` attribute

3. **SwarmUI** ([swarm_agent.py](swarm_agent.py) lines 1100+) - Streamlit interface with:
   - Live metrics dashboard with Plotly charts
   - Advanced features: voting, fact-checking, benchmarking
   - File upload support (PDF/CSV/JSON analysis)
   - Query templates for common tasks

### Data Flow
```
User Prompt → SwarmAgent.query()
  ↓ (ThreadPoolExecutor parallel execution)
  ├→ Model 1 → _query_single_model() → ModelResponse
  ├→ Model 2 → _query_single_model() → ModelResponse
  └→ Model N → _query_single_model() → ModelResponse
    ↓ (all responses collected)
  → _strip_think_tags() → Clean responses
  → _generate_consensus_review() → Consensus summary
  → _save_results() → JSON file (if enabled)
  → Return structured dict with responses + consensus
```

## Key Conventions

### Model Configuration Pattern
Edit `SWARM_MODELS` dict in [swarm_agent.py](swarm_agent.py#L58-L80):
```python
"provider_key": (enabled: bool, model_type: str, model_name: str)
```
- Set first value to `True` to enable
- Provider types: `claude`, `openai`, `deepseek`, `xai`, `openrouter`, `ollama`
- OpenRouter gives access to 200+ models via single API

### Environment Variables
Copy [.env.example](.env.example) to `.env` and configure:
- `ANTHROPIC_API_KEY` - Claude models
- `OPENAI_API_KEY` - GPT models
- `DEEPSEEK_API_KEY` - DeepSeek models
- `XAI_API_KEY` - Grok models
- `OPENROUTER_API_KEY` - 200+ models via OpenRouter
- Ollama uses local HTTP (no key needed)

### Response Structure
All API calls return:
```python
{
  "timestamp": ISO datetime,
  "prompt": original question,
  "consensus_summary": "3-sentence AI review",
  "model_mapping": {"AI #1": "CLAUDE", "AI #2": "OPENAI", ...},
  "responses": {
    "provider": {
      "response": clean_text,  # <think> tags stripped
      "response_time": float,
      "success": bool
    }
  },
  "metadata": {
    "total_models": int,
    "successful_responses": int,
    "total_time": float
  }
}
```

## Developer Workflows

### Running the Project
```bash
# Terminal mode (interactive CLI)
python swarm_agent.py

# Web UI (Streamlit)
streamlit run swarm_agent.py -- --web-ui
# OR
python swarm_agent.py --web-ui  # then follow prompt

# Import as module
from swarm_agent import SwarmAgent
swarm = SwarmAgent()
result = swarm.query("Should I buy Bitcoin?")
```

### Adding New AI Models
1. Add provider to `SWARM_MODELS` dict in [swarm_agent.py](swarm_agent.py#L58)
2. Implement `_get_<provider>_model()` in [model_factory.py](src/models/model_factory.py)
3. Use OpenAI-compatible client pattern (see DeepSeek/XAI examples)
4. Add API key to [.env.example](.env.example)

### Debugging Tips
- Results saved to `src/data/swarm_agent/swarm_result_*.json` (if `SAVE_RESULTS=True`)
- Streamlit context: Use `_is_streamlit()` check before `cprint()` calls
- Model timeout: `MODEL_TIMEOUT = 120` seconds (line 91)
- Failed models don't block - swarm continues with partial results

### Testing Changes
No formal test suite yet. Verify by:
```bash
# Quick test with real models
python swarm_agent.py
# Enter prompt: "What is 2+2?"
# Check all models respond correctly

# UI test
streamlit run swarm_agent.py -- --web-ui
# Test file upload, voting, and advanced features
```

## Project-Specific Patterns

### Parallel Execution
- Uses `ThreadPoolExecutor` with `as_completed()` for real-time results
- Timeout handling: individual model timeouts + global timeout
- Never raises on timeout - returns partial results

### Think Tag Stripping
DeepSeek-R1 and Ollama reasoning models wrap thinking in `<think>` tags:
```python
def _strip_think_tags(self, text: str) -> str:
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    return text.strip()
```
Called automatically before returning responses and before consensus review.

### Consensus Generation
- Uses separate AI (DeepSeek Chat) to synthesize all responses
- Never shows provider names in consensus prompt - uses "AI #1", "AI #2" to avoid bias
- Limited to 3 sentences, 200 tokens max
- Temperature 0.3 for focused summaries

### Streamlit State Management
- API keys stored in `st.session_state.api_keys`
- Chat history in `st.session_state.chat_history`
- Models reinitialized on config change (sidebar updates)

## Integration Points

### External APIs
- **Anthropic SDK** - Direct Claude API calls
- **OpenAI SDK** - Used for GPT, DeepSeek, XAI, OpenRouter (all OpenAI-compatible)
- **Ollama** - Local HTTP requests to `localhost:11434`
- **OpenRouter** - Proxy to 200+ models via `openrouter.ai/api/v1`

### File Processing
- Upload handling in `render_sidebar()` → stores in `st.session_state.uploaded_files`
- Content truncated to 5000 chars per file before sending to models
- Supported: TXT, PDF, CSV, JSON, PY, MD

## Common Pitfalls

1. **Missing API Keys** - Models fall back to `MockModel` silently. Check terminal output for "⚠️ No API key" warnings.
2. **Streamlit `cprint()` errors** - Always wrap terminal output in `if not self._is_streamlit()` checks.
3. **Timeout confusion** - `MODEL_TIMEOUT` is per-model. Global timeout is `MODEL_TIMEOUT + 10`.
4. **OpenRouter headers** - Must include `HTTP-Referer` and `X-Title` headers (see [model_factory.py](src/models/model_factory.py#L237-L240)).
5. **Model name format** - OpenRouter uses `provider/model-name` format (e.g., `qwen/qwen3-max`), not just model name.

## Advanced Features

- **Voting** - `vote_on_decision()` for predefined options (Buy/Sell/Hold)
- **Fact Checking** - `fact_check()` cross-references statement across models
- **Benchmarking** - `benchmark_models()` tests response times/success rates
- **Consensus Strength** - `calculate_consensus_strength()` measures response similarity
- **Context Enrichment** - `add_conversation_context()` for persistent state (trading positions, portfolio)

---

*Generated for AI coding agents. Focus: architecture, conventions, integration points, and workflow commands.*
