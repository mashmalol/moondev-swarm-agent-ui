# 🌙 Moon Dev's Swarm Agent

A powerful multi-AI swarm agent that queries multiple language models in parallel (Claude, GPT, DeepSeek, Grok, Qwen, etc.) and generates consensus summaries. Perfect for trading analysis, decision-making, and research.

## Features

- **Multi-Model Queries**: Query 5-7 AI models simultaneously using ThreadPoolExecutor
- **Consensus Generation**: AI-powered synthesis of all responses using DeepSeek
- **Dual Interface**: Terminal CLI and Streamlit web UI
- **Advanced Features**: Voting, fact-checking, benchmarking, market analysis
- **File Upload**: Analyze PDFs, CSVs, JSON files
- **Live Metrics**: Real-time dashboard with response times and success rates

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/mashmalol/moondev-swarm-agent-ui.git
cd moondev-swarm-agent-ui

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy environment template
cp .env.example .env
# Edit .env and add your API keys
```

### Running the App

**Web UI (Recommended):**
```bash
streamlit run swarm_agent.py -- --web-ui
```

**Terminal CLI:**
```bash
python swarm_agent.py
```

**As Python Module:**
```python
from swarm_agent import SwarmAgent

swarm = SwarmAgent()
result = swarm.query("Should I buy Bitcoin?")
print(result["consensus_summary"])
```

## Configuration

### API Keys

Add your API keys to `.env`:
```bash
ANTHROPIC_API_KEY=sk-ant-xxxxx
OPENAI_API_KEY=sk-xxxxx
DEEPSEEK_API_KEY=sk-xxxxx
XAI_API_KEY=xai-xxxxx
OPENROUTER_API_KEY=sk-or-xxxxx
```

### Model Selection

Edit `SWARM_MODELS` in `swarm_agent.py`:
```python
SWARM_MODELS = {
    "deepseek": (True, "deepseek", "deepseek-chat"),
    "claude": (True, "claude", "claude-sonnet-4-5"),
    "gpt5": (True, "openai", "gpt-5"),
    # ... add more models
}
```

## Architecture

- **SwarmAgent**: Orchestrates parallel model queries
- **ModelFactory**: Factory pattern for creating model instances
- **SwarmUI**: Streamlit web interface with advanced features

## Response Structure

```python
{
  "consensus_summary": "AI-generated 3-sentence summary",
  "model_mapping": {"AI #1": "CLAUDE", "AI #2": "OPENAI"},
  "responses": {
    "provider": {
      "response": "clean_text",
      "response_time": 1.23,
      "success": true
    }
  },
  "metadata": {
    "total_models": 5,
    "successful_responses": 5,
    "total_time": 3.45
  }
}
```

## Advanced Features

- **Voting**: Multi-model voting on predefined options (Buy/Sell/Hold)
- **Fact Checking**: Cross-reference statements across models
- **Benchmarking**: Test response times and success rates
- **Market Analysis**: AI-powered trading analysis
- **Context Enrichment**: Persistent conversation context

## Contributing

Contributions welcome! See [.github/copilot-instructions.md](.github/copilot-instructions.md) for developer guidance.

## Built by Moon Dev 🌙

For trading analysis, AI research, and decision-making.
