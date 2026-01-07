# Visualization Guide Tester

AI-powered tool for testing and improving a Plotly visualization style guide.

## Prerequisites

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) if not already installed:

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or with Homebrew
brew install uv

# Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

## Quick Start

1. Clone and setup:
   ```bash
   git clone <repo-url>
   cd viz_guide
   uv sync
   ```

2. Configure API key:
   ```bash
   cp .env.example .env
   # Edit .env with your Anthropic API key
   ```

3. Run the app:
   ```bash
   uv run streamlit run app/main.py
   ```

## Workflow

1. **Generate**: Describe a visualization, paste data sample
2. **Refine**: Provide feedback, iterate until satisfied
3. **Approve**: When chart is correct, approve to save capture
4. **Suggest**: Review guide improvements, apply via PR

## Contributing Guide Changes

Guide improvements should go through PR review:

1. Create a branch: `git checkout -b guide/your-improvement`
2. Apply suggestions or edit guide/*.md directly
3. Test with the app to verify improvement works
4. Push and create PR for review

## Project Structure

- `app/` - Streamlit application
- `guide/` - 14-section visualization style guide
- `feedback/captures/` - Gallery of approved visualizations
- `references/` - Logo/watermark images
