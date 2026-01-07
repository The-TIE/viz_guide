"""Random visualization prompt generation using AI."""

import os
import random
from pathlib import Path

from dotenv import load_dotenv
import anthropic

# Load .env file
load_dotenv(Path(__file__).parent.parent / ".env")

# Fallback prompts if API is unavailable (analytical questions, not chart specs)
FALLBACK_PROMPTS = [
    "How has Bitcoin's price trended over the last 6 months?",
    "Compare the daily return patterns of ETH and SOL",
    "Which tokens have the largest market caps?",
    "How correlated are the top 10 tokens with each other?",
    "What does the distribution of BTC daily returns look like?",
    "Is there a relationship between trading volume and price changes?",
    "What's the market share breakdown among stablecoins?",
    "How do funding rates compare across major exchanges?",
    "Which DeFi protocols have the highest TVL?",
    "How volatile are different crypto assets compared to each other?",
]

PROMPT_GEN_SYSTEM = """You are a financial data analyst. Generate analytical questions that someone might want to answer with a visualization.

IMPORTANT: Generate QUESTIONS about data, NOT specifications for charts.

GOOD examples (analytical questions):
- "How has Bitcoin's price volatility changed over the past year?"
- "Which exchanges have the highest trading volume?"
- "What's the correlation between ETH and SOL returns?"
- "How is trading volume distributed across different times of day?"
- "Compare the market cap growth of the top 5 DeFi protocols"

BAD examples (chart specifications - DO NOT generate these):
- "Create a violin plot with white dots showing median..."
- "Make a horizontal stacked bar chart with..."
- "Display a 3D surface showing..."

The question should describe WHAT insight is needed, not HOW to visualize it.
Let the visualization system figure out the best chart type.

Data domains to draw questions from:
- Price trends and performance
- Volume and liquidity patterns
- Return distributions and risk
- Correlations between assets
- Market share and rankings
- Time-based patterns (hourly, daily, weekly)
- Derivatives and funding rates
- DeFi protocol metrics
- Exchange comparisons
"""


def get_random_prompt(api_key: str | None = None) -> str:
    """
    Generate a random visualization prompt using AI.

    Falls back to static list if API unavailable.
    """
    api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")

    if not api_key:
        return random.choice(FALLBACK_PROMPTS)

    try:
        client = anthropic.Anthropic(api_key=api_key)

        # Add some randomness to the request
        topic_hints = [
            "Ask about price trends or performance.",
            "Ask about comparing multiple assets.",
            "Ask about trading volume patterns.",
            "Ask about return distributions or risk.",
            "Ask about correlations between assets.",
            "Ask about market share or rankings.",
            "Ask about time-based patterns.",
            "Ask about exchange comparisons.",
        ]

        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=150,
            temperature=0.9,  # High temperature for variety
            system=PROMPT_GEN_SYSTEM,
            messages=[{
                "role": "user",
                "content": f"Generate ONE analytical question about financial/crypto data. {random.choice(topic_hints)} Return ONLY the question, no explanation. Do NOT specify chart types."
            }]
        )

        prompt = response.content[0].text.strip()
        # Clean up any quotes or extra formatting
        prompt = prompt.strip('"\'')
        return prompt

    except Exception as e:
        print(f"AI prompt generation failed: {e}")
        return random.choice(FALLBACK_PROMPTS)


def get_random_prompts(n: int = 5, api_key: str | None = None) -> list[str]:
    """Return n random visualization prompts."""
    return [get_random_prompt(api_key) for _ in range(n)]
