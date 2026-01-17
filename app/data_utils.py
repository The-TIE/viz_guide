"""Data loading and synthetic data generation utilities."""

import asyncio
import io
import json
import os
import re
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from claude_agent_sdk import query, ClaudeAgentOptions, AssistantMessage, TextBlock


def load_csv(file) -> pd.DataFrame:
    """Load CSV file into DataFrame."""
    return pd.read_csv(file)


def load_json(file) -> pd.DataFrame:
    """Load JSON file into DataFrame."""
    content = file.read()
    if isinstance(content, bytes):
        content = content.decode("utf-8")
    data = json.loads(content)

    # Handle different JSON structures
    if isinstance(data, list):
        return pd.DataFrame(data)
    elif isinstance(data, dict):
        # Try to find array in dict values
        for key, value in data.items():
            if isinstance(value, list):
                return pd.DataFrame(value)
        # Otherwise treat dict as single row or columns
        return pd.DataFrame(data)

    return pd.DataFrame()


def load_file(file, filename: str) -> pd.DataFrame:
    """Load file based on extension."""
    if filename.endswith(".csv"):
        return load_csv(file)
    elif filename.endswith(".json"):
        return load_json(file)
    else:
        raise ValueError(f"Unsupported file type: {filename}")


def dataframe_to_sample(df: pd.DataFrame, max_rows: int = 10) -> str:
    """Convert DataFrame to sample string for the prompt."""
    sample_df = df.head(max_rows)

    # Include column info
    info_lines = [
        f"Columns: {list(df.columns)}",
        f"Shape: {df.shape[0]} rows x {df.shape[1]} columns",
        f"Data types: {df.dtypes.to_dict()}",
        "",
        "Sample data (first rows):",
        sample_df.to_string(index=False),
    ]

    return "\n".join(info_lines)


DATA_GEN_PROMPT = """Generate Python code that creates a pandas DataFrame with realistic sample data for this visualization:

DESCRIPTION: {description}

REQUIREMENTS:
1. Create a DataFrame named `df` with appropriate columns for this visualization
2. Generate realistic, plausible data (not random noise)
3. Use numpy and pandas for data generation
4. Include 50-200 rows of data (appropriate for the visualization type)
5. Column names should be descriptive and match what the visualization would expect
6. For financial data: use realistic price ranges, volumes, percentages
7. For time series: generate a proper date range
8. For categorical: include realistic category names

Return ONLY Python code wrapped in ```python``` markers.
The code must:
- Import numpy as np and pandas as pd
- Define a `df` variable containing the DataFrame
- NOT include any visualization code, only data generation
"""


def generate_sample_data(description: str, api_key: str | None = None) -> pd.DataFrame:
    """
    Generate synthetic data using Claude Agent SDK (keyless mode).

    Falls back to keyword-based generation if SDK fails.
    """
    try:
        return asyncio.run(_generate_sample_data_async(description))
    except Exception as e:
        print(f"AI data generation failed: {e}, falling back to keyword-based")
        return _generate_sample_data_keywords(description)


async def _generate_sample_data_async(description: str) -> pd.DataFrame:
    """Async implementation using Claude Agent SDK."""
    options = ClaudeAgentOptions(
        system_prompt="You are a data generation assistant. Generate Python code that creates realistic sample data.",
        max_turns=1,
        model="haiku",  # Use Haiku for speed/cost
        cwd=str(Path(__file__).parent.parent),
    )

    raw_response = ""
    async for message in query(prompt=DATA_GEN_PROMPT.format(description=description), options=options):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    raw_response += block.text

    code = _extract_code(raw_response)

    # Execute the code to get the DataFrame
    exec_globals = {"np": np, "pd": pd, "datetime": datetime, "timedelta": timedelta}
    exec(code, exec_globals)

    if "df" in exec_globals and isinstance(exec_globals["df"], pd.DataFrame):
        return exec_globals["df"]
    else:
        raise ValueError("Generated code did not create a 'df' DataFrame")


def _extract_code(response_text: str) -> str:
    """Extract Python code from response."""
    import re
    pattern = r"```python\s*(.*?)\s*```"
    matches = re.findall(pattern, response_text, re.DOTALL)
    if matches:
        return matches[0].strip()
    return response_text


def _generate_sample_data_keywords(description: str) -> pd.DataFrame:
    """
    Fallback: Generate synthetic data based on description keywords.
    """
    desc_lower = description.lower()

    # Detect data type patterns
    if _is_time_series(desc_lower):
        return _generate_time_series(desc_lower)
    elif _is_heatmap(desc_lower):
        return _generate_heatmap_data(desc_lower)
    elif _is_distribution(desc_lower):
        return _generate_distribution_data(desc_lower)
    elif _is_categorical(desc_lower):
        return _generate_categorical_data(desc_lower)
    elif _is_scatter(desc_lower):
        return _generate_scatter_data(desc_lower)
    else:
        # Default to time series
        return _generate_time_series(desc_lower)


def _is_time_series(desc: str) -> bool:
    """Check if description suggests time series data."""
    keywords = [
        "over time", "time series", "trend", "daily", "monthly", "weekly",
        "historical", "past", "last", "price", "volume", "returns",
        "funding rate", "open interest", "liquidation", "stacked area",
        "moving average", "ma ", "ema", "sma", "bollinger", "rsi",
        "ohlc", "candlestick", "evolution"
    ]
    return any(kw in desc for kw in keywords)


def _is_heatmap(desc: str) -> bool:
    """Check if description suggests heatmap data."""
    keywords = ["heatmap", "correlation", "matrix", "hourly", "day of week"]
    return any(kw in desc for kw in keywords)


def _is_distribution(desc: str) -> bool:
    """Check if description suggests distribution data."""
    keywords = [
        "distribution", "histogram", "box plot", "violin",
        "spread", "density", "frequency"
    ]
    return any(kw in desc for kw in keywords)


def _is_categorical(desc: str) -> bool:
    """Check if description suggests categorical comparison."""
    keywords = [
        "rank", "top", "compare", "bar", "horizontal", "market cap",
        "breakdown", "share", "donut", "pie", "composition", "allocation"
    ]
    return any(kw in desc for kw in keywords)


def _is_scatter(desc: str) -> bool:
    """Check if description suggests scatter/relationship data."""
    keywords = ["scatter", "bubble", "relationship", "vs", "versus", "correlation"]
    return any(kw in desc for kw in keywords)


def _generate_time_series(desc: str) -> pd.DataFrame:
    """Generate time series data."""
    np.random.seed(42)

    # Determine time range - check plural forms first
    if "years" in desc:
        days = 730  # 2 years
    elif "year" in desc:
        days = 365
    elif "months" in desc:
        days = 180  # ~6 months
    elif "month" in desc:
        days = 30  # single month
    elif "weeks" in desc:
        days = 60  # ~2 months
    elif "week" in desc:
        days = 7
    else:
        days = 180  # 6 months default

    end_date = datetime.now()
    dates = [end_date - timedelta(days=x) for x in range(days)]
    dates.reverse()

    # Detect multiple series
    tokens = _extract_tokens(desc)
    if len(tokens) > 1:
        # Multiple series comparison
        df_data = {"date": dates}
        base_prices = {"btc": 45000, "eth": 2500, "sol": 100, "avax": 35, "bnb": 300}

        for token in tokens:
            base = base_prices.get(token.lower(), 100)
            returns = np.random.normal(0.001, 0.03, days)
            prices = [base]
            for r in returns[1:]:
                prices.append(prices[-1] * (1 + r))
            df_data[f"{token.upper()}_price"] = prices

        return pd.DataFrame(df_data)

    # Single series with optional volume/secondary
    base_price = 45000 if "btc" in desc or "bitcoin" in desc else 2500
    returns = np.random.normal(0.0005, 0.025, days)
    prices = [base_price]
    for r in returns[1:]:
        prices.append(prices[-1] * (1 + r))

    df = pd.DataFrame({
        "date": dates,
        "price": prices,
    })

    # Add volume if mentioned
    if "volume" in desc:
        df["volume"] = np.random.uniform(1e9, 5e9, days)

    # Add open interest if mentioned
    if "open interest" in desc:
        df["open_interest"] = np.random.uniform(5e9, 15e9, days)

    # Add funding rate if mentioned
    if "funding" in desc:
        df["funding_rate"] = np.random.normal(0.0001, 0.0005, days)

    # Add liquidations if mentioned
    if "liquidation" in desc:
        df["long_liquidations"] = np.abs(np.random.normal(50e6, 30e6, days))
        df["short_liquidations"] = np.abs(np.random.normal(50e6, 30e6, days))

    # Add OHLC if mentioned
    if "ohlc" in desc or "candlestick" in desc:
        df["open"] = df["price"] * np.random.uniform(0.98, 1.02, days)
        df["high"] = df[["price", "open"]].max(axis=1) * np.random.uniform(1.0, 1.03, days)
        df["low"] = df[["price", "open"]].min(axis=1) * np.random.uniform(0.97, 1.0, days)
        df["close"] = df["price"]

    return df


def _extract_tokens(desc: str) -> list[str]:
    """Extract cryptocurrency token names from description."""
    known_tokens = [
        "btc", "bitcoin", "eth", "ethereum", "sol", "solana",
        "avax", "avalanche", "bnb", "xrp", "ada", "dot", "link",
        "matic", "atom", "near", "arb", "op"
    ]

    found = []
    for token in known_tokens:
        if token in desc.lower():
            # Normalize to standard symbol
            normalized = {
                "bitcoin": "btc", "ethereum": "eth", "solana": "sol",
                "avalanche": "avax"
            }.get(token, token)
            if normalized not in found:
                found.append(normalized)

    return found if found else ["btc"]


def _generate_heatmap_data(desc: str) -> pd.DataFrame:
    """Generate heatmap/matrix data."""
    np.random.seed(42)

    if "correlation" in desc:
        # Correlation matrix
        tokens = ["BTC", "ETH", "SOL", "AVAX", "BNB", "XRP", "ADA", "DOT", "LINK", "MATIC"]
        n = len(tokens)
        # Generate realistic correlation matrix
        corr = np.eye(n)
        for i in range(n):
            for j in range(i + 1, n):
                c = np.random.uniform(0.3, 0.9)
                corr[i, j] = c
                corr[j, i] = c

        return pd.DataFrame(corr, index=tokens, columns=tokens)

    elif "hourly" in desc or "day of week" in desc:
        # Hour x Day of week heatmap
        days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        hours = list(range(24))

        data = []
        for day in days:
            for hour in hours:
                # Generate some pattern
                value = np.random.normal(0, 0.02) + (0.01 if 9 <= hour <= 17 else -0.005)
                data.append({"day": day, "hour": hour, "returns": value})

        return pd.DataFrame(data)

    else:
        # Generic heatmap - assets x metrics
        assets = ["BTC", "ETH", "SOL", "AVAX", "BNB"]
        metrics = ["1h", "4h", "1d", "7d", "30d"]

        data = []
        for asset in assets:
            row = {"asset": asset}
            for metric in metrics:
                row[metric] = np.random.normal(0, 0.1)
            data.append(row)

        return pd.DataFrame(data)


def _generate_distribution_data(desc: str) -> pd.DataFrame:
    """Generate distribution data."""
    np.random.seed(42)

    if "across" in desc or "multiple" in desc or "compare" in desc:
        # Multiple distributions
        tokens = ["BTC", "ETH", "SOL", "AVAX", "BNB"]
        data = []
        for token in tokens:
            returns = np.random.normal(0, 0.03, 365)
            for r in returns:
                data.append({"token": token, "return": r})
        return pd.DataFrame(data)

    else:
        # Single distribution
        returns = np.random.normal(0.001, 0.025, 1000)
        return pd.DataFrame({"daily_return": returns})


def _generate_categorical_data(desc: str) -> pd.DataFrame:
    """Generate categorical/ranking data."""
    np.random.seed(42)

    if "exchange" in desc:
        exchanges = ["Binance", "Coinbase", "Bybit", "OKX", "Kraken", "KuCoin", "Bitfinex", "Gate.io"]
        volumes = np.random.uniform(1e9, 20e9, len(exchanges))
        volumes = sorted(volumes, reverse=True)
        return pd.DataFrame({
            "exchange": exchanges[:len(volumes)],
            "volume_24h": volumes
        })

    elif "market cap" in desc or "token" in desc or "top" in desc:
        tokens = ["BTC", "ETH", "BNB", "SOL", "XRP", "ADA", "AVAX", "DOT", "LINK", "MATIC",
                  "ATOM", "UNI", "LTC", "BCH", "NEAR"]
        market_caps = [800e9, 300e9, 80e9, 60e9, 50e9, 40e9, 35e9, 30e9, 25e9, 20e9,
                       15e9, 12e9, 10e9, 8e9, 6e9]
        n = min(15, len(tokens))
        return pd.DataFrame({
            "token": tokens[:n],
            "market_cap": market_caps[:n],
            "price_change_24h": np.random.uniform(-0.1, 0.1, n),
            "volume_24h": np.random.uniform(1e8, 5e9, n)
        })

    elif "allocation" in desc or "portfolio" in desc:
        categories = ["Bitcoin", "Ethereum", "DeFi", "L1s", "Stablecoins", "Other"]
        allocations = [40, 25, 15, 10, 5, 5]
        return pd.DataFrame({
            "category": categories,
            "allocation": allocations
        })

    elif "stablecoin" in desc:
        stables = ["USDT", "USDC", "DAI", "BUSD", "TUSD", "FRAX"]
        shares = [65, 25, 5, 3, 1, 1]
        return pd.DataFrame({
            "stablecoin": stables,
            "market_share": shares
        })

    else:
        # Generic categories
        categories = ["Category A", "Category B", "Category C", "Category D", "Category E"]
        values = np.random.uniform(100, 1000, len(categories))
        return pd.DataFrame({
            "category": categories,
            "value": sorted(values, reverse=True)
        })


def _generate_scatter_data(desc: str) -> pd.DataFrame:
    """Generate scatter/relationship data."""
    np.random.seed(42)

    n_points = 50

    # Generate correlated data
    x = np.random.uniform(1e6, 1e10, n_points)  # Volume
    y = 0.00001 * np.log(x) + np.random.normal(0, 0.02, n_points)  # Returns

    df = pd.DataFrame({
        "volume": x,
        "price_change": y,
    })

    # Add size for bubble chart
    if "bubble" in desc or "market cap" in desc or "size" in desc:
        df["market_cap"] = np.random.uniform(1e8, 100e9, n_points)

    # Add labels
    tokens = ["BTC", "ETH", "SOL", "AVAX", "BNB", "XRP", "ADA", "DOT", "LINK", "MATIC"] * 5
    df["token"] = tokens[:n_points]

    return df
