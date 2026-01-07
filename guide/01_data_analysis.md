# 01 - Data Analysis

> Before selecting a chart, understand your data.
> This section provides a systematic approach to analyzing datasets for visualization.

---

## Why Data Analysis Matters

Visualization decisions depend on understanding:
- **What type of data** you have (temporal, categorical, numerical, hierarchical)
- **How many variables** are involved (univariate, bivariate, multivariate)
- **Data characteristics** (volume, density, outliers, missing values)
- **Aggregation needs** (raw data vs. summarized)

An AI agent should perform this analysis before any chart selection.

---

## Data Types

### Type Classification Reference

| Data Type | Subtype | Examples | Typical Charts |
|-----------|---------|----------|----------------|
| **Temporal** | Datetime | `2025-01-15 14:30:00` | Line, Area |
| | Date | `2025-01-15` | Line, Bar |
| | Duration | `2h 30m`, `45 days` | Bar, Histogram |
| **Categorical** | Nominal | Exchange names, coin symbols | Bar, Donut |
| | Ordinal | Risk levels (Low/Med/High), rankings | Bar, Heatmap |
| **Numerical** | Continuous | Price, market cap, volume | Line, Scatter |
| | Discrete | Trade count, block number | Bar, Histogram |
| | Ratio | Returns, growth rates | Line, Bar |
| | Percentage | Market share, allocation | Donut, Stacked Area |
| **Hierarchical** | Parent-child | Sector > Asset > Exchange | Treemap, Sunburst |

---

### Temporal Data

Time-indexed data is the most common type in financial dashboards.

#### Subtypes

| Subtype | Description | Format Examples |
|---------|-------------|-----------------|
| **Datetime** | Full timestamp with time component | `2025-01-15 14:30:00`, ISO 8601 |
| **Date** | Date only, no time component | `2025-01-15`, `Jan 15, 2025` |
| **Duration** | Time elapsed or intervals | `2h 30m`, `45 days`, `timedelta` |
| **Period** | Discrete time buckets | `2025-Q1`, `2025-W03`, `January 2025` |

#### Detection Rules

```python
import pandas as pd
import numpy as np

def identify_temporal_column(series: pd.Series) -> dict:
    """
    Identify if a column contains temporal data and its subtype.

    Returns:
        dict with keys: is_temporal, subtype, format_detected
    """
    result = {'is_temporal': False, 'subtype': None, 'format_detected': None}

    # Check if already datetime
    if pd.api.types.is_datetime64_any_dtype(series):
        result['is_temporal'] = True
        # Check if time component is meaningful
        if series.dt.time.nunique() > 1:
            result['subtype'] = 'datetime'
        else:
            result['subtype'] = 'date'
        return result

    # Check if timedelta
    if pd.api.types.is_timedelta64_dtype(series):
        result['is_temporal'] = True
        result['subtype'] = 'duration'
        return result

    # Try parsing string columns
    if series.dtype == 'object':
        sample = series.dropna().head(100)
        try:
            parsed = pd.to_datetime(sample, infer_datetime_format=True)
            result['is_temporal'] = True
            if parsed.dt.time.nunique() > 1:
                result['subtype'] = 'datetime'
            else:
                result['subtype'] = 'date'
            return result
        except (ValueError, TypeError):
            pass

    return result
```

#### Time Series Characteristics

| Characteristic | Question | Impact on Visualization |
|----------------|----------|------------------------|
| **Frequency** | Hourly? Daily? Monthly? | Determines tick format, aggregation |
| **Regularity** | Are intervals consistent? | Irregular data may need resampling |
| **Span** | Days? Months? Years? | Affects axis scaling, tick density |
| **Gaps** | Are there missing periods? | May need explicit handling |

```python
def analyze_time_series(df: pd.DataFrame, date_col: str) -> dict:
    """
    Analyze characteristics of a time series.

    Returns:
        dict with frequency, span, regularity, gaps info
    """
    dates = pd.to_datetime(df[date_col])
    dates_sorted = dates.sort_values()

    # Calculate time span
    span = dates_sorted.max() - dates_sorted.min()
    span_days = span.days

    # Infer frequency
    if len(dates_sorted) > 1:
        diffs = dates_sorted.diff().dropna()
        median_diff = diffs.median()

        if median_diff < pd.Timedelta(hours=1):
            frequency = 'minute'
        elif median_diff < pd.Timedelta(days=1):
            frequency = 'hourly'
        elif median_diff < pd.Timedelta(days=7):
            frequency = 'daily'
        elif median_diff < pd.Timedelta(days=32):
            frequency = 'weekly'
        else:
            frequency = 'monthly'
    else:
        frequency = 'unknown'

    # Check regularity (coefficient of variation of intervals)
    if len(diffs) > 1:
        cv = diffs.std() / diffs.mean() if diffs.mean() > pd.Timedelta(0) else 0
        is_regular = cv < 0.1  # Less than 10% variation
    else:
        is_regular = True

    return {
        'span_days': span_days,
        'frequency': frequency,
        'is_regular': is_regular,
        'num_points': len(dates),
        'start_date': dates_sorted.min(),
        'end_date': dates_sorted.max()
    }
```

---

### Categorical Data

Discrete groups or labels, not numerical values.

#### Subtypes

| Subtype | Description | Examples | Order Matters? |
|---------|-------------|----------|----------------|
| **Nominal** | No inherent order | Exchange names, coin symbols, regions | No |
| **Ordinal** | Has meaningful order | Risk levels, rankings, ratings | Yes |
| **Binary** | Two categories only | Long/Short, Buy/Sell, Active/Inactive | Sometimes |

#### Detection Rules

```python
def identify_categorical_column(series: pd.Series, max_unique_ratio: float = 0.05) -> dict:
    """
    Identify if a column is categorical.

    Args:
        series: Column to analyze
        max_unique_ratio: Max ratio of unique values to total (default 5%)

    Returns:
        dict with is_categorical, subtype, categories
    """
    result = {'is_categorical': False, 'subtype': None, 'categories': None}

    # Already categorical type
    if pd.api.types.is_categorical_dtype(series):
        result['is_categorical'] = True
        result['subtype'] = 'ordinal' if series.cat.ordered else 'nominal'
        result['categories'] = list(series.cat.categories)
        return result

    # Boolean is binary categorical
    if pd.api.types.is_bool_dtype(series):
        result['is_categorical'] = True
        result['subtype'] = 'binary'
        result['categories'] = [True, False]
        return result

    # String columns with few unique values
    if series.dtype == 'object':
        n_unique = series.nunique()
        n_total = len(series)

        # Check if it's categorical (few unique values relative to total)
        if n_unique <= 50 or (n_total > 100 and n_unique / n_total <= max_unique_ratio):
            result['is_categorical'] = True
            result['categories'] = list(series.unique())

            # Check for ordinal patterns
            ordinal_patterns = [
                ['low', 'medium', 'high'],
                ['low', 'med', 'high'],
                ['small', 'medium', 'large'],
                ['poor', 'fair', 'good', 'excellent'],
                ['1', '2', '3', '4', '5'],
            ]
            lower_cats = [str(c).lower() for c in result['categories']]
            for pattern in ordinal_patterns:
                if set(lower_cats) == set(pattern):
                    result['subtype'] = 'ordinal'
                    return result

            result['subtype'] = 'binary' if n_unique == 2 else 'nominal'
            return result

    # Integer columns with few unique values might be categorical
    if pd.api.types.is_integer_dtype(series):
        n_unique = series.nunique()
        if n_unique <= 10:
            result['is_categorical'] = True
            result['subtype'] = 'ordinal'  # Integer categories usually have order
            result['categories'] = sorted(series.unique())
            return result

    return result
```

#### Categorical Data Considerations

| Consideration | Question | Action |
|---------------|----------|--------|
| **Cardinality** | How many unique categories? | >15: aggregate to "Other" |
| **Label length** | Are labels long? | Use horizontal bars |
| **Order** | Does order matter? | Sort by value or preserve order |
| **Hierarchy** | Are categories nested? | Consider treemap/sunburst |

---

### Numerical Data

Quantitative values that can be measured or counted.

#### Subtypes

| Subtype | Description | Examples | Zero Meaningful? |
|---------|-------------|----------|------------------|
| **Continuous** | Any value in range | Price, temperature, percentage | Context-dependent |
| **Discrete** | Whole numbers only | Trade count, blocks, users | Usually yes |
| **Ratio** | Relative change | Returns, growth rates | No (ratio to baseline) |
| **Percentage** | Parts of whole (0-100%) | Market share, allocation | Yes |
| **Index** | Normalized to baseline | Price index (100 = baseline) | No |

#### Detection Rules

```python
def identify_numerical_column(series: pd.Series) -> dict:
    """
    Identify numerical data type and subtype.

    Returns:
        dict with is_numerical, subtype, range_info
    """
    result = {
        'is_numerical': False,
        'subtype': None,
        'range_info': None
    }

    # Must be numeric type
    if not pd.api.types.is_numeric_dtype(series):
        return result

    result['is_numerical'] = True
    clean_series = series.dropna()

    if len(clean_series) == 0:
        return result

    min_val = clean_series.min()
    max_val = clean_series.max()

    result['range_info'] = {
        'min': min_val,
        'max': max_val,
        'mean': clean_series.mean(),
        'median': clean_series.median(),
        'std': clean_series.std()
    }

    # Check for percentage (0-1 or 0-100 range)
    if 0 <= min_val and max_val <= 1:
        # Could be percentage as decimal
        result['subtype'] = 'percentage_decimal'
    elif 0 <= min_val and max_val <= 100:
        # Could be percentage or just bounded numeric
        if clean_series.sum() > 0 and abs(clean_series.sum() - 100) < 1:
            result['subtype'] = 'percentage'
        else:
            result['subtype'] = 'continuous'
    # Check for discrete (integer values)
    elif pd.api.types.is_integer_dtype(series):
        result['subtype'] = 'discrete'
    elif (clean_series == clean_series.astype(int)).all():
        result['subtype'] = 'discrete'
    # Check for ratio-like (centered around 0 or 1)
    elif min_val < 0 and max_val > 0:
        if abs(clean_series.mean()) < clean_series.std() * 0.5:
            result['subtype'] = 'ratio'  # Centered around 0 (returns)
        else:
            result['subtype'] = 'continuous'
    else:
        result['subtype'] = 'continuous'

    return result
```

#### Scale Considerations

| Value Range | Recommended Scale | Example |
|-------------|-------------------|---------|
| Spans orders of magnitude | Logarithmic | Market cap ($1K to $1T) |
| Linear progression | Linear | Price over time |
| Centered around zero | Linear, diverging colors | Returns (-50% to +50%) |
| Always positive, skewed | Log or linear | Volume, count data |

---

### Hierarchical Data

Data with parent-child relationships or nested structure.

#### Detection

```python
def identify_hierarchical_structure(df: pd.DataFrame) -> dict:
    """
    Identify potential hierarchical relationships in columns.

    Returns:
        dict with is_hierarchical, hierarchy_columns, levels
    """
    result = {
        'is_hierarchical': False,
        'hierarchy_columns': [],
        'levels': 0
    }

    # Look for common hierarchical patterns in column names
    hierarchy_patterns = [
        ['sector', 'industry', 'company'],
        ['category', 'subcategory', 'item'],
        ['region', 'country', 'city'],
        ['parent', 'child'],
        ['level1', 'level2', 'level3'],
        ['type', 'subtype'],
    ]

    columns_lower = [c.lower() for c in df.columns]

    for pattern in hierarchy_patterns:
        matches = [p for p in pattern if any(p in col for col in columns_lower)]
        if len(matches) >= 2:
            result['is_hierarchical'] = True
            result['hierarchy_columns'] = matches
            result['levels'] = len(matches)
            return result

    # Check for nested cardinality (parent has fewer unique values than child)
    categorical_cols = [col for col in df.columns
                        if df[col].dtype == 'object' and df[col].nunique() < len(df) * 0.5]

    if len(categorical_cols) >= 2:
        # Sort by cardinality
        sorted_cols = sorted(categorical_cols, key=lambda c: df[c].nunique())

        # Check if lower cardinality columns determine higher ones
        for i in range(len(sorted_cols) - 1):
            parent_col = sorted_cols[i]
            child_col = sorted_cols[i + 1]

            # Each unique value in parent should map to consistent child values
            grouped = df.groupby(parent_col)[child_col].nunique()
            if (grouped > 1).any():
                # Parent doesn't uniquely determine child - potential hierarchy
                result['is_hierarchical'] = True
                result['hierarchy_columns'] = sorted_cols[:3]  # Top 3 levels
                result['levels'] = min(3, len(sorted_cols))
                return result

    return result
```

---

## Dimensionality

The number of variables involved determines chart complexity.

### Classification

| Dimensionality | Variables | Typical Charts | Complexity |
|----------------|-----------|----------------|------------|
| **Univariate** | 1 | Histogram, Box plot, KDE | Low |
| **Bivariate** | 2 | Scatter, Line, Bar | Medium |
| **Multivariate** | 3+ | Bubble, Heatmap, Small multiples | High |

### Univariate Analysis

Single variable examination - understanding distribution before relationships.

**Common visualizations:**
- **Histogram**: Distribution shape, bins
- **Box plot**: Quartiles, outliers
- **KDE (Kernel Density)**: Smooth distribution

```python
def univariate_summary(series: pd.Series) -> dict:
    """
    Generate univariate statistics for a numeric column.

    Returns:
        dict with distribution statistics and recommended visualizations
    """
    clean = series.dropna()

    stats = {
        'count': len(clean),
        'mean': clean.mean(),
        'median': clean.median(),
        'std': clean.std(),
        'min': clean.min(),
        'max': clean.max(),
        'q25': clean.quantile(0.25),
        'q75': clean.quantile(0.75),
        'skewness': clean.skew(),
        'kurtosis': clean.kurtosis()
    }

    # Determine distribution shape
    if abs(stats['skewness']) < 0.5:
        stats['shape'] = 'symmetric'
    elif stats['skewness'] > 0.5:
        stats['shape'] = 'right_skewed'
    else:
        stats['shape'] = 'left_skewed'

    # Recommend visualizations
    stats['recommended_charts'] = ['histogram']
    if stats['count'] > 20:
        stats['recommended_charts'].append('box_plot')
    if stats['count'] > 50:
        stats['recommended_charts'].append('kde')

    return stats
```

### Bivariate Analysis

Two-variable relationships.

| Variable Types | Relationship | Chart Type |
|---------------|--------------|------------|
| Numeric + Numeric | Correlation | Scatter plot |
| Numeric + Time | Trend | Line chart |
| Numeric + Categorical | Comparison | Bar chart |
| Categorical + Categorical | Association | Heatmap, Grouped bar |

```python
def bivariate_recommendation(col1_type: str, col2_type: str) -> str:
    """
    Recommend chart type for two-variable visualization.

    Args:
        col1_type: 'numeric', 'categorical', or 'temporal'
        col2_type: 'numeric', 'categorical', or 'temporal'

    Returns:
        Recommended chart type
    """
    pair = frozenset([col1_type, col2_type])

    recommendations = {
        frozenset(['numeric', 'numeric']): 'scatter',
        frozenset(['numeric', 'temporal']): 'line',
        frozenset(['numeric', 'categorical']): 'bar',
        frozenset(['categorical', 'categorical']): 'heatmap',
        frozenset(['temporal', 'categorical']): 'grouped_line',
    }

    return recommendations.get(pair, 'scatter')
```

### Multivariate Analysis

Three or more variables require encoding strategies.

| Encoding Channel | Best For | Limitations |
|-----------------|----------|-------------|
| X position | Primary numeric/temporal | One variable |
| Y position | Primary numeric | One variable |
| Color (hue) | Categorical distinction | Max 7-10 categories |
| Color (intensity) | Numeric magnitude | Hard to read precisely |
| Size | Numeric magnitude | Area perception is imprecise |
| Shape | Categorical | Max 5-6 shapes |
| Faceting | Categorical comparison | Max 12 subplots |

```python
def multivariate_encoding_plan(variables: list[dict]) -> dict:
    """
    Create encoding plan for multiple variables.

    Args:
        variables: List of dicts with 'name', 'type', 'importance'

    Returns:
        dict mapping variables to encoding channels
    """
    # Sort by importance
    sorted_vars = sorted(variables, key=lambda x: x.get('importance', 0), reverse=True)

    plan = {}
    available_channels = ['x', 'y', 'color', 'size', 'facet_row', 'facet_col']

    for var in sorted_vars:
        if not available_channels:
            plan[var['name']] = 'aggregate_or_filter'
            continue

        var_type = var['type']

        # Assign channel based on type
        if var_type == 'temporal' and 'x' in available_channels:
            plan[var['name']] = 'x'
            available_channels.remove('x')
        elif var_type == 'numeric':
            if 'y' in available_channels:
                plan[var['name']] = 'y'
                available_channels.remove('y')
            elif 'x' in available_channels:
                plan[var['name']] = 'x'
                available_channels.remove('x')
            elif 'size' in available_channels:
                plan[var['name']] = 'size'
                available_channels.remove('size')
            elif 'color' in available_channels:
                plan[var['name']] = 'color_intensity'
                available_channels.remove('color')
        elif var_type == 'categorical':
            if 'color' in available_channels:
                plan[var['name']] = 'color'
                available_channels.remove('color')
            elif 'facet_col' in available_channels:
                plan[var['name']] = 'facet_col'
                available_channels.remove('facet_col')
            elif 'facet_row' in available_channels:
                plan[var['name']] = 'facet_row'
                available_channels.remove('facet_row')

    return plan
```

---

## Data Characteristics

### Volume (Data Point Count)

| Volume | Classification | Visualization Impact |
|--------|---------------|---------------------|
| < 100 | Small | Show all points, direct labels |
| 100 - 1,000 | Medium | Standard charts work well |
| 1,000 - 10,000 | Large | Consider aggregation, opacity |
| 10,000 - 100,000 | Very Large | Use WebGL, aggregation required |
| > 100,000 | Massive | Must aggregate or sample |

```python
def assess_volume(n_points: int) -> dict:
    """
    Assess data volume and recommend handling strategies.

    Returns:
        dict with classification, recommendations, and warnings
    """
    result = {
        'count': n_points,
        'classification': None,
        'recommendations': [],
        'warnings': []
    }

    if n_points < 100:
        result['classification'] = 'small'
        result['recommendations'] = [
            'Show all data points',
            'Consider direct labeling',
            'No aggregation needed'
        ]
    elif n_points < 1000:
        result['classification'] = 'medium'
        result['recommendations'] = [
            'Standard plotting works well',
            'Consider hover for details'
        ]
    elif n_points < 10000:
        result['classification'] = 'large'
        result['recommendations'] = [
            'Use opacity (0.3-0.7) for scatter plots',
            'Consider aggregation for bar charts',
            'Use scattergl for better performance'
        ]
        result['warnings'] = ['May be slow in browser without optimization']
    elif n_points < 100000:
        result['classification'] = 'very_large'
        result['recommendations'] = [
            'Aggregation strongly recommended',
            'Use WebGL traces (scattergl, etc.)',
            'Consider downsampling for initial view'
        ]
        result['warnings'] = ['Performance will suffer without aggregation']
    else:
        result['classification'] = 'massive'
        result['recommendations'] = [
            'Aggregation required',
            'Consider server-side processing',
            'Use binned representations (heatmap, hexbin)'
        ]
        result['warnings'] = ['Cannot render raw data - must aggregate']

    return result
```

### Density

How spread out or clustered the data is.

| Density | Description | Visualization Strategy |
|---------|-------------|----------------------|
| **Sparse** | Few points, wide spacing | Show all points, markers visible |
| **Moderate** | Some overlap | Reduce opacity, jitter if needed |
| **Dense** | Significant overlap | Heatmap, binning, aggregation |
| **Overplotted** | Points completely obscure each other | 2D histogram, contour, hexbin |

```python
def assess_density_2d(x: pd.Series, y: pd.Series, n_bins: int = 50) -> dict:
    """
    Assess 2D density of scatter data.

    Returns:
        dict with density classification and recommendations
    """
    import numpy as np

    # Create 2D histogram to assess density
    counts, _, _ = np.histogram2d(x.dropna(), y.dropna(), bins=n_bins)

    # Calculate density metrics
    non_empty_bins = (counts > 0).sum()
    total_bins = n_bins * n_bins
    fill_ratio = non_empty_bins / total_bins

    max_per_bin = counts.max()
    mean_per_occupied = counts[counts > 0].mean() if non_empty_bins > 0 else 0

    result = {
        'fill_ratio': fill_ratio,
        'max_per_bin': max_per_bin,
        'mean_per_occupied_bin': mean_per_occupied
    }

    # Classify density
    if fill_ratio < 0.1:
        result['classification'] = 'sparse'
        result['recommendations'] = [
            'Standard scatter plot',
            'Full opacity markers'
        ]
    elif fill_ratio < 0.3 and max_per_bin < 20:
        result['classification'] = 'moderate'
        result['recommendations'] = [
            'Scatter with reduced opacity (0.5-0.7)',
            'Consider adding jitter if discrete values'
        ]
    elif max_per_bin < 100:
        result['classification'] = 'dense'
        result['recommendations'] = [
            'Use low opacity (0.2-0.4)',
            'Consider 2D histogram overlay',
            'Use smaller marker size'
        ]
    else:
        result['classification'] = 'overplotted'
        result['recommendations'] = [
            'Use heatmap or 2D histogram',
            'Consider contour plot',
            'Hexbin if showing distribution'
        ]

    return result
```

### Outliers

Extreme values that may need special handling.

#### Detection Methods

| Method | Description | Best For |
|--------|-------------|----------|
| **IQR** | 1.5x interquartile range | General purpose |
| **Z-score** | Standard deviations from mean | Normal distributions |
| **Modified Z-score** | Uses median, robust to outliers | Skewed data |
| **Percentile** | Beyond 1st/99th percentile | Any distribution |

```python
def detect_outliers(series: pd.Series, method: str = 'iqr') -> dict:
    """
    Detect outliers using specified method.

    Args:
        series: Numeric series to analyze
        method: 'iqr', 'zscore', 'modified_zscore', or 'percentile'

    Returns:
        dict with outlier indices, bounds, and count
    """
    import numpy as np

    clean = series.dropna()
    result = {
        'method': method,
        'lower_bound': None,
        'upper_bound': None,
        'outlier_indices': [],
        'outlier_count': 0,
        'outlier_pct': 0
    }

    if method == 'iqr':
        q1 = clean.quantile(0.25)
        q3 = clean.quantile(0.75)
        iqr = q3 - q1
        result['lower_bound'] = q1 - 1.5 * iqr
        result['upper_bound'] = q3 + 1.5 * iqr

    elif method == 'zscore':
        mean = clean.mean()
        std = clean.std()
        result['lower_bound'] = mean - 3 * std
        result['upper_bound'] = mean + 3 * std

    elif method == 'modified_zscore':
        median = clean.median()
        mad = np.median(np.abs(clean - median))
        if mad == 0:
            mad = clean.std()
        modified_z = 0.6745 * (clean - median) / mad
        result['outlier_indices'] = list(series.index[abs(modified_z) > 3.5])
        result['outlier_count'] = len(result['outlier_indices'])
        result['outlier_pct'] = result['outlier_count'] / len(series) * 100
        return result

    elif method == 'percentile':
        result['lower_bound'] = clean.quantile(0.01)
        result['upper_bound'] = clean.quantile(0.99)

    # Find outliers based on bounds
    mask = (series < result['lower_bound']) | (series > result['upper_bound'])
    result['outlier_indices'] = list(series.index[mask])
    result['outlier_count'] = mask.sum()
    result['outlier_pct'] = result['outlier_count'] / len(series) * 100

    return result
```

#### Outlier Visualization Strategies

| Strategy | When to Use | Implementation |
|----------|-------------|----------------|
| **Include** | Outliers are meaningful | Extend axis range |
| **Clip** | Outliers distort view | Set axis range limits |
| **Highlight** | Draw attention to outliers | Different color/marker |
| **Separate** | Analyze outliers independently | Small multiples or annotation |
| **Transform** | Reduce outlier impact | Log scale |

```python
def outlier_visualization_strategy(outlier_info: dict, context: str = 'exploratory') -> dict:
    """
    Recommend visualization strategy for handling outliers.

    Args:
        outlier_info: Output from detect_outliers()
        context: 'exploratory' or 'explanatory'

    Returns:
        dict with strategy recommendation and implementation
    """
    pct = outlier_info['outlier_pct']

    if pct < 1:
        return {
            'strategy': 'include',
            'reason': 'Very few outliers, include for completeness',
            'implementation': 'No special handling needed'
        }
    elif pct < 5:
        if context == 'exploratory':
            return {
                'strategy': 'highlight',
                'reason': 'Some outliers worth investigating',
                'implementation': 'Use different color/marker for outliers'
            }
        else:
            return {
                'strategy': 'clip',
                'reason': 'Focus on main distribution for clarity',
                'implementation': f"Set axis range to [{outlier_info['lower_bound']:.2g}, {outlier_info['upper_bound']:.2g}]"
            }
    else:
        return {
            'strategy': 'transform',
            'reason': 'Many outliers suggest need for scale transformation',
            'implementation': 'Consider log scale or separate analysis'
        }
```

### Missing Values

Gaps in data that require explicit handling.

#### Types of Missingness

| Type | Pattern | Handling |
|------|---------|----------|
| **Random** | No pattern | Can often interpolate |
| **Time gaps** | Missing periods in time series | Show gaps or interpolate |
| **Categorical** | Missing categories | Include or note as N/A |
| **Systematic** | Consistent pattern | Investigate root cause |

```python
def analyze_missing_values(df: pd.DataFrame) -> dict:
    """
    Analyze missing value patterns in a DataFrame.

    Returns:
        dict with missing value summary and recommendations
    """
    result = {
        'total_missing': df.isnull().sum().sum(),
        'total_cells': df.size,
        'missing_pct': df.isnull().sum().sum() / df.size * 100,
        'by_column': {},
        'recommendations': []
    }

    for col in df.columns:
        missing_count = df[col].isnull().sum()
        missing_pct = missing_count / len(df) * 100

        result['by_column'][col] = {
            'count': missing_count,
            'pct': missing_pct
        }

        # Check for patterns in missing data (for time series)
        if pd.api.types.is_datetime64_any_dtype(df.index):
            missing_idx = df[df[col].isnull()].index
            if len(missing_idx) > 1:
                # Check if missing values are consecutive
                diffs = pd.Series(missing_idx).diff()
                if diffs.nunique() == 1:
                    result['by_column'][col]['pattern'] = 'consecutive_gaps'
                else:
                    result['by_column'][col]['pattern'] = 'scattered'

    # Generate recommendations
    if result['missing_pct'] < 1:
        result['recommendations'].append('Minimal missing data - safe to proceed')
    elif result['missing_pct'] < 5:
        result['recommendations'].append('Consider interpolation for numeric columns')
        result['recommendations'].append('Drop rows only if analysis requires completeness')
    elif result['missing_pct'] < 20:
        result['recommendations'].append('Investigate cause of missing data')
        result['recommendations'].append('Consider imputation strategies')
        result['recommendations'].append('Note missing data in visualization annotations')
    else:
        result['recommendations'].append('High missing rate - data quality issue')
        result['recommendations'].append('May need to exclude columns with >50% missing')
        result['recommendations'].append('Clearly communicate data limitations')

    return result
```

#### Time Series Gap Handling

```python
def handle_time_series_gaps(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    strategy: str = 'show_gap'
) -> pd.DataFrame:
    """
    Handle gaps in time series data.

    Args:
        df: DataFrame with time series
        date_col: Name of date column
        value_col: Name of value column
        strategy: 'show_gap', 'interpolate', 'fill_zero', 'fill_forward'

    Returns:
        DataFrame with gaps handled according to strategy
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)

    if strategy == 'show_gap':
        # Insert None values where gaps exist (Plotly will show breaks)
        # Identify gaps larger than typical interval
        diffs = df[date_col].diff()
        median_diff = diffs.median()
        large_gaps = diffs > median_diff * 2

        # Insert None row before each large gap
        gap_rows = []
        for idx in df[large_gaps].index:
            prev_idx = df.index.get_loc(idx) - 1
            if prev_idx >= 0:
                gap_rows.append({
                    date_col: df.iloc[prev_idx][date_col] + median_diff,
                    value_col: None
                })

        if gap_rows:
            gap_df = pd.DataFrame(gap_rows)
            df = pd.concat([df, gap_df]).sort_values(date_col)

    elif strategy == 'interpolate':
        # Reindex to regular frequency and interpolate
        df = df.set_index(date_col)
        freq = pd.infer_freq(df.index)
        if freq:
            df = df.resample(freq).mean()
            df[value_col] = df[value_col].interpolate(method='linear')
        df = df.reset_index()

    elif strategy == 'fill_zero':
        df[value_col] = df[value_col].fillna(0)

    elif strategy == 'fill_forward':
        df[value_col] = df[value_col].ffill()

    return df
```

---

## Aggregation Requirements

### When to Aggregate

| Condition | Aggregate? | Method |
|-----------|------------|--------|
| Too many data points (>10K) | Yes | Time-based or categorical grouping |
| Multiple series, detailed data | Yes | Group by category |
| Need summary statistics | Yes | Mean, median, sum, count |
| Comparing periods | Yes | Period-based aggregation |
| Raw data tells the story | No | Show individual points |

### Decision Matrix

```
SHOULD_AGGREGATE:
├── Volume > 10,000 points
│   └── YES: Aggregate to reduce points
│
├── Chart type is bar/pie/donut
│   └── YES: Aggregate by category (sum, mean, count)
│
├── Comparing periods (this week vs last week)
│   └── YES: Aggregate to matching periods
│
├── Intent is "total" or "average"
│   └── YES: Aggregate with appropriate function
│
├── Time series with intraday data, showing months
│   └── YES: Aggregate to daily/weekly
│
└── Scatter plot showing relationship
    └── MAYBE: Only if overplotted
```

### Common Aggregation Functions

| Function | Use Case | Plotly/Pandas |
|----------|----------|---------------|
| **Sum** | Totals, volumes, counts | `df.groupby(...).sum()` |
| **Mean** | Averages, rates, ratios | `df.groupby(...).mean()` |
| **Median** | Robust central tendency | `df.groupby(...).median()` |
| **Count** | Frequency, occurrences | `df.groupby(...).count()` |
| **Min/Max** | Ranges, extremes | `df.groupby(...).agg(['min', 'max'])` |
| **First/Last** | Point-in-time values | `df.groupby(...).first()` |
| **Std** | Variability | `df.groupby(...).std()` |
| **Percentile** | Distribution characteristics | `df.groupby(...).quantile([0.25, 0.75])` |

```python
def recommend_aggregation(
    df: pd.DataFrame,
    value_col: str,
    intent: str
) -> dict:
    """
    Recommend aggregation function based on intent.

    Args:
        df: DataFrame to analyze
        value_col: Column to aggregate
        intent: 'total', 'average', 'distribution', 'comparison', 'trend'

    Returns:
        dict with recommended aggregation function and reasoning
    """
    recommendations = {
        'total': {
            'function': 'sum',
            'reason': 'Sum shows total magnitude',
            'code': f"df.groupby(group_col)['{value_col}'].sum()"
        },
        'average': {
            'function': 'mean',
            'reason': 'Mean shows typical value',
            'code': f"df.groupby(group_col)['{value_col}'].mean()"
        },
        'distribution': {
            'function': 'describe',
            'reason': 'Need multiple statistics',
            'code': f"df.groupby(group_col)['{value_col}'].describe()"
        },
        'comparison': {
            'function': 'mean',
            'reason': 'Mean enables fair comparison across groups',
            'code': f"df.groupby(group_col)['{value_col}'].mean()"
        },
        'trend': {
            'function': 'mean',
            'reason': 'Mean smooths noise in trend',
            'code': f"df.resample(freq)['{value_col}'].mean()"
        }
    }

    return recommendations.get(intent, recommendations['average'])
```

### Time-Based Aggregation

| Data Frequency | Display Period | Aggregation |
|---------------|----------------|-------------|
| Per-minute | 1 hour | Resample to 1-minute (as-is or aggregate) |
| Per-minute | 1 day | Resample to hourly |
| Per-minute | 1 week | Resample to 4-hour or daily |
| Hourly | 1 month | Resample to daily |
| Hourly | 1 year | Resample to weekly |
| Daily | 1 year | As-is (365 points is fine) |
| Daily | 5 years | Resample to weekly or monthly |

```python
def auto_resample_frequency(
    span_days: int,
    current_freq: str,
    max_points: int = 500
) -> str:
    """
    Determine appropriate resampling frequency.

    Args:
        span_days: Time span in days
        current_freq: Current data frequency ('minute', 'hourly', 'daily', etc.)
        max_points: Maximum desired data points

    Returns:
        Recommended pandas resample frequency string
    """
    # Estimate current points
    points_per_day = {
        'minute': 1440,
        'hourly': 24,
        'daily': 1,
        'weekly': 1/7,
        'monthly': 1/30
    }

    current_ppd = points_per_day.get(current_freq, 24)
    estimated_points = span_days * current_ppd

    if estimated_points <= max_points:
        return None  # No resampling needed

    # Calculate target points per day
    target_ppd = max_points / span_days

    # Select appropriate frequency
    if target_ppd >= 24:
        return '1h'  # Hourly
    elif target_ppd >= 6:
        return '4h'  # 4-hourly
    elif target_ppd >= 1:
        return '1D'  # Daily
    elif target_ppd >= 1/7:
        return '1W'  # Weekly
    else:
        return '1M'  # Monthly


def resample_time_series(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    freq: str,
    agg_func: str = 'mean'
) -> pd.DataFrame:
    """
    Resample time series to specified frequency.

    Args:
        df: DataFrame with time series
        date_col: Name of date column
        value_col: Name of value column to aggregate
        freq: Pandas frequency string ('1h', '1D', '1W', etc.)
        agg_func: Aggregation function ('mean', 'sum', 'last', 'ohlc')

    Returns:
        Resampled DataFrame
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col)

    if agg_func == 'ohlc':
        resampled = df[value_col].resample(freq).ohlc()
    else:
        resampled = df[value_col].resample(freq).agg(agg_func)

    return resampled.reset_index()
```

---

## Complete Data Analysis Workflow

### AI Agent Analysis Procedure

When analyzing a dataset for visualization, follow this sequence:

```
1. LOAD DATA
   └── Read file/query results into DataFrame

2. IDENTIFY COLUMNS
   ├── For each column:
   │   ├── Identify data type (temporal, categorical, numerical, hierarchical)
   │   ├── Identify subtype (datetime/date, nominal/ordinal, continuous/discrete)
   │   └── Note special characteristics (percentage, currency, ratio)

3. ASSESS CHARACTERISTICS
   ├── Volume: How many rows?
   ├── Density: Sparse or dense (for scatter plots)?
   ├── Outliers: Present? Significant?
   └── Missing: Gaps? Nulls? Patterns?

4. DETERMINE DIMENSIONALITY
   ├── Primary variable (x-axis candidate)
   ├── Secondary variable (y-axis candidate)
   └── Additional variables (color, size, facet candidates)

5. DECIDE AGGREGATION
   ├── Is aggregation needed? (volume, intent)
   ├── What aggregation function? (sum, mean, count)
   └── What grouping? (time period, category)

6. OUTPUT ANALYSIS SUMMARY
   └── Structured report for chart selection
```

### Master Analysis Function

```python
def analyze_dataset_for_visualization(df: pd.DataFrame) -> dict:
    """
    Complete data analysis for visualization planning.

    Args:
        df: DataFrame to analyze

    Returns:
        Comprehensive analysis dictionary for chart selection
    """
    analysis = {
        'shape': {'rows': len(df), 'columns': len(df.columns)},
        'columns': {},
        'temporal_columns': [],
        'categorical_columns': [],
        'numerical_columns': [],
        'hierarchical': None,
        'volume_assessment': None,
        'missing_analysis': None,
        'recommendations': []
    }

    # Analyze each column
    for col in df.columns:
        col_analysis = {'name': col}

        # Check temporal
        temporal = identify_temporal_column(df[col])
        if temporal['is_temporal']:
            col_analysis['type'] = 'temporal'
            col_analysis['subtype'] = temporal['subtype']
            analysis['temporal_columns'].append(col)

            # Get time series characteristics
            ts_info = analyze_time_series(df, col)
            col_analysis['time_series_info'] = ts_info

        # Check categorical
        elif not pd.api.types.is_numeric_dtype(df[col]):
            categorical = identify_categorical_column(df[col])
            if categorical['is_categorical']:
                col_analysis['type'] = 'categorical'
                col_analysis['subtype'] = categorical['subtype']
                col_analysis['cardinality'] = df[col].nunique()
                analysis['categorical_columns'].append(col)

        # Check numerical
        if pd.api.types.is_numeric_dtype(df[col]):
            numerical = identify_numerical_column(df[col])
            if numerical['is_numerical']:
                col_analysis['type'] = 'numerical'
                col_analysis['subtype'] = numerical['subtype']
                col_analysis['range'] = numerical['range_info']
                analysis['numerical_columns'].append(col)

                # Check for outliers
                outliers = detect_outliers(df[col])
                if outliers['outlier_count'] > 0:
                    col_analysis['outliers'] = outliers

        analysis['columns'][col] = col_analysis

    # Check for hierarchical structure
    analysis['hierarchical'] = identify_hierarchical_structure(df)

    # Volume assessment
    analysis['volume_assessment'] = assess_volume(len(df))

    # Missing value analysis
    analysis['missing_analysis'] = analyze_missing_values(df)

    # Generate recommendations
    if len(analysis['temporal_columns']) > 0:
        analysis['recommendations'].append('Time series detected - consider line/area charts')

    if len(analysis['categorical_columns']) > 0:
        max_cardinality = max(df[c].nunique() for c in analysis['categorical_columns'])
        if max_cardinality > 15:
            analysis['recommendations'].append(f'High cardinality ({max_cardinality}) - consider aggregating to top N')

    if analysis['volume_assessment']['classification'] in ['large', 'very_large', 'massive']:
        analysis['recommendations'].append('Large dataset - aggregation or WebGL recommended')

    if analysis['missing_analysis']['missing_pct'] > 5:
        analysis['recommendations'].append('Significant missing data - address before visualization')

    if analysis['hierarchical']['is_hierarchical']:
        analysis['recommendations'].append('Hierarchical structure detected - consider treemap/sunburst')

    return analysis
```

### Example Output

```python
# Example analysis output
{
    'shape': {'rows': 10000, 'columns': 5},
    'columns': {
        'date': {
            'name': 'date',
            'type': 'temporal',
            'subtype': 'datetime',
            'time_series_info': {
                'span_days': 365,
                'frequency': 'daily',
                'is_regular': True,
                'num_points': 365
            }
        },
        'exchange': {
            'name': 'exchange',
            'type': 'categorical',
            'subtype': 'nominal',
            'cardinality': 12
        },
        'volume': {
            'name': 'volume',
            'type': 'numerical',
            'subtype': 'continuous',
            'range': {'min': 1000, 'max': 50000000, 'mean': 5000000},
            'outliers': {'outlier_count': 45, 'outlier_pct': 0.45}
        },
        'price': {
            'name': 'price',
            'type': 'numerical',
            'subtype': 'continuous',
            'range': {'min': 30000, 'max': 70000, 'mean': 45000}
        },
        'change_pct': {
            'name': 'change_pct',
            'type': 'numerical',
            'subtype': 'percentage_decimal',
            'range': {'min': -0.15, 'max': 0.12, 'mean': 0.001}
        }
    },
    'temporal_columns': ['date'],
    'categorical_columns': ['exchange'],
    'numerical_columns': ['volume', 'price', 'change_pct'],
    'hierarchical': {'is_hierarchical': False},
    'volume_assessment': {
        'count': 10000,
        'classification': 'large',
        'recommendations': ['Use opacity for scatter plots', 'Consider aggregation']
    },
    'missing_analysis': {
        'total_missing': 50,
        'missing_pct': 0.1,
        'recommendations': ['Minimal missing data - safe to proceed']
    },
    'recommendations': [
        'Time series detected - consider line/area charts',
        'Large dataset - aggregation or WebGL recommended'
    ]
}
```

---

## Quick Reference

### Data Type Decision Table

| If column contains... | Data Type | Subtype | Example Charts |
|----------------------|-----------|---------|----------------|
| Datetime/timestamps | Temporal | datetime | Line, Area |
| Dates only | Temporal | date | Line, Bar |
| Exchange/coin names | Categorical | nominal | Bar, Donut |
| Ratings (1-5 stars) | Categorical | ordinal | Bar, Heatmap |
| Prices, volumes | Numerical | continuous | Line, Scatter |
| Counts, integers | Numerical | discrete | Bar, Histogram |
| Returns, changes | Numerical | ratio | Line, Bar |
| Percentages 0-100 | Numerical | percentage | Donut, Stacked |
| Sector > Asset | Hierarchical | parent-child | Treemap |

### Volume Handling Cheat Sheet

| Row Count | Action | Chart Considerations |
|-----------|--------|---------------------|
| < 100 | Show all | Direct labels possible |
| 100-1K | Show all | Hover for details |
| 1K-10K | Consider aggregating | Use WebGL for scatter |
| 10K-100K | Must aggregate | Aggregation required |
| > 100K | Server-side processing | Heatmap or binned |

### Aggregation Function Selector

| Intent | Function | Code Pattern |
|--------|----------|--------------|
| Total value | `sum` | `df.groupby('cat').sum()` |
| Average value | `mean` | `df.groupby('cat').mean()` |
| How many | `count` | `df.groupby('cat').count()` |
| Latest value | `last` | `df.groupby('cat').last()` |
| Range | `['min', 'max']` | `df.groupby('cat').agg(['min', 'max'])` |
| Distribution | `describe` | `df.groupby('cat').describe()` |

---

## Analysis Checklist

Before chart selection, verify:

- [ ] All columns classified by data type
- [ ] Temporal columns: frequency and span identified
- [ ] Categorical columns: cardinality assessed
- [ ] Numerical columns: range and outliers checked
- [ ] Missing values: pattern and percentage known
- [ ] Volume: appropriate for chosen chart type
- [ ] Aggregation: decided if needed and which function
- [ ] Hierarchical structure: checked if present
