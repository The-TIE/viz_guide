# 14 - Performance

> Rules and strategies for optimizing Plotly visualizations with large datasets.
> Performance optimization prevents browser crashes, slow rendering, and poor user experience.

---

## Data Point Thresholds

### When to Consider Optimization

| Data Points | Performance Level | Action Required |
|-------------|-------------------|-----------------|
| < 1,000 | Excellent | No optimization needed |
| 1,000 - 10,000 | Good | Monitor render time |
| 10,000 - 50,000 | Moderate | Use WebGL traces |
| 50,000 - 100,000 | Degraded | WebGL + reduce trace count |
| 100,000 - 500,000 | Poor | Downsample or aggregate |
| 500,000+ | Critical | Aggressive downsampling required |

**Rule of thumb:** If total data points across all traces exceeds 50,000, optimization is mandatory.

### Calculating Total Data Points

```python
def calculate_total_points(fig):
    """
    Calculate total data points in a Plotly figure.

    Args:
        fig: Plotly figure object

    Returns:
        Total number of data points across all traces
    """
    total = 0
    for trace in fig.data:
        if hasattr(trace, 'x') and trace.x is not None:
            total += len(trace.x)
        elif hasattr(trace, 'values') and trace.values is not None:
            total += len(trace.values)
        elif hasattr(trace, 'z') and trace.z is not None:
            # For heatmaps: rows * columns
            z = trace.z
            if hasattr(z, 'shape'):
                total += z.shape[0] * z.shape[1]
            else:
                total += len(z) * len(z[0]) if z else 0
    return total


def should_optimize(fig, threshold=50_000):
    """Check if figure requires optimization."""
    total = calculate_total_points(fig)
    return total > threshold, total
```

---

## Large Dataset Handling

### When to Aggregate Before Plotting

**Aggregate when:**
- Raw data has more points than pixels available (screen width ~1920px max)
- Time series granularity exceeds visual resolution (e.g., 1-second data over months)
- Multiple overlapping series create visual noise
- Data patterns are visible at lower resolution

**Aggregation strategies by chart type:**

| Chart Type | Aggregation Method | Typical Reduction |
|------------|-------------------|-------------------|
| Time series | Resample to lower frequency | 10-100x |
| Scatter plot | Bin into 2D grid, use size/color for count | 10-50x |
| Bar chart | Pre-aggregate at source | N/A (already aggregated) |
| Histogram | Pre-compute bins | N/A (bins are aggregation) |

### Pre-Aggregation for Time Series

```python
import pandas as pd
import numpy as np

def aggregate_time_series(df, date_col, value_col, target_points=1000):
    """
    Aggregate time series to target number of points.

    Args:
        df: DataFrame with time series data
        date_col: Column name for datetime
        value_col: Column name for values
        target_points: Maximum number of points in output

    Returns:
        Aggregated DataFrame
    """
    if len(df) <= target_points:
        return df

    # Calculate required frequency
    date_range = df[date_col].max() - df[date_col].min()
    freq_seconds = date_range.total_seconds() / target_points

    # Map to pandas frequency strings
    if freq_seconds < 60:
        freq = f'{int(freq_seconds)}s'
    elif freq_seconds < 3600:
        freq = f'{int(freq_seconds / 60)}min'
    elif freq_seconds < 86400:
        freq = f'{int(freq_seconds / 3600)}h'
    else:
        freq = f'{int(freq_seconds / 86400)}D'

    # Resample with OHLC-style aggregation for accurate representation
    df_indexed = df.set_index(date_col)
    resampled = df_indexed[value_col].resample(freq).agg(['first', 'max', 'min', 'last', 'mean'])
    resampled = resampled.dropna()

    return resampled.reset_index()


def smart_resample(df, date_col, value_cols, max_points=2000):
    """
    Intelligently resample multiple columns preserving statistical properties.

    Args:
        df: DataFrame with time series data
        date_col: Column name for datetime
        value_cols: List of value columns or dict of {col: agg_func}
        max_points: Maximum points to return

    Returns:
        Resampled DataFrame
    """
    if len(df) <= max_points:
        return df

    # Determine resample frequency
    n_rows = len(df)
    skip_factor = n_rows // max_points

    # For price-like data, use OHLC
    # For volume-like data, use sum
    # For count-like data, use mean

    if isinstance(value_cols, dict):
        agg_funcs = value_cols
    else:
        # Default: use mean for all
        agg_funcs = {col: 'mean' for col in value_cols}

    df_indexed = df.set_index(date_col)
    date_range = df_indexed.index.max() - df_indexed.index.min()
    freq_seconds = date_range.total_seconds() / max_points

    if freq_seconds < 60:
        freq = f'{max(1, int(freq_seconds))}s'
    elif freq_seconds < 3600:
        freq = f'{max(1, int(freq_seconds / 60))}min'
    elif freq_seconds < 86400:
        freq = f'{max(1, int(freq_seconds / 3600))}h'
    else:
        freq = f'{max(1, int(freq_seconds / 86400))}D'

    resampled = df_indexed.resample(freq).agg(agg_funcs).dropna()
    return resampled.reset_index()
```

### Downsampling Strategies

#### 1. LTTB (Largest-Triangle-Three-Buckets)

Best for: **Time series where visual shape must be preserved**

LTTB preserves the visual appearance of the data by selecting points that maximize the area of triangles formed with neighboring points. It maintains peaks, valleys, and trends.

```python
import numpy as np

def lttb_downsample(x, y, target_points):
    """
    Largest-Triangle-Three-Buckets downsampling algorithm.

    Preserves visual appearance of time series by selecting points
    that maximize triangle areas with neighbors.

    Args:
        x: Array of x values (typically timestamps or indices)
        y: Array of y values
        target_points: Number of points to return (minimum 3)

    Returns:
        Tuple of (sampled_x, sampled_y) arrays

    Reference:
        Sveinn Steinarsson, "Downsampling Time Series for Visual Representation"
        https://skemman.is/bitstream/1946/15343/3/SS_MSthesis.pdf
    """
    n = len(x)
    if target_points >= n or target_points < 3:
        return x, y

    # Convert to numpy arrays
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    # Always keep first and last points
    sampled_x = [x[0]]
    sampled_y = [y[0]]

    # Bucket size (excluding first and last points)
    bucket_size = (n - 2) / (target_points - 2)

    # Previous selected point
    a_x, a_y = x[0], y[0]

    for i in range(target_points - 2):
        # Calculate bucket boundaries
        bucket_start = int(np.floor((i) * bucket_size)) + 1
        bucket_end = int(np.floor((i + 1) * bucket_size)) + 1

        # Next bucket boundaries (for averaging)
        next_bucket_start = int(np.floor((i + 1) * bucket_size)) + 1
        next_bucket_end = int(np.floor((i + 2) * bucket_size)) + 1
        next_bucket_end = min(next_bucket_end, n - 1)

        # Average point of next bucket
        avg_x = np.mean(x[next_bucket_start:next_bucket_end + 1])
        avg_y = np.mean(y[next_bucket_start:next_bucket_end + 1])

        # Find point in current bucket with largest triangle area
        max_area = -1
        max_idx = bucket_start

        for j in range(bucket_start, bucket_end):
            # Triangle area using cross product
            area = abs(
                (a_x - avg_x) * (y[j] - a_y) -
                (a_x - x[j]) * (avg_y - a_y)
            ) * 0.5

            if area > max_area:
                max_area = area
                max_idx = j

        sampled_x.append(x[max_idx])
        sampled_y.append(y[max_idx])
        a_x, a_y = x[max_idx], y[max_idx]

    # Add last point
    sampled_x.append(x[-1])
    sampled_y.append(y[-1])

    return np.array(sampled_x), np.array(sampled_y)


def lttb_downsample_df(df, x_col, y_col, target_points):
    """
    Apply LTTB downsampling to a DataFrame.

    Args:
        df: DataFrame with data
        x_col: Column name for x values
        y_col: Column name for y values
        target_points: Number of points to return

    Returns:
        Downsampled DataFrame
    """
    x = df[x_col].values
    y = df[y_col].values

    # Convert datetime to numeric for LTTB
    if np.issubdtype(df[x_col].dtype, np.datetime64):
        x_numeric = x.astype(np.int64)
        sampled_x, sampled_y = lttb_downsample(x_numeric, y, target_points)
        sampled_x = pd.to_datetime(sampled_x)
    else:
        sampled_x, sampled_y = lttb_downsample(x, y, target_points)

    return pd.DataFrame({x_col: sampled_x, y_col: sampled_y})
```

#### 2. Min-Max Downsampling

Best for: **Data where extremes must be preserved (price data, sensor readings)**

Min-max keeps both the minimum and maximum values in each bucket, ensuring peaks and valleys are never lost.

```python
def minmax_downsample(x, y, target_points):
    """
    Min-Max downsampling preserving extremes.

    For each bucket, keeps both the min and max points,
    ensuring no peaks or valleys are lost.

    Args:
        x: Array of x values
        y: Array of y values
        target_points: Approximate number of points (will return ~2x due to min+max)

    Returns:
        Tuple of (sampled_x, sampled_y) arrays
    """
    n = len(x)
    if target_points >= n:
        return x, y

    x = np.asarray(x)
    y = np.asarray(y)

    # Each bucket contributes 2 points (min and max)
    n_buckets = target_points // 2
    bucket_size = n / n_buckets

    sampled_x = []
    sampled_y = []

    for i in range(n_buckets):
        start = int(i * bucket_size)
        end = int((i + 1) * bucket_size)
        end = min(end, n)

        if start >= end:
            continue

        bucket_y = y[start:end]
        bucket_x = x[start:end]

        min_idx = np.argmin(bucket_y)
        max_idx = np.argmax(bucket_y)

        # Add in order of x value
        if min_idx <= max_idx:
            sampled_x.extend([bucket_x[min_idx], bucket_x[max_idx]])
            sampled_y.extend([bucket_y[min_idx], bucket_y[max_idx]])
        else:
            sampled_x.extend([bucket_x[max_idx], bucket_x[min_idx]])
            sampled_y.extend([bucket_y[max_idx], bucket_y[min_idx]])

    return np.array(sampled_x), np.array(sampled_y)
```

#### 3. Random Sampling

Best for: **Scatter plots where distribution matters more than individual points**

```python
def random_downsample(df, target_points, random_state=42):
    """
    Random downsampling for scatter plots.

    Preserves the statistical distribution of the data
    but loses specific point identity.

    Args:
        df: DataFrame to sample from
        target_points: Number of points to return
        random_state: Random seed for reproducibility

    Returns:
        Sampled DataFrame
    """
    if len(df) <= target_points:
        return df

    return df.sample(n=target_points, random_state=random_state)


def stratified_downsample(df, target_points, stratify_col, random_state=42):
    """
    Stratified random sampling preserving category proportions.

    Args:
        df: DataFrame to sample from
        target_points: Total number of points to return
        stratify_col: Column to stratify by
        random_state: Random seed

    Returns:
        Sampled DataFrame maintaining proportions of stratify_col
    """
    if len(df) <= target_points:
        return df

    # Calculate proportions
    proportions = df[stratify_col].value_counts(normalize=True)

    samples = []
    remaining = target_points
    categories = list(proportions.index)

    for i, cat in enumerate(categories):
        cat_df = df[df[stratify_col] == cat]

        if i == len(categories) - 1:
            # Last category gets remaining points
            n_samples = remaining
        else:
            n_samples = int(target_points * proportions[cat])

        n_samples = min(n_samples, len(cat_df))
        samples.append(cat_df.sample(n=n_samples, random_state=random_state))
        remaining -= n_samples

    return pd.concat(samples, ignore_index=True)
```

#### 4. Density-Based Downsampling

Best for: **Scatter plots where dense regions should remain visible**

```python
def density_downsample(x, y, target_points, grid_size=50):
    """
    Density-based downsampling for scatter plots.

    Divides space into grid cells and samples proportionally
    to density, preventing loss of sparse outliers.

    Args:
        x: Array of x values
        y: Array of y values
        target_points: Number of points to return
        grid_size: Number of cells per axis

    Returns:
        Tuple of (sampled_x, sampled_y, sampled_indices)
    """
    n = len(x)
    if target_points >= n:
        return x, y, np.arange(n)

    x = np.asarray(x)
    y = np.asarray(y)

    # Create grid
    x_bins = np.linspace(x.min(), x.max(), grid_size + 1)
    y_bins = np.linspace(y.min(), y.max(), grid_size + 1)

    # Assign points to cells
    x_cell = np.digitize(x, x_bins) - 1
    y_cell = np.digitize(y, y_bins) - 1

    # Clip to valid range
    x_cell = np.clip(x_cell, 0, grid_size - 1)
    y_cell = np.clip(y_cell, 0, grid_size - 1)

    # Cell ID for each point
    cell_id = x_cell * grid_size + y_cell

    # Count points per cell
    unique_cells, cell_counts = np.unique(cell_id, return_counts=True)
    n_cells = len(unique_cells)

    # Allocate samples per cell (at least 1 from each occupied cell)
    min_per_cell = 1
    remaining = target_points - n_cells * min_per_cell

    if remaining < 0:
        # Too few points, just random sample
        indices = np.random.choice(n, target_points, replace=False)
    else:
        # Proportional allocation
        cell_samples = dict(zip(unique_cells, [min_per_cell] * n_cells))

        # Add extra samples proportionally
        if remaining > 0:
            proportions = cell_counts / cell_counts.sum()
            extra = (proportions * remaining).astype(int)
            for cell, extra_count in zip(unique_cells, extra):
                cell_samples[cell] += extra_count

        # Sample from each cell
        indices = []
        for cell, n_samples in cell_samples.items():
            cell_indices = np.where(cell_id == cell)[0]
            n_samples = min(n_samples, len(cell_indices))
            sampled = np.random.choice(cell_indices, n_samples, replace=False)
            indices.extend(sampled)

        indices = np.array(indices)

    return x[indices], y[indices], indices
```

### Choosing a Downsampling Strategy

| Strategy | Best For | Preserves | Loses |
|----------|----------|-----------|-------|
| LTTB | Time series lines | Visual shape, trends | Some middle points |
| Min-Max | Price/sensor data | Peaks, valleys | Exact timing of extremes |
| Random | Scatter density | Statistical distribution | Outlier emphasis |
| Density-based | Scatter with outliers | Sparse regions | Some dense region detail |
| Aggregation | Any | Statistical summary | Individual points |

**Decision flow:**

```
Is it a time series?
├─ Yes: Do extremes matter (price, sensor)?
│       ├─ Yes: Use Min-Max
│       └─ No:  Use LTTB
└─ No (scatter/other):
       Are there outliers or clusters?
       ├─ Yes: Use Density-based
       └─ No:  Use Random
```

---

## WebGL Mode

### When to Use WebGL Traces

Use WebGL versions of traces when data points exceed 10,000:

| Standard Trace | WebGL Trace | Use WebGL When |
|----------------|-------------|----------------|
| `go.Scatter` | `go.Scattergl` | > 10,000 points |
| `go.Scattermapbox` | `go.Scattermapboxgl` | > 5,000 points |
| `go.Scatter3d` | `go.Scatter3d` (WebGL by default) | Always |

### Scattergl Implementation

```python
import plotly.graph_objects as go
import numpy as np

def create_scatter_trace(x, y, name, color='#60A5FA', mode='markers', **kwargs):
    """
    Create scatter trace, automatically using WebGL for large datasets.

    Args:
        x: X values
        y: Y values
        name: Trace name
        color: Marker/line color
        mode: 'markers', 'lines', or 'lines+markers'
        **kwargs: Additional trace parameters

    Returns:
        Appropriate Plotly trace object
    """
    n_points = len(x)

    # Use WebGL for large datasets
    TraceClass = go.Scattergl if n_points > 10_000 else go.Scatter

    trace_params = {
        'x': x,
        'y': y,
        'name': name,
        'mode': mode,
    }

    if 'markers' in mode:
        trace_params['marker'] = dict(
            color=color,
            size=kwargs.get('marker_size', 4 if n_points > 50_000 else 6),
            opacity=kwargs.get('opacity', 0.7 if n_points > 10_000 else 1.0)
        )

    if 'lines' in mode:
        trace_params['line'] = dict(
            color=color,
            width=kwargs.get('line_width', 1 if n_points > 50_000 else 2)
        )

    trace_params.update(kwargs)
    return TraceClass(**trace_params)


# Example: Large time series with WebGL
np.random.seed(42)
n_points = 100_000
dates = pd.date_range('2020-01-01', periods=n_points, freq='min')
values = np.cumsum(np.random.randn(n_points)) + 100

fig = go.Figure()
fig.add_trace(create_scatter_trace(
    x=dates,
    y=values,
    name='Sensor Reading',
    mode='lines',
    color='#60A5FA'
))

fig.update_layout(
    title='100K Point Time Series with WebGL',
    paper_bgcolor='#0e1729',
    plot_bgcolor='#0e1729',
    font=dict(color='#d3d4d6')
)
```

### WebGL Limitations

**Be aware of these WebGL constraints:**

| Limitation | Description | Workaround |
|------------|-------------|------------|
| No fill | `fill='tozeroy'` not supported | Use separate filled trace or accept lines only |
| Limited markers | Fewer marker symbol options | Use basic symbols (circle, square) |
| No gradients | Cannot use gradient fills | Use solid colors |
| Text rendering | Text labels less crisp | Reduce label count or use annotations |
| Memory limits | GPU memory constraints | Stay under 1M points per trace |

```python
# Check if WebGL is appropriate
def can_use_webgl(trace_config):
    """
    Check if trace configuration is compatible with WebGL.

    Args:
        trace_config: Dict with trace parameters

    Returns:
        Tuple of (can_use: bool, reason: str)
    """
    incompatible_features = []

    if trace_config.get('fill') and trace_config['fill'] != 'none':
        incompatible_features.append('fill')

    if trace_config.get('text') and len(trace_config.get('x', [])) > 1000:
        incompatible_features.append('text labels on many points')

    marker = trace_config.get('marker', {})
    if marker.get('symbol') and marker['symbol'] not in [
        'circle', 'circle-open', 'square', 'square-open',
        'diamond', 'diamond-open', 'cross', 'x'
    ]:
        incompatible_features.append(f"marker symbol '{marker['symbol']}'")

    if incompatible_features:
        return False, f"WebGL incompatible: {', '.join(incompatible_features)}"

    return True, "WebGL compatible"
```

### Performance Comparison

Benchmark results for different data sizes:

| Data Points | go.Scatter | go.Scattergl | Improvement |
|-------------|------------|--------------|-------------|
| 1,000 | 50ms | 60ms | -20% (overhead) |
| 10,000 | 200ms | 80ms | 2.5x faster |
| 50,000 | 1,500ms | 150ms | 10x faster |
| 100,000 | 5,000ms | 300ms | 16x faster |
| 500,000 | 25,000ms+ | 800ms | 30x+ faster |

**Note:** WebGL has initialization overhead, so it's slower for small datasets (< 5,000 points).

---

## Plotly-Specific Optimizations

### Reducing Trace Count

Each trace adds overhead. Combine when possible:

```python
# BAD: One trace per category (slow for many categories)
for category in categories:  # 50 categories = 50 traces
    cat_data = df[df['category'] == category]
    fig.add_trace(go.Scatter(
        x=cat_data['x'],
        y=cat_data['y'],
        name=category
    ))

# GOOD: Single trace with color encoding
fig.add_trace(go.Scatter(
    x=df['x'],
    y=df['y'],
    mode='markers',
    marker=dict(
        color=df['category_code'],  # Numeric encoding
        colorscale='Viridis',
        showscale=True
    ),
    text=df['category'],  # For hover
    hovertemplate='%{text}<br>x: %{x}<br>y: %{y}<extra></extra>'
))
```

**Maximum recommended traces:**

| Trace Type | Maximum Count | Notes |
|------------|---------------|-------|
| Lines | 20-30 | Legend becomes unwieldy |
| Scatter (SVG) | 15-20 | Performance degrades |
| Scatter (WebGL) | 50+ | GPU handles well |
| Bar | 20-30 | Visual clarity limit |
| Filled areas | 10-15 | Layering becomes confusing |

### Simplifying Hover Templates

Complex hover templates with many data points cause lag:

```python
# SLOW: Formatting operations on every hover
hovertemplate = (
    '<b>%{customdata[0]}</b><br>'
    'Date: %{x|%B %d, %Y at %H:%M:%S}<br>'
    'Value: $%{y:,.2f}<br>'
    'Change: %{customdata[1]:+.2%}<br>'
    'Volume: %{customdata[2]:,.0f}<br>'
    'High: $%{customdata[3]:,.2f}<br>'
    'Low: $%{customdata[4]:,.2f}<br>'
    '<extra></extra>'
)

# FAST: Minimal hover for large datasets
hovertemplate_minimal = '%{x|%b %d}<br>$%{y:,.0f}<extra></extra>'
```

**Hover optimization guidelines:**

| Data Points | Hover Complexity | Recommendation |
|-------------|------------------|----------------|
| < 5,000 | Full detail | Use customdata freely |
| 5,000 - 20,000 | Moderate | Limit to 3-4 fields |
| 20,000 - 50,000 | Minimal | 1-2 fields only |
| 50,000+ | Disabled | Consider `hoverinfo='skip'` |

```python
def get_hover_template(n_points, full_template, minimal_template):
    """Select hover template based on data size."""
    if n_points < 5_000:
        return full_template
    elif n_points < 20_000:
        return minimal_template
    else:
        return None  # Will use hoverinfo='skip'


def configure_hover_for_performance(fig, n_points):
    """Configure hover settings for optimal performance."""
    if n_points > 50_000:
        fig.update_traces(hoverinfo='skip')
        fig.update_layout(hovermode=False)
    elif n_points > 20_000:
        fig.update_layout(hovermode='closest')  # Not 'x unified'
    else:
        fig.update_layout(hovermode='x unified')  # Full interactivity
```

### Disabling Animations for Large Data

Plotly animations re-render on each frame. Disable for large datasets:

```python
fig.update_layout(
    # Disable transitions for large data
    transition={'duration': 0} if n_points > 10_000 else {'duration': 300},

    # Disable uirevision animation
    uirevision='constant',
)

# For updates via Dash or fig.update_*
config = {
    'staticPlot': True if n_points > 100_000 else False,  # Disable all interaction
    'scrollZoom': False if n_points > 50_000 else True,
}
```

### Axis Optimization

Reduce tick calculation overhead:

```python
# For large datasets, set explicit tick values
fig.update_xaxes(
    # Avoid automatic tick calculation
    tickmode='linear' if n_points > 10_000 else 'auto',
    tick0=x_min,
    dtick=(x_max - x_min) / 10,  # Exactly 10 ticks

    # Disable spike lines for performance
    showspikes=False if n_points > 20_000 else True,
)

fig.update_yaxes(
    tickmode='linear' if n_points > 10_000 else 'auto',
    tick0=y_min,
    dtick=(y_max - y_min) / 8,
    showspikes=False if n_points > 20_000 else True,
)
```

---

## Animation Considerations

### When to Animate

**Animate when:**
- Showing state transitions (before/after)
- Time-lapse visualization (days, months evolving)
- Drawing attention to change
- Educational/explanatory context

**Do NOT animate when:**
- Data points exceed 5,000
- Users need to analyze specific values
- Animation would take > 10 seconds
- Mobile/low-power devices are expected

### Animation Performance Limits

| Metric | Recommended Maximum | Hard Limit |
|--------|---------------------|------------|
| Frames | 100 | 500 |
| Points per frame | 5,000 | 20,000 |
| Total points (all frames) | 100,000 | 500,000 |
| Frame duration | 50ms minimum | 20ms |
| Total animation | 10 seconds | 30 seconds |

### Creating Efficient Animations

```python
import plotly.graph_objects as go
import numpy as np

def create_animation(df, date_col, x_col, y_col, max_frames=100, max_points_per_frame=5000):
    """
    Create animation with performance constraints.

    Args:
        df: DataFrame with data
        date_col: Column for animation frames (usually date)
        x_col: X axis column
        y_col: Y axis column
        max_frames: Maximum number of frames
        max_points_per_frame: Maximum points per frame

    Returns:
        Plotly figure with animation
    """
    unique_dates = df[date_col].unique()
    n_frames = len(unique_dates)

    # Subsample frames if too many
    if n_frames > max_frames:
        step = n_frames // max_frames
        unique_dates = unique_dates[::step]

    frames = []
    for date in unique_dates:
        frame_df = df[df[date_col] == date]

        # Downsample if needed
        if len(frame_df) > max_points_per_frame:
            frame_df = frame_df.sample(n=max_points_per_frame)

        frames.append(go.Frame(
            data=[go.Scatter(
                x=frame_df[x_col],
                y=frame_df[y_col],
                mode='markers'
            )],
            name=str(date)
        ))

    # Initial frame
    initial_df = df[df[date_col] == unique_dates[0]]
    if len(initial_df) > max_points_per_frame:
        initial_df = initial_df.sample(n=max_points_per_frame)

    fig = go.Figure(
        data=[go.Scatter(
            x=initial_df[x_col],
            y=initial_df[y_col],
            mode='markers'
        )],
        frames=frames
    )

    # Add animation controls
    fig.update_layout(
        updatemenus=[{
            'type': 'buttons',
            'showactive': False,
            'buttons': [
                {
                    'label': 'Play',
                    'method': 'animate',
                    'args': [None, {
                        'frame': {'duration': 100, 'redraw': True},
                        'fromcurrent': True,
                        'transition': {'duration': 0}  # No transition for speed
                    }]
                },
                {
                    'label': 'Pause',
                    'method': 'animate',
                    'args': [[None], {
                        'frame': {'duration': 0, 'redraw': False},
                        'mode': 'immediate'
                    }]
                }
            ]
        }],
        sliders=[{
            'steps': [
                {'args': [[str(d)], {'frame': {'duration': 0}, 'mode': 'immediate'}],
                 'label': str(d)[:10],
                 'method': 'animate'}
                for d in unique_dates
            ]
        }]
    )

    return fig
```

### Alternatives to Animation

When animation is not performant, consider these alternatives:

| Alternative | Use Case | Implementation |
|-------------|----------|----------------|
| Small multiples | Compare time periods | Subplot grid by period |
| Slider filter | User-controlled time | Range slider or dropdown |
| Static key frames | Highlight specific moments | Multi-panel image |
| Interactive traces | Toggle visibility | Legend click or buttons |

```python
# Alternative: Small multiples instead of animation
from plotly.subplots import make_subplots

def create_small_multiples_timeline(df, date_col, x_col, y_col, n_panels=6):
    """Create small multiples instead of animation for large data."""
    unique_dates = sorted(df[date_col].unique())

    # Select evenly spaced dates
    step = len(unique_dates) // n_panels
    selected_dates = unique_dates[::step][:n_panels]

    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=[str(d)[:10] for d in selected_dates]
    )

    for i, date in enumerate(selected_dates):
        row = (i // 3) + 1
        col = (i % 3) + 1

        frame_df = df[df[date_col] == date]
        fig.add_trace(
            go.Scatter(
                x=frame_df[x_col],
                y=frame_df[y_col],
                mode='markers',
                marker=dict(size=4, opacity=0.7),
                showlegend=False
            ),
            row=row, col=col
        )

    return fig
```

---

## Static Export Optimization

### Resolution Settings

| Export Format | Use Case | Recommended Settings |
|---------------|----------|---------------------|
| PNG (screen) | Web display | scale=1, width=auto |
| PNG (print) | Reports, documents | scale=2-3, width=1200+ |
| SVG | Scalable graphics | width/height in px, no scale |
| PDF | Print documents | scale=2, explicit dimensions |
| WebP | Web (modern) | scale=1, quality=90 |

```python
def export_figure(fig, filename, purpose='screen', width=None, height=None):
    """
    Export figure with appropriate settings.

    Args:
        fig: Plotly figure
        filename: Output filename with extension
        purpose: 'screen', 'print', 'web', 'thumbnail'
        width: Override width (optional)
        height: Override height (optional)
    """
    ext = filename.split('.')[-1].lower()

    settings = {
        'screen': {'scale': 1, 'width': 900, 'height': 500},
        'print': {'scale': 2, 'width': 1200, 'height': 675},
        'web': {'scale': 1, 'width': 800, 'height': 450},
        'thumbnail': {'scale': 1, 'width': 400, 'height': 225},
    }

    config = settings.get(purpose, settings['screen'])

    if width:
        config['width'] = width
    if height:
        config['height'] = height

    if ext == 'svg':
        # SVG doesn't use scale
        fig.write_image(filename, width=config['width'], height=config['height'])
    elif ext == 'pdf':
        fig.write_image(filename, **config, engine='kaleido')
    else:  # PNG, WebP, JPEG
        fig.write_image(filename, **config)

    return filename
```

### File Size Considerations

| Factor | Impact on Size | Optimization |
|--------|----------------|--------------|
| Data points | Linear increase | Downsample before export |
| Number of traces | Moderate increase | Combine traces |
| SVG complexity | Can be huge | Use PNG for complex charts |
| Image dimensions | Quadratic increase | Use appropriate scale |
| Color depth | Minor | Use PNG-8 for simple charts |

```python
def estimate_export_size(fig, format='png', scale=1):
    """
    Estimate file size for export.

    Args:
        fig: Plotly figure
        format: 'png', 'svg', 'pdf'
        scale: Scale factor

    Returns:
        Estimated size in bytes
    """
    n_points = calculate_total_points(fig)
    n_traces = len(fig.data)

    # Base sizes in bytes
    base_sizes = {
        'png': 50_000,   # 50KB base
        'svg': 10_000,   # 10KB base (but scales with complexity)
        'pdf': 100_000,  # 100KB base
    }

    base = base_sizes.get(format, 50_000)

    # Scale factors
    point_factor = n_points * (0.1 if format == 'png' else 0.5)  # SVG stores all points
    trace_factor = n_traces * 1000
    scale_factor = scale ** 2  # Image size scales quadratically

    if format == 'svg':
        estimated = base + point_factor * 10 + trace_factor
    else:
        estimated = (base + point_factor + trace_factor) * scale_factor

    return int(estimated)


def recommend_export_format(fig, max_size_kb=500):
    """Recommend export format based on figure complexity."""
    n_points = calculate_total_points(fig)

    if n_points < 1000:
        return 'svg', "SVG recommended for small datasets (scalable, sharp)"
    elif n_points < 10_000:
        png_size = estimate_export_size(fig, 'png', scale=2)
        if png_size < max_size_kb * 1024:
            return 'png', "PNG recommended (good quality, reasonable size)"
        return 'png', "PNG with scale=1 recommended (size constraints)"
    else:
        return 'png', "PNG required for large datasets (SVG would be too large)"
```

### When to Use Static vs Interactive

| Scenario | Format | Reasoning |
|----------|--------|-----------|
| Email/Slack | PNG | Universal compatibility |
| Web dashboard | Interactive | Full functionality |
| PDF report | PDF/PNG | Print-ready |
| Documentation | SVG | Scales perfectly |
| Presentation | PNG | Predictable display |
| Data exploration | Interactive | User needs to drill down |
| Large dataset (100K+) | PNG | Interactive would lag |

```python
def should_use_static(n_points, context):
    """
    Determine if static export is preferable.

    Args:
        n_points: Number of data points
        context: 'dashboard', 'report', 'email', 'exploration'

    Returns:
        Tuple of (use_static: bool, reason: str)
    """
    static_contexts = {'report', 'email', 'presentation', 'documentation'}

    if context in static_contexts:
        return True, f"Static recommended for {context} context"

    if n_points > 100_000:
        return True, "Static recommended: interactive would be too slow"

    if n_points > 50_000:
        return False, "Interactive possible but consider downsampling"

    return False, "Interactive recommended for exploration"
```

---

## Browser and Rendering Limits

### Maximum Recommended Values

| Resource | Recommended Max | Hard Limit | Symptom When Exceeded |
|----------|-----------------|------------|----------------------|
| Total data points | 500,000 | 2,000,000 | Browser freeze |
| Points per trace | 100,000 | 500,000 | Slow render |
| Number of traces | 50 | 200 | Legend overflow, slow toggle |
| SVG elements | 10,000 | 50,000 | DOM slowdown |
| Frames (animation) | 100 | 500 | Memory exhaustion |
| Figure size (MB) | 50 | 200 | Memory crash |

### Memory Estimation

```python
def estimate_memory_usage(fig):
    """
    Estimate memory usage of a Plotly figure.

    Args:
        fig: Plotly figure object

    Returns:
        Dict with memory estimates in MB
    """
    import sys

    n_points = calculate_total_points(fig)
    n_traces = len(fig.data)
    n_frames = len(fig.frames) if fig.frames else 0

    # Rough estimates in bytes
    # Each point: ~64 bytes (x, y as float64) + metadata
    point_memory = n_points * 100

    # Each trace: ~10KB overhead
    trace_memory = n_traces * 10_000

    # Each frame: roughly duplicates data
    frame_memory = n_frames * point_memory * 0.5

    # Layout: typically 50-100KB
    layout_memory = 75_000

    total_bytes = point_memory + trace_memory + frame_memory + layout_memory

    return {
        'data_points_mb': point_memory / (1024 ** 2),
        'traces_mb': trace_memory / (1024 ** 2),
        'frames_mb': frame_memory / (1024 ** 2),
        'layout_mb': layout_memory / (1024 ** 2),
        'total_mb': total_bytes / (1024 ** 2),
        'warning': total_bytes > 50 * 1024 ** 2
    }


def check_browser_compatibility(fig):
    """
    Check if figure is compatible with browser limits.

    Returns:
        List of warnings/recommendations
    """
    warnings = []

    n_points = calculate_total_points(fig)
    n_traces = len(fig.data)
    n_frames = len(fig.frames) if fig.frames else 0
    memory = estimate_memory_usage(fig)

    if n_points > 500_000:
        warnings.append(f"CRITICAL: {n_points:,} points exceeds 500K limit. Downsample required.")
    elif n_points > 100_000:
        warnings.append(f"WARNING: {n_points:,} points may cause slowness. Consider WebGL.")

    if n_traces > 50:
        warnings.append(f"WARNING: {n_traces} traces exceeds recommended 50. Combine traces.")

    if n_frames > 100:
        warnings.append(f"WARNING: {n_frames} frames may cause memory issues.")

    if memory['total_mb'] > 50:
        warnings.append(f"WARNING: Estimated {memory['total_mb']:.1f}MB exceeds 50MB limit.")

    # Check for WebGL usage
    webgl_traces = sum(1 for t in fig.data if type(t).__name__ in ['Scattergl', 'Scattermapboxgl'])
    svg_large_traces = sum(1 for t in fig.data
                          if type(t).__name__ == 'Scatter'
                          and hasattr(t, 'x')
                          and t.x is not None
                          and len(t.x) > 10_000)

    if svg_large_traces > 0:
        warnings.append(f"RECOMMENDATION: {svg_large_traces} large traces should use WebGL (Scattergl)")

    return warnings
```

### Mobile Device Constraints

Mobile devices have stricter limits:

| Resource | Desktop | Mobile | Mobile Recommendation |
|----------|---------|--------|----------------------|
| Data points | 500K | 50K | Aggressive downsampling |
| Traces | 50 | 15 | Combine or filter |
| Memory | 200MB | 50MB | Smaller figures |
| Touch targets | Any | 44px+ | Larger markers |
| Hover | Full | Limited | Simplified or disabled |

```python
def optimize_for_mobile(fig, max_points=50_000, max_traces=15):
    """
    Optimize figure for mobile devices.

    Args:
        fig: Plotly figure
        max_points: Maximum points for mobile
        max_traces: Maximum traces for mobile

    Returns:
        Modified figure
    """
    import copy
    mobile_fig = copy.deepcopy(fig)

    # Reduce traces if needed
    if len(mobile_fig.data) > max_traces:
        # Keep only first N traces
        mobile_fig.data = mobile_fig.data[:max_traces]

    # Increase marker size for touch
    mobile_fig.update_traces(
        marker=dict(size=10),  # Minimum 44px touch target
        selector=dict(mode='markers')
    )

    # Simplify hover
    total_points = calculate_total_points(mobile_fig)
    if total_points > 10_000:
        mobile_fig.update_traces(hoverinfo='skip')
        mobile_fig.update_layout(hovermode=False)

    # Adjust layout for mobile
    mobile_fig.update_layout(
        margin=dict(l=40, r=20, t=50, b=60),
        legend=dict(
            orientation='h',
            yanchor='top',
            y=-0.15,
            xanchor='center',
            x=0.5
        )
    )

    return mobile_fig


def get_device_config(device_type='desktop'):
    """
    Get Plotly config for device type.

    Args:
        device_type: 'desktop', 'tablet', or 'mobile'

    Returns:
        Config dict for fig.show() or dcc.Graph
    """
    configs = {
        'desktop': {
            'scrollZoom': True,
            'displayModeBar': True,
            'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
            'doubleClick': 'reset+autosize',
        },
        'tablet': {
            'scrollZoom': False,  # Conflicts with page scroll
            'displayModeBar': True,
            'modeBarButtonsToRemove': ['lasso2d', 'select2d', 'pan2d'],
            'doubleClick': 'reset',
        },
        'mobile': {
            'scrollZoom': False,
            'displayModeBar': False,  # Save screen space
            'staticPlot': False,  # Still allow pinch-zoom
            'doubleClick': 'reset',
        }
    }
    return configs.get(device_type, configs['desktop'])
```

---

## Performance Optimization Checklist

Before deploying a visualization, verify:

### Data Volume

- [ ] Total data points calculated and within limits
- [ ] Downsampling applied if needed (LTTB, Min-Max, or aggregation)
- [ ] WebGL traces used for 10K+ points
- [ ] Memory usage estimated and acceptable

### Trace Optimization

- [ ] Trace count minimized (combined where possible)
- [ ] WebGL used for large scatter/line traces
- [ ] Unused trace properties removed
- [ ] Colors use efficient encoding (not per-point if avoidable)

### Hover and Interaction

- [ ] Hover templates simplified for large data
- [ ] Hover disabled if > 50K points
- [ ] Animations disabled for large data
- [ ] Zoom/pan configured appropriately

### Export

- [ ] Appropriate format selected (PNG vs SVG vs interactive)
- [ ] Resolution appropriate for use case
- [ ] File size acceptable for delivery method

### Device Compatibility

- [ ] Desktop browser limits respected
- [ ] Mobile optimization applied if needed
- [ ] Touch targets adequate for mobile (44px+)

---

## Quick Reference

### Data Point Thresholds

| Points | Action |
|--------|--------|
| < 10K | No optimization needed |
| 10K - 50K | Use WebGL |
| 50K - 100K | WebGL + simplify hover |
| 100K - 500K | Downsample |
| 500K+ | Aggressive downsample + static export |

### Downsampling Selection

| Data Type | Algorithm |
|-----------|-----------|
| Time series (shape matters) | LTTB |
| Price/sensor (extremes matter) | Min-Max |
| Scatter (distribution matters) | Random or Density |
| Categorical | Pre-aggregate |

### WebGL Trace Mapping

| Standard | WebGL |
|----------|-------|
| `go.Scatter` | `go.Scattergl` |
| `go.Scattermapbox` | `go.Scattermapboxgl` |
| `go.Scatter3d` | Already WebGL |

### Export Format Selection

| Context | Format | Scale |
|---------|--------|-------|
| Web display | PNG | 1x |
| Print/report | PNG/PDF | 2-3x |
| Scalable | SVG | N/A |
| Email | PNG | 1x |
| Large data | PNG only | 1x |

### Mobile Limits

| Resource | Maximum |
|----------|---------|
| Points | 50,000 |
| Traces | 15 |
| Memory | 50MB |
| Marker size | 10px minimum |
