# 11 - Interaction Modes

> Configure zoom, pan, range sliders, buttons, and modebar controls.
> Interaction design should match the chart type and user needs.

---

## Zoom Behavior

Zoom behavior should be configured based on chart type and expected user interaction patterns.

### Zoom Modes

| Mode | Plotly Value | Behavior | Best For |
|------|-------------|----------|----------|
| **Horizontal (X-axis)** | `dragmode='zoom'` + `xaxis.fixedrange=False`, `yaxis.fixedrange=True` | Zoom only on time/x-axis | Time series |
| **Box Zoom (2D)** | `dragmode='zoom'` | Select rectangular region | Scatter plots, bubble charts |
| **Disabled** | `dragmode=False` or `fixedrange=True` | No zoom interaction | Bar charts, static presentations |

### Decision Tree

```
ZOOM MODE SELECTION
|
+-- Is this a time series chart?
|   YES --> X-axis zoom only (horizontal)
|
+-- Is this a scatter/bubble plot?
|   YES --> Box zoom (2D)
|
+-- Is this a bar/categorical chart?
|   YES --> Disable zoom (fixed range)
|
+-- Is this for static export/presentation?
|   YES --> Disable all interactions
|
+-- Default --> Box zoom (2D)
```

### Time Series: X-Axis Zoom

For time series, users typically want to zoom into a time range while preserving the y-axis auto-scaling.

```python
fig.update_layout(
    dragmode='zoom',
    xaxis=dict(
        fixedrange=False,
        rangeslider=dict(visible=False)  # Optional: add range slider below
    ),
    yaxis=dict(
        fixedrange=True,  # Prevent y-axis zoom
        autorange=True    # Y-axis auto-scales to visible data
    )
)
```

**Note on Y-Axis Auto-Scaling:**

By default, Plotly does not automatically rescale the y-axis when you zoom on the x-axis. To enable this behavior, you need `yaxis.autorange=True` combined with `fixedrange=True`. For more sophisticated auto-scaling, consider using `uirevision`:

```python
fig.update_layout(
    uirevision='constant',  # Preserve UI state across updates
    yaxis=dict(
        autorange=True,
        fixedrange=True
    )
)
```

### Scatter Plot: Box Zoom

For scatter plots, enable both axes for zoom selection:

```python
fig.update_layout(
    dragmode='zoom',
    xaxis=dict(fixedrange=False),
    yaxis=dict(fixedrange=False)
)
```

### Bar Charts: Disable Zoom

Bar charts typically should not be zoomable as it distorts the categorical comparison:

```python
fig.update_layout(
    dragmode=False,
    xaxis=dict(fixedrange=True),
    yaxis=dict(fixedrange=True)
)
```

### Lasso and Box Select

For scatter plots where selection (not zoom) is the primary interaction:

```python
fig.update_layout(
    dragmode='select',      # Box selection
    # OR
    dragmode='lasso',       # Freeform selection
    selectdirection='any',  # 'h', 'v', or 'any'
)

# Configure selection appearance
fig.update_traces(
    selected=dict(
        marker=dict(color='#FBBF24', opacity=1)
    ),
    unselected=dict(
        marker=dict(opacity=0.3)
    )
)
```

---

## Pan Behavior

Panning allows users to navigate within the data without changing the zoom level.

### When to Enable Pan

| Scenario | Pan Recommended | Reason |
|----------|----------------|--------|
| Long time series with range slider | No | Use range slider instead |
| Long time series without range slider | Yes | Navigate time range |
| Zoomed scatter plot | Yes | Explore zoomed region |
| Bar charts | No | All data should be visible |
| Dashboard charts | No | Prefer fixed view |

### Pan Configuration

```python
# Enable pan as default drag mode
fig.update_layout(
    dragmode='pan',
    xaxis=dict(fixedrange=False),
    yaxis=dict(fixedrange=False)
)
```

### Constrained Panning

Limit panning to specific axis or range:

```python
# X-axis only panning (for time series)
fig.update_layout(
    dragmode='pan',
    xaxis=dict(
        fixedrange=False,
        constrain='domain',       # Keep within data bounds
        constraintoward='center'  # Constrain from center
    ),
    yaxis=dict(fixedrange=True)  # No vertical pan
)
```

### Scroll Wheel Zoom

Enable scroll wheel for zoom while keeping drag for pan:

```python
fig.update_layout(
    dragmode='pan',
    xaxis=dict(
        fixedrange=False
    )
)

# Scroll wheel zoom is enabled by default when fixedrange=False
# To disable scroll zoom specifically:
config = {'scrollZoom': False}
fig.show(config=config)
```

---

## Range Sliders

Range sliders provide a miniature overview of the full data range with a draggable selection window.

### When to Use Range Sliders

| Scenario | Use Range Slider | Alternative |
|----------|-----------------|-------------|
| Long time series (>6 months daily) | Yes | Time range buttons |
| Short time series (<1 month) | No | Direct zoom |
| Multi-panel charts | No (too much space) | Shared zoom only |
| Dashboard with limited height | No | Time range buttons |
| Exploratory analysis | Yes | None |

### Basic Configuration

```python
fig.update_layout(
    xaxis=dict(
        rangeslider=dict(
            visible=True,
            thickness=0.05,        # Height as fraction of plot (0-1)
            bgcolor='#0e1729',     # Match background
            bordercolor='#334155',
            borderwidth=1
        ),
        rangeselector=dict(visible=False)  # Separate from range slider
    )
)
```

### Setting Default Visible Range

Show the most recent portion of data by default:

```python
import pandas as pd

# Calculate default range (e.g., last 90 days)
end_date = df['date'].max()
start_date = end_date - pd.Timedelta(days=90)

fig.update_layout(
    xaxis=dict(
        range=[start_date, end_date],  # Initial visible range
        rangeslider=dict(
            visible=True,
            range=[df['date'].min(), df['date'].max()]  # Full range in slider
        )
    )
)
```

### Range Slider Styling

Match range slider to the dark theme:

```python
fig.update_layout(
    xaxis=dict(
        rangeslider=dict(
            visible=True,
            thickness=0.04,
            bgcolor='#0e1729',
            bordercolor='#334155',
            borderwidth=1,
            yaxis=dict(rangemode='auto')
        ),
        # Style the range selector handles
        rangeselector=dict(
            bgcolor='#1e293b',
            activecolor='#3B82F6',
            bordercolor='#334155',
            font=dict(color='#d3d4d6')
        )
    )
)
```

### Range Slider with Line Preview

Show a miniature line chart in the range slider:

```python
# By default, the range slider shows a simplified version of the data
# To customize:
fig.update_layout(
    xaxis=dict(
        rangeslider=dict(
            visible=True,
            yaxis=dict(
                rangemode='match'  # Match main y-axis range
            )
        )
    )
)
```

---

## Buttons and Dropdowns

### Range Selector Buttons (Time Presets)

Add quick time range selection buttons for time series:

```python
fig.update_layout(
    xaxis=dict(
        rangeselector=dict(
            buttons=[
                dict(count=7, label='1W', step='day', stepmode='backward'),
                dict(count=1, label='1M', step='month', stepmode='backward'),
                dict(count=3, label='3M', step='month', stepmode='backward'),
                dict(count=6, label='6M', step='month', stepmode='backward'),
                dict(count=1, label='YTD', step='year', stepmode='todate'),
                dict(count=1, label='1Y', step='year', stepmode='backward'),
                dict(label='ALL', step='all')
            ],
            bgcolor='#1e293b',
            activecolor='#3B82F6',
            bordercolor='#334155',
            borderwidth=1,
            font=dict(color='#d3d4d6', size=11),
            x=0,
            xanchor='left',
            y=1.0,
            yanchor='bottom'
        )
    )
)
```

### Range Selector Options

| Parameter | Options | Description |
|-----------|---------|-------------|
| `step` | `'year'`, `'month'`, `'day'`, `'hour'`, `'minute'`, `'second'`, `'all'` | Time unit |
| `stepmode` | `'backward'`, `'todate'` | `backward`: from end, `todate`: from start of period |
| `count` | integer | Number of step units |

### Common Time Range Presets

```python
# Financial / Trading
TRADING_RANGES = [
    dict(count=1, label='1D', step='day', stepmode='backward'),
    dict(count=7, label='1W', step='day', stepmode='backward'),
    dict(count=1, label='1M', step='month', stepmode='backward'),
    dict(count=3, label='3M', step='month', stepmode='backward'),
    dict(count=1, label='YTD', step='year', stepmode='todate'),
    dict(count=1, label='1Y', step='year', stepmode='backward'),
    dict(label='ALL', step='all')
]

# Long-term Analysis
LONGTERM_RANGES = [
    dict(count=1, label='1Y', step='year', stepmode='backward'),
    dict(count=2, label='2Y', step='year', stepmode='backward'),
    dict(count=5, label='5Y', step='year', stepmode='backward'),
    dict(label='ALL', step='all')
]

# Intraday
INTRADAY_RANGES = [
    dict(count=1, label='1H', step='hour', stepmode='backward'),
    dict(count=4, label='4H', step='hour', stepmode='backward'),
    dict(count=1, label='1D', step='day', stepmode='backward'),
    dict(count=7, label='1W', step='day', stepmode='backward'),
    dict(label='ALL', step='all')
]
```

### Updatemenus: Trace Visibility Toggles

Create buttons to toggle trace visibility:

```python
fig.update_layout(
    updatemenus=[
        dict(
            type='buttons',
            direction='right',
            active=0,
            x=0,
            xanchor='left',
            y=1.15,
            yanchor='top',
            bgcolor='#1e293b',
            bordercolor='#334155',
            font=dict(color='#d3d4d6'),
            buttons=[
                dict(
                    label='All',
                    method='update',
                    args=[{'visible': [True, True, True]}]
                ),
                dict(
                    label='Series A Only',
                    method='update',
                    args=[{'visible': [True, False, False]}]
                ),
                dict(
                    label='Series B Only',
                    method='update',
                    args=[{'visible': [False, True, False]}]
                )
            ]
        )
    ]
)
```

### Updatemenus: Dropdown for Metric Selection

Create dropdown to switch between different views:

```python
fig.update_layout(
    updatemenus=[
        dict(
            type='dropdown',
            direction='down',
            active=0,
            x=0,
            xanchor='left',
            y=1.15,
            yanchor='top',
            bgcolor='#1e293b',
            bordercolor='#334155',
            font=dict(color='#d3d4d6'),
            buttons=[
                dict(
                    label='Linear Scale',
                    method='relayout',
                    args=[{'yaxis.type': 'linear'}]
                ),
                dict(
                    label='Log Scale',
                    method='relayout',
                    args=[{'yaxis.type': 'log'}]
                )
            ]
        )
    ]
)
```

### Combining Multiple Updatemenus

```python
fig.update_layout(
    updatemenus=[
        # Scale toggle (dropdown)
        dict(
            type='dropdown',
            direction='down',
            x=0,
            xanchor='left',
            y=1.15,
            buttons=[
                dict(label='Linear', method='relayout', args=[{'yaxis.type': 'linear'}]),
                dict(label='Log', method='relayout', args=[{'yaxis.type': 'log'}])
            ],
            bgcolor='#1e293b',
            bordercolor='#334155',
            font=dict(color='#d3d4d6')
        ),
        # Normalization toggle (buttons)
        dict(
            type='buttons',
            direction='right',
            x=0.15,
            xanchor='left',
            y=1.15,
            buttons=[
                dict(label='Absolute', method='update', args=[{'y': [y_absolute]}]),
                dict(label='Normalized', method='update', args=[{'y': [y_normalized]}])
            ],
            bgcolor='#1e293b',
            bordercolor='#334155',
            font=dict(color='#d3d4d6')
        )
    ]
)
```

### Updatemenu Method Reference

| Method | Purpose | Args Format |
|--------|---------|-------------|
| `'update'` | Update traces AND layout | `[trace_args, layout_args]` or `[trace_args]` |
| `'restyle'` | Update trace properties only | `[trace_args]` |
| `'relayout'` | Update layout properties only | `[layout_args]` |
| `'animate'` | Trigger animation | `[frame_args, animation_args]` |

---

## Reset Behavior

### Double-Click Reset

By default, double-clicking the plot area resets to the original view. Configure this behavior:

```python
# Customize double-click behavior
config = {
    'doubleClick': 'reset+autosize',  # Default: reset zoom and autosize
    # Options: 'reset', 'autosize', 'reset+autosize', False
}

fig.show(config=config)

# Or disable double-click reset
config = {'doubleClick': False}
```

### Reset Axis Button in Modebar

The modebar includes a "Reset axes" button by default. To ensure it's visible:

```python
config = {
    'modeBarButtonsToAdd': ['resetScale2d'],  # Explicitly add reset
}
```

### Custom Reset Button

Add an explicit reset button to the chart:

```python
# Store original ranges
original_x_range = [df['date'].min(), df['date'].max()]
original_y_range = [df['value'].min() * 0.95, df['value'].max() * 1.05]

fig.update_layout(
    updatemenus=[
        dict(
            type='buttons',
            direction='right',
            x=1,
            xanchor='right',
            y=1.15,
            yanchor='top',
            buttons=[
                dict(
                    label='Reset View',
                    method='relayout',
                    args=[{
                        'xaxis.range': original_x_range,
                        'yaxis.range': original_y_range,
                        'xaxis.autorange': True,
                        'yaxis.autorange': True
                    }]
                )
            ],
            bgcolor='#1e293b',
            bordercolor='#334155',
            font=dict(color='#d3d4d6')
        )
    ]
)
```

---

## Modebar Configuration

The modebar is the toolbar that appears when hovering over a Plotly chart.

### Default Modebar Buttons

Standard buttons (varies by chart type):

| Button | Purpose | When Useful |
|--------|---------|-------------|
| `zoom2d` | Box zoom | Scatter plots |
| `pan2d` | Pan mode | Time series, zoomed views |
| `select2d` | Box select | Data selection |
| `lasso2d` | Lasso select | Irregular selections |
| `zoomIn2d` | Zoom in | General |
| `zoomOut2d` | Zoom out | General |
| `autoScale2d` | Auto-scale axes | After zooming |
| `resetScale2d` | Reset to original | Always useful |
| `toImage` | Download as PNG | Export |
| `sendDataToCloud` | Plotly cloud | Usually remove |
| `hoverClosestCartesian` | Closest hover | Scatter |
| `hoverCompareCartesian` | Compare hover | Time series |
| `toggleSpikelines` | Spike lines | Time series |

### Customizing Modebar

```python
config = {
    'displayModeBar': True,          # Show modebar (True, False, 'hover')
    'displaylogo': False,            # Hide Plotly logo
    'modeBarButtonsToRemove': [
        'sendDataToCloud',
        'lasso2d',
        'select2d'
    ],
    'modeBarButtonsToAdd': [
        'drawline',
        'drawopenpath',
        'eraseshape'
    ],
    'toImageButtonOptions': {
        'format': 'png',             # 'png', 'svg', 'jpeg', 'webp'
        'filename': 'chart_export',
        'height': 600,
        'width': 1200,
        'scale': 2                   # Resolution multiplier
    }
}

fig.show(config=config)
```

### Modebar Configuration by Chart Type

#### Time Series

```python
TIME_SERIES_CONFIG = {
    'displayModeBar': True,
    'displaylogo': False,
    'modeBarButtonsToRemove': [
        'sendDataToCloud',
        'lasso2d',
        'select2d',
        'zoomIn2d',
        'zoomOut2d'
    ],
    'modeBarButtonsToAdd': [],
    'scrollZoom': True
}
```

#### Scatter Plot

```python
SCATTER_CONFIG = {
    'displayModeBar': True,
    'displaylogo': False,
    'modeBarButtonsToRemove': [
        'sendDataToCloud'
    ],
    'modeBarButtonsToAdd': [],
    'scrollZoom': True
}
```

#### Bar Chart (Minimal Interaction)

```python
BAR_CHART_CONFIG = {
    'displayModeBar': 'hover',       # Only show on hover
    'displaylogo': False,
    'modeBarButtonsToRemove': [
        'sendDataToCloud',
        'zoom2d',
        'pan2d',
        'select2d',
        'lasso2d',
        'zoomIn2d',
        'zoomOut2d',
        'autoScale2d',
        'resetScale2d'
    ],
    'modeBarButtonsToAdd': []
}
```

#### Dashboard Widget

```python
DASHBOARD_CONFIG = {
    'displayModeBar': False,         # Hide completely
    'displaylogo': False,
    'staticPlot': False              # Still allow hover
}
```

### Hiding Modebar Completely

```python
# Option 1: Config
config = {'displayModeBar': False}
fig.show(config=config)

# Option 2: Show on hover only
config = {'displayModeBar': 'hover'}
fig.show(config=config)
```

### Custom Modebar Button

```python
config = {
    'modeBarButtonsToAdd': [
        {
            'name': 'Custom Action',
            'icon': Plotly.Icons.pencil,  # Built-in icon
            'click': 'function(gd) { alert("Custom button clicked!"); }'
        }
    ]
}
```

---

## Disabling Interactivity (Static Mode)

For static exports, presentations, or embedded contexts where interaction is unwanted.

### When to Disable Interactivity

| Context | Disable Interaction | Reason |
|---------|---------------------|--------|
| PDF export | Yes | No interaction possible |
| Email embedding | Yes | Most clients block scripts |
| Print reports | Yes | Static medium |
| Presentation slides | Usually | Avoid accidental interaction |
| Small dashboard widgets | Sometimes | Limited space for interaction |
| Documentation | Usually | Cleaner appearance |

### Full Static Mode

```python
config = {
    'staticPlot': True  # Disables ALL interaction including hover
}

fig.show(config=config)
```

### Static with Hover Preserved

Keep hover tooltips but disable zoom/pan:

```python
fig.update_layout(
    dragmode=False,
    xaxis=dict(fixedrange=True),
    yaxis=dict(fixedrange=True)
)

config = {
    'displayModeBar': False,
    'scrollZoom': False
}

fig.show(config=config)
```

### For HTML Export

```python
# Static HTML export
fig.write_html(
    'chart.html',
    config={'staticPlot': True},
    include_plotlyjs='cdn'
)

# Interactive HTML export with limited controls
fig.write_html(
    'chart.html',
    config={
        'displayModeBar': False,
        'scrollZoom': False
    },
    include_plotlyjs='cdn'
)
```

### For Image Export

```python
# PNG export (always static)
fig.write_image('chart.png', width=1200, height=600, scale=2)

# SVG export (always static)
fig.write_image('chart.svg', width=1200, height=600)

# PDF export (always static)
fig.write_image('chart.pdf', width=1200, height=600)
```

---

## Interaction Decision Table by Chart Type

### Quick Reference

| Chart Type | Zoom | Pan | Range Slider | Range Buttons | Modebar |
|------------|------|-----|--------------|---------------|---------|
| **Time series (single)** | X-axis | Optional | Long series | Yes | Minimal |
| **Time series (multi)** | X-axis | Optional | Long series | Yes | Minimal |
| **Scatter plot** | Box (2D) | After zoom | No | No | Full |
| **Bubble chart** | Box (2D) | After zoom | No | No | Full |
| **Bar chart (vertical)** | Disabled | Disabled | No | No | Hidden |
| **Bar chart (horizontal)** | Disabled | Disabled | No | No | Hidden |
| **Heatmap** | Box (2D) | Optional | No | No | Minimal |
| **Pie/Donut** | Disabled | Disabled | No | No | Hidden |
| **Small multiples** | Linked | Disabled | No | Optional | Hidden |
| **Dashboard widget** | Disabled | Disabled | No | Optional | Hidden |

### Configuration Templates

#### Time Series Template

```python
def configure_time_series_interaction(fig, show_range_slider=False, show_range_buttons=True):
    """Configure interactions for time series charts."""

    range_buttons = [
        dict(count=7, label='1W', step='day', stepmode='backward'),
        dict(count=1, label='1M', step='month', stepmode='backward'),
        dict(count=3, label='3M', step='month', stepmode='backward'),
        dict(count=1, label='YTD', step='year', stepmode='todate'),
        dict(count=1, label='1Y', step='year', stepmode='backward'),
        dict(label='ALL', step='all')
    ]

    fig.update_layout(
        dragmode='zoom',
        xaxis=dict(
            fixedrange=False,
            rangeslider=dict(
                visible=show_range_slider,
                thickness=0.04,
                bgcolor='#0e1729',
                bordercolor='#334155'
            ),
            rangeselector=dict(
                visible=show_range_buttons,
                buttons=range_buttons,
                bgcolor='#1e293b',
                activecolor='#3B82F6',
                bordercolor='#334155',
                font=dict(color='#d3d4d6', size=11)
            ) if show_range_buttons else dict(visible=False)
        ),
        yaxis=dict(
            fixedrange=True,
            autorange=True
        )
    )

    config = {
        'displayModeBar': 'hover',
        'displaylogo': False,
        'modeBarButtonsToRemove': ['sendDataToCloud', 'lasso2d', 'select2d'],
        'scrollZoom': True
    }

    return fig, config
```

#### Scatter Plot Template

```python
def configure_scatter_interaction(fig, enable_selection=False):
    """Configure interactions for scatter plots."""

    fig.update_layout(
        dragmode='lasso' if enable_selection else 'zoom',
        xaxis=dict(fixedrange=False),
        yaxis=dict(fixedrange=False)
    )

    if enable_selection:
        fig.update_traces(
            selected=dict(marker=dict(color='#FBBF24', opacity=1)),
            unselected=dict(marker=dict(opacity=0.3))
        )

    config = {
        'displayModeBar': True,
        'displaylogo': False,
        'modeBarButtonsToRemove': ['sendDataToCloud'],
        'scrollZoom': True
    }

    return fig, config
```

#### Bar Chart Template

```python
def configure_bar_interaction(fig):
    """Configure interactions for bar charts (minimal)."""

    fig.update_layout(
        dragmode=False,
        xaxis=dict(fixedrange=True),
        yaxis=dict(fixedrange=True)
    )

    config = {
        'displayModeBar': False,
        'displaylogo': False,
        'staticPlot': False  # Preserve hover
    }

    return fig, config
```

#### Dashboard Widget Template

```python
def configure_dashboard_widget(fig, allow_legend_click=True):
    """Configure interactions for dashboard widgets."""

    fig.update_layout(
        dragmode=False,
        xaxis=dict(fixedrange=True),
        yaxis=dict(fixedrange=True)
    )

    if not allow_legend_click:
        fig.update_layout(
            legend=dict(
                itemclick=False,
                itemdoubleclick=False
            )
        )

    config = {
        'displayModeBar': False,
        'displaylogo': False,
        'staticPlot': False
    }

    return fig, config
```

---

## Complete Example

Full interaction configuration for a multi-series time series chart:

```python
import plotly.graph_objects as go
import pandas as pd

# Sample data
dates = pd.date_range('2023-01-01', periods=365, freq='D')
df = pd.DataFrame({
    'date': dates,
    'revenue': [100 + i * 0.5 + (i % 30) * 2 for i in range(365)],
    'profit': [50 + i * 0.3 + (i % 20) * 1.5 for i in range(365)]
})

# Create figure
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=df['date'],
    y=df['revenue'],
    name='Revenue',
    line=dict(color='#60A5FA', width=2),
    hovertemplate='%{y:$,.0f}<extra></extra>'
))

fig.add_trace(go.Scatter(
    x=df['date'],
    y=df['profit'],
    name='Profit',
    line=dict(color='#34D399', width=2),
    hovertemplate='%{y:$,.0f}<extra></extra>'
))

# Configure layout with interactions
fig.update_layout(
    template='plotly_dark',
    paper_bgcolor='#0e1729',
    plot_bgcolor='#0e1729',
    title=dict(
        text='Revenue and Profit Over Time',
        font=dict(color='#d3d4d6', size=16)
    ),

    # Hover configuration
    hovermode='x unified',
    hoverlabel=dict(
        bgcolor='#1e293b',
        font=dict(color='#d3d4d6', size=12),
        bordercolor='#334155'
    ),

    # Zoom configuration
    dragmode='zoom',

    # X-axis with range selector
    xaxis=dict(
        fixedrange=False,
        showgrid=False,  # Gridlines OFF by default
        gridcolor='rgba(255,255,255,0.1)',  # If enabled
        rangeselector=dict(
            buttons=[
                dict(count=1, label='1M', step='month', stepmode='backward'),
                dict(count=3, label='3M', step='month', stepmode='backward'),
                dict(count=6, label='6M', step='month', stepmode='backward'),
                dict(count=1, label='YTD', step='year', stepmode='todate'),
                dict(label='ALL', step='all')
            ],
            bgcolor='#1e293b',
            activecolor='#3B82F6',
            bordercolor='#334155',
            borderwidth=1,
            font=dict(color='#d3d4d6', size=11),
            x=0,
            xanchor='left',
            y=1.0,
            yanchor='bottom'
        ),
        rangeslider=dict(
            visible=True,
            thickness=0.04,
            bgcolor='#0e1729',
            bordercolor='#334155'
        )
    ),

    # Y-axis (fixed range, auto-scales)
    yaxis=dict(
        fixedrange=True,
        autorange=True,
        showgrid=False,  # Gridlines OFF by default
        gridcolor='rgba(255,255,255,0.1)',  # If enabled
        tickformat='$,.0f'
    ),

    # Legend
    legend=dict(
        orientation='h',
        yanchor='bottom',
        y=1.02,
        xanchor='left',
        x=0,
        font=dict(color='#d3d4d6', size=11)
    ),

    # Margins
    margin=dict(l=60, r=30, t=100, b=80)
)

# Configure modebar
config = {
    'displayModeBar': 'hover',
    'displaylogo': False,
    'modeBarButtonsToRemove': [
        'sendDataToCloud',
        'lasso2d',
        'select2d'
    ],
    'scrollZoom': True,
    'doubleClick': 'reset+autosize',
    'toImageButtonOptions': {
        'format': 'png',
        'filename': 'revenue_profit_chart',
        'height': 600,
        'width': 1200,
        'scale': 2
    }
}

fig.show(config=config)
```

---

## Interaction Checklist

Before finalizing interaction configuration:

### Zoom Behavior
- [ ] Time series uses X-axis only zoom (`yaxis.fixedrange=True`)
- [ ] Scatter plots use box zoom (both axes free)
- [ ] Bar charts have zoom disabled (`fixedrange=True` on both axes)
- [ ] Y-axis auto-scales when zooming X-axis (if desired)

### Pan Behavior
- [ ] Pan is disabled if range slider is present
- [ ] Pan is constrained to data bounds if enabled
- [ ] Scroll wheel zoom configured appropriately

### Range Controls
- [ ] Long time series (>6 months) has range slider OR range buttons
- [ ] Range buttons match expected analysis timeframes
- [ ] Default visible range is appropriate (not too zoomed out)
- [ ] Range selector styling matches theme (`bgcolor='#1e293b'`)

### Buttons and Menus
- [ ] Updatemenus positioned to not overlap chart
- [ ] Button styling matches theme
- [ ] Method type is correct (`update`, `relayout`, `restyle`)
- [ ] Visibility arrays match trace count

### Reset Behavior
- [ ] Double-click reset is enabled (or explicitly disabled if not wanted)
- [ ] Reset button included if significant zooming expected

### Modebar
- [ ] Plotly logo hidden (`displaylogo=False`)
- [ ] Unnecessary buttons removed (especially `sendDataToCloud`)
- [ ] `toImageButtonOptions` configured for high-quality export
- [ ] Modebar visibility appropriate for context (`True`, `False`, `'hover'`)

### Static Export
- [ ] Export images have sufficient resolution (`scale=2`)
- [ ] Static HTML uses appropriate config
- [ ] Hover preserved if static but interactive export

### Theme Consistency
- [ ] Range selector uses `bgcolor='#1e293b'`, `bordercolor='#334155'`
- [ ] Range selector active color is `#3B82F6`
- [ ] All button text uses `color='#d3d4d6'`
- [ ] Range slider background matches chart (`#0e1729`)
