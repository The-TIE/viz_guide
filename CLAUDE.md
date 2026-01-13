# Visualization Guide Project

This project contains a Plotly visualization guide and an interactive tester app.

## Project Structure

- `guide/` - 14-section visualization guide (markdown)
- `app/` - Streamlit tester app
- `feedback/` - Collected feedback (JSONL)
- `audit/` - Dashboard audit notes

## Commands

### /review-feedback

Review collected feedback and propose guide improvements.

**Usage:** Just say `/review-feedback` or "review the feedback"

**What it does:**
1. Reads `feedback/feedback.jsonl`
2. Analyzes patterns (low ratings, common themes)
3. Proposes specific guide edits
4. Applies changes after your approval

**Feedback fields:**
- `rating` (1-5)
- `guide_suggestions` - free-text improvement ideas
- `generated_code` - the code that was generated
- `description` - the visualization prompt used

### Running the App

```bash
cd /home/quantfiction/repositories/viz_guide
source .venv/bin/activate
streamlit run app/main.py --server.headless true
```

The `--server.headless true` flag prevents Streamlit from prompting for email input on first run.

## Guide Sections

1. Data Analysis
2. Intent Classification
3. Chart Type Selection
4. Data Encoding
5. Axis Configuration
6. Color
7. Text Formatting
8. Layout
9. Legends
10. Hover & Tooltips
11. Interactions
12. Annotations
13. Accessibility
14. Performance

---

## Roadmap

### Phase 1: SDK Migration (COMPLETED)

Converted from direct Anthropic API to Claude Agent SDK (keyless mode).

**Files changed:**
- `app/mcp_tools.py` (new) - MCP server with @tool-decorated functions
- `app/agent.py` - Async SDK pattern
- `app/refine.py` - Async SDK pattern
- `app/generator.py` - Async SDK pattern (legacy mode)
- `app/suggest.py` - Async SDK pattern
- `app/main.py` - Removed API key UI
- `.env` - API key commented out
- `pyproject.toml` - Added claude-agent-sdk, anyio

### Phase 2: Standardized Visualization Templates (COMPLETED)

Created reusable Python template functions in `app/templates/`:

**Files:**
- `base.py` - Utilities (`format_with_B`, annotations) + imports from `token_labs.visualization.plotly`
- `line.py` - `line_chart()`, `multi_line_chart()`, `small_multiples_chart()`
- `bar.py` - `bar_chart()`, `horizontal_bar_chart()`, `stacked_bar_chart()`, `grouped_bar_chart()`
- `detector.py` - `detect_chart_type()` AST-based detection from code

**Usage:**
```python
from app.templates import multi_line_chart, horizontal_bar_chart

fig = multi_line_chart(df, 'date', ['BTC', 'ETH'], 'Price Comparison', normalize=True)
fig = horizontal_bar_chart(df, 'exchange', 'volume', 'Top Exchanges', sort=True)
```

Agent prompt now includes template function documentation. Check generated code for `from app.templates import` to see if template was used.

### Phase 2.5: Template Refinement Feedback Loop

**Goal:** When a template-based visualization is refined, feed improvements back into the template code itself (not just guide markdown).

**Current behavior:**
- Refinements modify the specific code instance
- "Suggest" proposes guide markdown changes
- Templates remain static

**Desired behavior:**
1. Detect if generated code used a template (`from app.templates import`)
2. When refinement feedback is applied, determine if it's:
   - A template parameter issue → update default params or add new param
   - A template code issue → update the template function itself
3. "Validate" re-generates using the updated template deterministically

**Implementation:**
1. Create `app/template_feedback.py`:
   - `analyze_template_refinement(session)` - Determine what template changes are needed
   - `apply_template_fix(template_id, fix)` - Modify `app/templates/*.py`

2. Modify `app/suggest.py`:
   - Check if template was used
   - Route feedback to template code vs guide markdown accordingly

3. Update refinement UI:
   - Show "Template used: multi_line_chart" indicator
   - After approval, option to "Apply to template" vs "Keep as one-off"

### Phase 3: Viz Polish Mode

New UI mode to refine existing Plotly code to match best practices.

**Goal:** User pastes existing Plotly code, agent "polishes" it to current best practices. If it fits a template from Phase 2, convert to that template.

**Implementation:**

1. Create `app/polish.py`:
   - `polish_visualization()` function
   - Detects chart type from existing code
   - Matches to template if applicable
   - Applies best practices via agent

2. Update `app/main.py`:
   - Add mode selector: "Generate" vs "Polish"
   - Polish mode UI:
     - Text area for pasting existing code
     - Optional description field
     - Side-by-side original/polished comparison
     - Shows detected chart type and matched template

**Polish rules to apply:**
- Dark theme colors (plot_bgcolor=#0e1729, etc.)
- Replace `tickformat=',.2s'` with `format_with_B()`
- Remove "Date" labels from time series x-axis
- Use unified hover mode with proper hovertemplate
- Position legend to avoid overlap with title
- Add source/updated annotations
- Ensure proper margins based on content
