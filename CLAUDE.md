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
streamlit run app/main.py
```

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
