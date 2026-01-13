"""Interactive Visualization Guide Tester - Streamlit App."""

import os
import re
import traceback
from pathlib import Path

from dotenv import load_dotenv
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from data_utils import dataframe_to_sample, generate_sample_data, load_file
from agent import generate_visualization as generate_agent
from generator import generate_visualization as generate_legacy
from random_prompts import get_random_prompt
from session import get_session_manager, RefinementSession
from refine import refine_visualization
from suggest import analyze_session, apply_suggestion, validate_improvement
from guide_index import reload_guide_index
from template_feedback import detect_template_usage, apply_template_suggestion
import templates  # For exec() context

# Load .env file from project root
load_dotenv(Path(__file__).parent.parent / ".env")

# Page config
st.set_page_config(
    page_title="Viz Guide Tester",
    page_icon="📊",
    layout="wide",
)

# Custom CSS for dark theme consistency
st.markdown("""
<style>
    .stApp {
        background-color: #0e1729;
    }
    .stTextArea textarea {
        background-color: #1a2332;
        color: #d3d4d6;
    }
    .stSelectbox, .stSlider {
        color: #d3d4d6;
    }
    code {
        background-color: #1a2332 !important;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.title("📊 Visualization Guide Tester")
st.markdown("*Test the Plotly visualization guide by generating charts from descriptions*")

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")

    # Note: API key no longer needed - uses Claude Code CLI auth (keyless mode)

    # Model selection
    model = st.selectbox(
        "Model",
        options=[
            "claude-sonnet-4-5-20250929",
            "claude-haiku-4-5-20251001",
            "claude-opus-4-5-20251101",
        ],
        index=0,
        help="Select the Claude model to use"
    )

    # Temperature
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.1,
        help="Lower = more deterministic, Higher = more creative"
    )

    st.divider()

    # Agent mode toggle
    st.header("🤖 Generation Mode")
    use_agent = st.toggle(
        "Use Agent Mode",
        value=True,
        help="Agent mode uses tools for progressive context loading (~80% fewer tokens)"
    )

    if use_agent:
        st.caption("✅ Agent mode: Loads only relevant guide sections")
    else:
        st.caption("⚠️ Legacy mode: Loads entire guide (~60k tokens)")

    st.divider()

    # Watermark selection
    st.header("🏷️ Watermark")
    watermark = st.selectbox(
        "Watermark",
        options=["none", "labs", "qf", "tie"],
        index=0,
        format_func=lambda x: {
            "none": "None",
            "labs": "TIE Labs",
            "qf": "QuantFiction",
            "tie": "The TIE",
        }.get(x, x),
        help="Add a watermark logo to generated charts",
        label_visibility="collapsed"
    )

    st.divider()

    # Session info
    st.header("📊 Session")
    session_manager = get_session_manager()
    sessions = session_manager.list_sessions(limit=5)
    if sessions:
        st.caption(f"{len(sessions)} recent sessions")
        approved = len([s for s in sessions if s.status == "approved"])
        st.caption(f"{approved} approved")

    st.divider()

    # Guide reload
    st.header("📚 Guide")
    if st.button("🔄 Reload Guide", help="Reload guide index after manual edits", use_container_width=True):
        reload_guide_index()
        st.success("Guide index reloaded!")

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📝 Input")

    # Description input
    st.subheader("Visualization Description")

    description = st.text_area(
        "What do you want to visualize?",
        value=st.session_state.get("description", "Show Bitcoin price over the last 6 months with volume as secondary axis"),
        height=100,
        placeholder="Describe the visualization you want to create...",
        label_visibility="collapsed"
    )

    # Random prompt button
    if st.button("🎲 Random Prompt", help="Generate a random visualization description", use_container_width=True):
        with st.spinner("Generating prompt..."):
            st.session_state["description"] = get_random_prompt()
        st.rerun()

    # Data input section
    st.subheader("Data")

    data_source = st.radio(
        "Data Source",
        options=["Generate Sample", "Upload File"],
        horizontal=True,
        label_visibility="collapsed"
    )

    df = None

    if data_source == "Upload File":
        uploaded_file = st.file_uploader(
            "Upload CSV or JSON",
            type=["csv", "json"],
            help="Upload your data file"
        )
        if uploaded_file:
            try:
                df = load_file(uploaded_file, uploaded_file.name)
                st.success(f"Loaded {len(df)} rows")
            except Exception as e:
                st.error(f"Error loading file: {e}")

    else:  # Generate Sample
        if st.button("🔄 Generate Data for Current Description", help="Generate synthetic data based on description above"):
            with st.spinner("Generating sample data..."):
                df = generate_sample_data(description)
                st.session_state["generated_df"] = df

        if "generated_df" in st.session_state:
            df = st.session_state["generated_df"]

    # Show data preview
    if df is not None:
        with st.expander("📋 Data Preview", expanded=True):
            st.dataframe(df.head(10), use_container_width=True)
            st.caption(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")

    # Note: Uses Claude Code CLI auth (keyless mode) - no API key needed

with col2:
    st.header("📊 Output")

    # Generate button at top of output column
    generate_clicked = st.button(
        "✨ Generate Visualization",
        type="primary",
        use_container_width=True,
        disabled=df is None
    )

    # Process generation
    if generate_clicked and df is not None:
        spinner_text = "Generating visualization (agent mode)..." if use_agent else "Generating visualization..."
        with st.spinner(spinner_text):
            try:
                data_sample = dataframe_to_sample(df)

                # Create a new refinement session
                session_manager = get_session_manager()
                session = session_manager.create_session(
                    description=description,
                    data_sample=data_sample,
                    watermark=watermark,
                )
                st.session_state["current_session_id"] = session.id

                if use_agent:
                    result = generate_agent(
                        description=description,
                        data_sample=data_sample,
                        model=model,
                        temperature=temperature,
                        watermark=watermark,
                    )
                    st.session_state["tool_calls"] = result.get("tool_calls", [])
                    st.session_state["agent_turns"] = result.get("turns", 0)
                    st.session_state["raw_response"] = result.get("raw_response", "")
                else:
                    result = generate_legacy(
                        description=description,
                        data_sample=data_sample,
                        model=model,
                        temperature=temperature,
                    )
                    st.session_state["tool_calls"] = []
                    st.session_state["agent_turns"] = 0
                    st.session_state["raw_response"] = ""

                # Add first iteration to session
                session_manager.add_iteration(session.id, result["code"])

                # Detect template usage (Phase 2.5)
                template_id, template_file = detect_template_usage(result["code"])
                if template_id:
                    session_manager.set_template_info(session.id, template_id, template_file)

                st.session_state["generated_code"] = result["code"]
                st.session_state["generation_success"] = True
                st.session_state["generation_error"] = None

            except Exception as e:
                st.session_state["generation_success"] = False
                st.session_state["generation_error"] = str(e)
                st.session_state["generated_code"] = None
                st.session_state["tool_calls"] = []

    # Display results
    if st.session_state.get("generation_error"):
        st.error(f"Generation failed: {st.session_state['generation_error']}")

    # Display capture error if any
    if st.session_state.get("capture_error"):
        st.error(f"Capture failed: {st.session_state['capture_error']}")
        st.session_state.pop("capture_error", None)  # Clear after showing

    if st.session_state.get("generated_code"):
        code = st.session_state["generated_code"]

        # Only render if data is available
        if df is None:
            st.warning("Load data to render the visualization")
            st.session_state["render_success"] = False
        else:
            # Try to execute and render chart
            try:
                # Create execution namespace with data
                exec_globals = {
                    "pd": pd,
                    "go": go,
                    "np": __import__("numpy"),
                    "df": df,
                    "make_subplots": __import__("plotly.subplots", fromlist=["make_subplots"]).make_subplots,
                    "Image": __import__("PIL", fromlist=["Image"]).Image,
                    "Path": __import__("pathlib").Path,
                }

                # Remove fig.show() calls that would open new browser tabs
                code_cleaned = re.sub(r'fig\.show\([^)]*\)', '# fig.show() removed', code)

                # Fix template imports for exec context (app.templates -> templates)
                code_cleaned = re.sub(r'from app\.templates', 'from templates', code_cleaned)

                # Execute the code
                exec(code_cleaned, exec_globals)

                # Get the figure
                if "fig" in exec_globals:
                    fig = exec_globals["fig"]
                    st.plotly_chart(fig, use_container_width=True)
                    st.session_state["render_success"] = True
                else:
                    st.error("Code did not create a 'fig' variable")
                    st.session_state["render_success"] = False

            except Exception as e:
                st.error(f"Error rendering chart: {e}")
                with st.expander("Error Details"):
                    st.code(traceback.format_exc())
                st.session_state["render_success"] = False

        # Code display below chart
        with st.expander("💻 Generated Code", expanded=False):
            st.code(code, language="python")

        # Tool calls display (agent mode only)
        tool_calls = st.session_state.get("tool_calls", [])
        if tool_calls:
            with st.expander(f"🔧 Agent Tool Calls ({len(tool_calls)})", expanded=False):
                agent_turns = st.session_state.get("agent_turns", 0)
                st.caption(f"Completed in {agent_turns} turn(s)")
                for i, tc in enumerate(tool_calls, 1):
                    st.markdown(f"**{i}. {tc['name']}**")
                    st.json(tc["input"])

        # Raw response display (for debugging template decisions)
        raw_response = st.session_state.get("raw_response")
        if raw_response:
            with st.expander("🔍 Agent Raw Response", expanded=False):
                st.text(raw_response[:3000] + ("..." if len(raw_response) > 3000 else ""))

# Refinement section
st.divider()
st.header("🔄 Refinement")

session_id = st.session_state.get("current_session_id")
if session_id and st.session_state.get("generated_code"):
    session_manager = get_session_manager()
    current_session = session_manager.load_session(session_id)

    if current_session:
        # Show iteration history
        if len(current_session.iterations) > 1:
            with st.expander("📜 Iteration History", expanded=False):
                for it in current_session.iterations:
                    col_v, col_f = st.columns([1, 3])
                    with col_v:
                        st.markdown(f"**v{it.version}**")
                    with col_f:
                        if it.feedback:
                            st.caption(f"💬 {it.feedback}")
                        else:
                            st.caption("✅ Current version")

        # Check if session is approved
        if current_session.status == "approved":
            st.success("✅ Visualization approved!")

            # Show template info if used (Phase 2.5)
            if current_session.used_template:
                st.info(f"🧩 Template used: `{current_session.used_template}` from `{current_session.template_file}`")

            # Show guide suggestions if available
            if current_session.guide_suggestions:
                st.subheader("📚 Suggested Guide Improvements")

                # Count applied vs pending
                applied_count = sum(1 for s in current_session.guide_suggestions if s.applied)
                pending_count = len(current_session.guide_suggestions) - applied_count
                if applied_count > 0:
                    st.caption(f"✅ {applied_count} applied, {pending_count} pending")

                # Initialize checkbox states - default pending ones to True
                for i, suggestion in enumerate(current_session.guide_suggestions):
                    key = f"suggestion_{session_id}_{i}"
                    if key not in st.session_state:
                        st.session_state[key] = not suggestion.applied  # True for pending

                for i, suggestion in enumerate(current_session.guide_suggestions):
                    key = f"suggestion_{session_id}_{i}"
                    if suggestion.applied:
                        # Already applied - show as disabled checked
                        st.checkbox(
                            f"✅ **{suggestion.file}** - {suggestion.section}",
                            value=True,
                            key=key,
                            disabled=True,
                        )
                    else:
                        # Not applied - use session state (defaults to True)
                        st.checkbox(
                            f"**{suggestion.file}** - {suggestion.section}",
                            key=key,
                        )
                    st.caption(f"_{suggestion.reason}_")
                    with st.expander("View suggestion"):
                        st.markdown(suggestion.content)
                        if suggestion.conflicts:
                            st.markdown("---")
                            st.markdown("**Conflicts to fix:**")
                            for conflict in suggestion.conflicts:
                                fix_type = "🔧 Auto-fix" if conflict.replacement else "👁️ Manual review"
                                st.markdown(f"- {fix_type}: `{conflict.pattern}` → {conflict.description}")

                # Only show buttons if there are pending suggestions
                if pending_count > 0:
                    col_all, col_selected, col_validate = st.columns(3)
                    with col_all:
                        if st.button("Apply All Pending", type="primary", use_container_width=True):
                            applied_now = 0
                            errors = []
                            messages = []
                            for i, suggestion in enumerate(current_session.guide_suggestions):
                                if not suggestion.applied:
                                    success, message = apply_suggestion(suggestion)
                                    if success:
                                        session_manager.mark_suggestion_applied(session_id, i)
                                        applied_now += 1
                                        messages.append(f"{suggestion.file}: {message}")
                                    else:
                                        errors.append(f"{suggestion.file}: {message}")
                            if applied_now:
                                reload_guide_index()  # Reload so agent sees changes
                                st.success(f"Applied {applied_now} suggestion(s) to guide! Index reloaded.")
                                for msg in messages:
                                    if "Manual review" in msg:
                                        st.warning(msg)
                                    elif "Auto-fixed" in msg:
                                        st.info(msg)
                            if errors:
                                st.error(f"Failed to apply: {', '.join(errors)}")
                            if applied_now or errors:
                                st.rerun()
                    with col_selected:
                        if st.button("Apply Selected", use_container_width=True):
                            applied_now = 0
                            errors = []
                            messages = []
                            for i, suggestion in enumerate(current_session.guide_suggestions):
                                key = f"suggestion_{session_id}_{i}"
                                if st.session_state.get(key, False) and not suggestion.applied:
                                    success, message = apply_suggestion(suggestion)
                                    if success:
                                        session_manager.mark_suggestion_applied(session_id, i)
                                        applied_now += 1
                                        messages.append(f"{suggestion.file}: {message}")
                                    else:
                                        errors.append(f"{suggestion.file}: {message}")
                            if applied_now:
                                reload_guide_index()  # Reload so agent sees changes
                                st.success(f"Applied {applied_now} suggestion(s) to guide! Index reloaded.")
                                for msg in messages:
                                    if "Manual review" in msg:
                                        st.warning(msg)
                                    elif "Auto-fixed" in msg:
                                        st.info(msg)
                            if errors:
                                st.error(f"Failed to apply: {', '.join(errors)}")
                            if applied_now or errors:
                                st.rerun()
                            elif not applied_now and not errors:
                                st.warning("No suggestions selected or all already applied.")
                else:
                    col_all, col_selected, col_validate = st.columns(3)

                with col_validate:
                    if st.button("Validate", help="Re-run original task with updated guide", use_container_width=True):
                        with st.spinner("Re-running with updated guide..."):
                            validation = validate_improvement(
                                current_session,
                                model=model,
                                watermark=watermark,
                            )
                            st.session_state["validation_result"] = validation
                            st.rerun()

            # Show template suggestions if available (Phase 2.5)
            if current_session.template_suggestions:
                st.subheader("🔧 Suggested Template Improvements")

                # Count applied vs pending
                applied_count = sum(1 for s in current_session.template_suggestions if s.applied)
                pending_count = len(current_session.template_suggestions) - applied_count
                if applied_count > 0:
                    st.caption(f"✅ {applied_count} applied, {pending_count} pending")

                for i, suggestion in enumerate(current_session.template_suggestions):
                    if suggestion.applied:
                        st.checkbox(
                            f"✅ **{suggestion.template_id}** - {suggestion.description}",
                            value=True,
                            disabled=True,
                            key=f"template_suggestion_{session_id}_{i}",
                        )
                    else:
                        st.markdown(f"**{suggestion.template_id}** - {suggestion.description}")

                    st.caption(f"_Category: {suggestion.category} | {suggestion.reason}_")

                    with st.expander("View suggestion"):
                        st.code(suggestion.suggested_code, language="python")

                        if not suggestion.applied:
                            col_apply, col_skip = st.columns(2)
                            with col_apply:
                                if st.button("Apply to Template", key=f"apply_template_{session_id}_{i}", use_container_width=True):
                                    success, message = apply_template_suggestion(suggestion)
                                    if success:
                                        session_manager.mark_template_suggestion_applied(session_id, i)
                                        st.success(message)
                                    else:
                                        st.warning(message)
                                    st.rerun()
                            with col_skip:
                                if st.button("Keep as One-Off", key=f"skip_template_{session_id}_{i}", use_container_width=True):
                                    # Mark as applied without actually applying
                                    session_manager.mark_template_suggestion_applied(session_id, i)
                                    st.info("Change kept as one-off, not applied to template")
                                    st.rerun()

            # Show Analyze button if no suggestions have been generated yet
            if not current_session.guide_suggestions and not current_session.template_suggestions:
                if len(current_session.iterations) > 1:
                    # Generate suggestions if not already done
                    if st.button("🔍 Analyze for Improvements", use_container_width=True):
                        with st.spinner("Analyzing session for improvements..."):
                            guide_suggestions, template_suggestions = analyze_session(current_session)
                            if guide_suggestions:
                                session_manager.set_suggestions(session_id, guide_suggestions)
                            if template_suggestions:
                                session_manager.set_template_suggestions(session_id, template_suggestions)
                            if guide_suggestions or template_suggestions:
                                st.rerun()
                            else:
                                st.info("No improvements suggested for this session.")

            # Show validation result if available
            if st.session_state.get("validation_result"):
                validation = st.session_state["validation_result"]
                st.subheader("Validation Result")
                st.json(validation.get("comparison", {}))

            # Start new session button
            if st.button("🆕 Start New Session", use_container_width=True):
                st.session_state.pop("current_session_id", None)
                st.session_state.pop("generated_code", None)
                st.session_state.pop("validation_result", None)
                st.rerun()

        else:
            # Active refinement UI
            st.markdown(f"**Version {len(current_session.iterations)}** - Provide feedback or approve")

            feedback_input = st.text_area(
                "What needs to change?",
                placeholder="List issues to fix, e.g.:\n- The legend is overlapping the title\n- Use B for billions instead of G\n- Add unified hover for all series",
                height=100,
                key="feedback_input",
            )

            col_refine, col_approve, col_abandon = st.columns([2, 1, 1])

            with col_refine:
                if st.button("🔄 Request Changes", use_container_width=True, disabled=not feedback_input):
                    if feedback_input:
                        # Add feedback to current iteration
                        session_manager.add_feedback(session_id, feedback_input)

                        # Refine the visualization
                        with st.spinner("Refining visualization..."):
                            current_session = session_manager.load_session(session_id)
                            result = refine_visualization(
                                session=current_session,
                                feedback=feedback_input,
                                model=model,
                            )

                            # Add new iteration
                            session_manager.add_iteration(session_id, result["code"])
                            st.session_state["generated_code"] = result["code"]
                            st.session_state["tool_calls"] = result.get("tool_calls", [])

                        st.rerun()

            with col_approve:
                if st.button("✅ Approve", type="primary", use_container_width=True):
                    # Save figure capture before approving
                    capture_path = None
                    if df is None:
                        st.warning("No data loaded - capture not saved. Reload data before approving to save capture.")
                    elif not st.session_state.get("generated_code"):
                        st.warning("No generated code - capture not saved.")
                    else:
                        try:
                            code = st.session_state["generated_code"]
                            exec_globals = {
                                "pd": pd,
                                "go": go,
                                "np": __import__("numpy"),
                                "df": df,
                                "make_subplots": __import__("plotly.subplots", fromlist=["make_subplots"]).make_subplots,
                                "Image": __import__("PIL", fromlist=["Image"]).Image,
                                "Path": __import__("pathlib").Path,
                                "datetime": __import__("datetime"),
                                "math": __import__("math"),
                            }
                            code_cleaned = re.sub(r'fig\.show\([^)]*\)', '', code)
                            code_cleaned = re.sub(r'from app\.templates', 'from templates', code_cleaned)
                            exec(code_cleaned, exec_globals)
                            if "fig" in exec_globals:
                                # Use resolve() to ensure absolute path
                                captures_dir = Path(__file__).resolve().parent.parent / "feedback" / "captures"
                                captures_dir.mkdir(parents=True, exist_ok=True)
                                capture_path = captures_dir / f"{session_id}.png"
                                exec_globals["fig"].write_image(str(capture_path), scale=2)
                                # Verify file was created
                                if not capture_path.exists():
                                    st.session_state["capture_error"] = f"write_image completed but file not found at {capture_path}"
                                    capture_path = None
                            else:
                                st.session_state["capture_error"] = "Code did not produce a figure - capture not saved."
                        except Exception as e:
                            import traceback
                            st.session_state["capture_error"] = f"Could not save capture: {e}\n{traceback.format_exc()}"

                    session_manager.approve_session(session_id, capture_path=str(capture_path) if capture_path else None)
                    st.success(f"Visualization approved!{' Saved to captures/' + session_id + '.png' if capture_path else ''}")
                    st.rerun()

            with col_abandon:
                if st.button("🗑️ Abandon", use_container_width=True):
                    session_manager.abandon_session(session_id)
                    st.session_state.pop("current_session_id", None)
                    st.session_state.pop("generated_code", None)
                    st.rerun()

else:
    st.info("Generate a visualization to start a refinement session")

# Footer
st.divider()
st.caption("Visualization Guide Tester | Powered by Claude & Plotly")
