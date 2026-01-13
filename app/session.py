"""Session management for iterative refinement workflow."""

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


SESSIONS_DIR = Path(__file__).parent.parent / "feedback" / "sessions"


@dataclass
class Iteration:
    """A single iteration in the refinement process."""

    version: int
    code: str
    feedback: str | None  # None for the final approved version
    timestamp: str  # ISO format string for JSON serialization

    @classmethod
    def create(cls, version: int, code: str, feedback: str | None = None) -> "Iteration":
        """Create a new iteration with current timestamp."""
        return cls(
            version=version,
            code=code,
            feedback=feedback,
            timestamp=datetime.utcnow().isoformat(),
        )


@dataclass
class ConflictFix:
    """A conflicting pattern to fix when applying a suggestion."""

    pattern: str  # Regex pattern to find (e.g., "showgrid=True")
    replacement: str | None  # Simple replacement, or None if needs manual review
    description: str  # Human-readable description of the conflict


@dataclass
class GuideSuggestion:
    """A suggested improvement to the visualization guide."""

    file: str  # e.g., "09_legends.md"
    section: str  # e.g., "Legend Positioning"
    content: str  # The suggested text to add
    reason: str  # Why this suggestion was made
    applied: bool = False  # Whether user applied it
    conflicts: list[ConflictFix] | None = None  # Conflicting patterns to fix


@dataclass
class TemplateSuggestion:
    """A suggested improvement to a template function."""

    template_id: str  # e.g., "multi_line_chart"
    template_file: str  # e.g., "line.py"
    category: str  # "parameter" | "template_code"
    description: str  # Human-readable description
    suggested_code: str  # The fix to apply
    reason: str  # Why this change is needed
    applied: bool = False  # Whether user applied it
    # For parameter changes only
    param_name: str | None = None
    param_type: str | None = None
    param_default: str | None = None


@dataclass
class RefinementSession:
    """A complete refinement session from initial generation to approval."""

    id: str
    started_at: str  # ISO format
    description: str  # Original visualization request
    data_sample: str  # Data used for generation
    watermark: str  # Watermark setting
    iterations: list[Iteration] = field(default_factory=list)
    status: str = "in_progress"  # "in_progress" | "approved" | "abandoned"
    approved_at: str | None = None
    capture_path: str | None = None  # Path to saved figure image
    guide_suggestions: list[GuideSuggestion] | None = None
    # Template tracking (Phase 2.5)
    used_template: str | None = None  # e.g., "multi_line_chart"
    template_file: str | None = None  # e.g., "line.py"
    template_suggestions: list[TemplateSuggestion] | None = None

    @classmethod
    def create(
        cls,
        description: str,
        data_sample: str,
        watermark: str = "none",
    ) -> "RefinementSession":
        """Create a new refinement session."""
        return cls(
            id=str(uuid.uuid4())[:8],
            started_at=datetime.utcnow().isoformat(),
            description=description,
            data_sample=data_sample,
            watermark=watermark,
        )

    def add_iteration(self, code: str, feedback: str | None = None) -> Iteration:
        """Add a new iteration to the session."""
        version = len(self.iterations) + 1
        iteration = Iteration.create(version=version, code=code, feedback=feedback)
        self.iterations.append(iteration)
        return iteration

    def add_feedback_to_latest(self, feedback: str) -> None:
        """Add feedback to the latest iteration."""
        if self.iterations:
            self.iterations[-1].feedback = feedback

    def get_current_code(self) -> str | None:
        """Get the code from the latest iteration."""
        if self.iterations:
            return self.iterations[-1].code
        return None

    def get_iteration_history(self) -> str:
        """Format iteration history for context in prompts."""
        if not self.iterations:
            return "No previous iterations."

        if len(self.iterations) <= 1:
            return "This is the first iteration."

        history_parts = []
        history_parts.append("IMPORTANT: The following issues have already been addressed. Do NOT revert these fixes:\n")

        for it in self.iterations[:-1]:  # Exclude current iteration
            if it.feedback:
                history_parts.append(f"- Version {it.version} feedback: \"{it.feedback}\" → FIXED in version {it.version + 1}")

        # Also include the previous version's code so model can see what changed
        if len(self.iterations) >= 2:
            prev_code = self.iterations[-2].code
            history_parts.append(f"\n\nPREVIOUS CODE (version {len(self.iterations) - 1}) for reference:")
            history_parts.append(f"```python\n{prev_code}\n```")

        return "\n".join(history_parts)

    def approve(self, capture_path: str | None = None) -> None:
        """Mark session as approved."""
        self.status = "approved"
        self.approved_at = datetime.utcnow().isoformat()
        if capture_path:
            self.capture_path = capture_path

    def abandon(self) -> None:
        """Mark session as abandoned."""
        self.status = "abandoned"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RefinementSession":
        """Create from dictionary."""
        # Convert nested structures
        iterations = [Iteration(**it) for it in data.get("iterations", [])]
        suggestions = None
        if data.get("guide_suggestions"):
            suggestions = []
            for s in data["guide_suggestions"]:
                # Handle nested ConflictFix objects
                conflicts = None
                if s.get("conflicts"):
                    conflicts = [ConflictFix(**c) for c in s["conflicts"]]
                suggestions.append(GuideSuggestion(
                    file=s["file"],
                    section=s["section"],
                    content=s["content"],
                    reason=s["reason"],
                    applied=s.get("applied", False),
                    conflicts=conflicts,
                ))

        # Parse template suggestions (Phase 2.5)
        template_suggestions = None
        if data.get("template_suggestions"):
            template_suggestions = []
            for ts in data["template_suggestions"]:
                template_suggestions.append(TemplateSuggestion(
                    template_id=ts["template_id"],
                    template_file=ts["template_file"],
                    category=ts["category"],
                    description=ts["description"],
                    suggested_code=ts["suggested_code"],
                    reason=ts["reason"],
                    applied=ts.get("applied", False),
                    param_name=ts.get("param_name"),
                    param_type=ts.get("param_type"),
                    param_default=ts.get("param_default"),
                ))

        return cls(
            id=data["id"],
            started_at=data["started_at"],
            description=data["description"],
            data_sample=data["data_sample"],
            watermark=data["watermark"],
            iterations=iterations,
            status=data["status"],
            approved_at=data.get("approved_at"),
            capture_path=data.get("capture_path"),
            guide_suggestions=suggestions,
            used_template=data.get("used_template"),
            template_file=data.get("template_file"),
            template_suggestions=template_suggestions,
        )


class SessionManager:
    """Manages refinement sessions with file-based persistence."""

    def __init__(self, sessions_dir: Path | None = None):
        """Initialize the session manager."""
        self.sessions_dir = sessions_dir or SESSIONS_DIR
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

    def _session_path(self, session_id: str) -> Path:
        """Get the file path for a session."""
        return self.sessions_dir / f"{session_id}.json"

    def save_session(self, session: RefinementSession) -> None:
        """Save a session to disk."""
        path = self._session_path(session.id)
        with open(path, "w") as f:
            json.dump(session.to_dict(), f, indent=2)

    def load_session(self, session_id: str) -> RefinementSession | None:
        """Load a session from disk."""
        path = self._session_path(session_id)
        if not path.exists():
            return None
        with open(path) as f:
            data = json.load(f)
        return RefinementSession.from_dict(data)

    def create_session(
        self,
        description: str,
        data_sample: str,
        watermark: str = "none",
    ) -> RefinementSession:
        """Create and save a new session."""
        session = RefinementSession.create(
            description=description,
            data_sample=data_sample,
            watermark=watermark,
        )
        self.save_session(session)
        return session

    def add_iteration(
        self,
        session_id: str,
        code: str,
        feedback: str | None = None,
    ) -> Iteration | None:
        """Add an iteration to an existing session."""
        session = self.load_session(session_id)
        if not session:
            return None
        iteration = session.add_iteration(code, feedback)
        self.save_session(session)
        return iteration

    def add_feedback(self, session_id: str, feedback: str) -> bool:
        """Add feedback to the latest iteration."""
        session = self.load_session(session_id)
        if not session:
            return False
        session.add_feedback_to_latest(feedback)
        self.save_session(session)
        return True

    def approve_session(self, session_id: str, capture_path: str | None = None) -> bool:
        """Mark a session as approved."""
        session = self.load_session(session_id)
        if not session:
            return False
        session.approve(capture_path=capture_path)
        self.save_session(session)
        return True

    def abandon_session(self, session_id: str) -> bool:
        """Mark a session as abandoned."""
        session = self.load_session(session_id)
        if not session:
            return False
        session.abandon()
        self.save_session(session)
        return True

    def set_suggestions(
        self,
        session_id: str,
        suggestions: list[GuideSuggestion],
    ) -> bool:
        """Set guide suggestions for a session."""
        session = self.load_session(session_id)
        if not session:
            return False
        session.guide_suggestions = suggestions
        self.save_session(session)
        return True

    def mark_suggestion_applied(
        self,
        session_id: str,
        suggestion_index: int,
    ) -> bool:
        """Mark a suggestion as applied."""
        session = self.load_session(session_id)
        if not session or not session.guide_suggestions:
            return False
        if 0 <= suggestion_index < len(session.guide_suggestions):
            session.guide_suggestions[suggestion_index].applied = True
            self.save_session(session)
            return True
        return False

    def set_template_info(
        self,
        session_id: str,
        template_id: str,
        template_file: str,
    ) -> bool:
        """Set template info for a session (Phase 2.5)."""
        session = self.load_session(session_id)
        if not session:
            return False
        session.used_template = template_id
        session.template_file = template_file
        self.save_session(session)
        return True

    def set_template_suggestions(
        self,
        session_id: str,
        suggestions: list[TemplateSuggestion],
    ) -> bool:
        """Set template suggestions for a session (Phase 2.5)."""
        session = self.load_session(session_id)
        if not session:
            return False
        session.template_suggestions = suggestions
        self.save_session(session)
        return True

    def mark_template_suggestion_applied(
        self,
        session_id: str,
        suggestion_index: int,
    ) -> bool:
        """Mark a template suggestion as applied (Phase 2.5)."""
        session = self.load_session(session_id)
        if not session or not session.template_suggestions:
            return False
        if 0 <= suggestion_index < len(session.template_suggestions):
            session.template_suggestions[suggestion_index].applied = True
            self.save_session(session)
            return True
        return False

    def list_sessions(
        self,
        status: str | None = None,
        limit: int = 50,
    ) -> list[RefinementSession]:
        """List all sessions, optionally filtered by status."""
        sessions = []
        for path in sorted(self.sessions_dir.glob("*.json"), reverse=True):
            try:
                with open(path) as f:
                    data = json.load(f)
                session = RefinementSession.from_dict(data)
                if status is None or session.status == status:
                    sessions.append(session)
                if len(sessions) >= limit:
                    break
            except (json.JSONDecodeError, KeyError):
                continue
        return sessions

    def delete_session(self, session_id: str) -> bool:
        """Delete a session file."""
        path = self._session_path(session_id)
        if path.exists():
            path.unlink()
            return True
        return False


# Singleton instance
_manager: SessionManager | None = None


def get_session_manager() -> SessionManager:
    """Get or create the singleton session manager."""
    global _manager
    if _manager is None:
        _manager = SessionManager()
    return _manager


if __name__ == "__main__":
    # Test the session manager
    manager = SessionManager()

    # Create a test session
    session = manager.create_session(
        description="Compare BTC and ETH prices over time",
        data_sample="date, BTC_price, ETH_price\n2024-01-01, 42000, 2200",
        watermark="labs",
    )
    print(f"Created session: {session.id}")

    # Add an iteration
    manager.add_iteration(
        session.id,
        code="import plotly.graph_objects as go\nfig = go.Figure()",
    )
    print("Added iteration 1")

    # Add feedback
    manager.add_feedback(session.id, "Legend is overlapping the title")
    print("Added feedback")

    # Add another iteration
    manager.add_iteration(
        session.id,
        code="import plotly.graph_objects as go\nfig = go.Figure()\nfig.update_layout(legend=dict(y=1.1))",
    )
    print("Added iteration 2")

    # Load and inspect
    loaded = manager.load_session(session.id)
    if loaded:
        print(f"\nLoaded session: {loaded.id}")
        print(f"Status: {loaded.status}")
        print(f"Iterations: {len(loaded.iterations)}")
        for it in loaded.iterations:
            print(f"  v{it.version}: feedback={it.feedback}")

    # List sessions
    all_sessions = manager.list_sessions()
    print(f"\nTotal sessions: {len(all_sessions)}")

    # Clean up test
    manager.delete_session(session.id)
    print(f"Deleted test session")
