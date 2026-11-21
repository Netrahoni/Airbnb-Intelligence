"""Small, framework-independent helpers for dashboard widget state."""


def reset_widget_state(state, keys):
    """Remove selected widget values so Streamlit restores declared defaults."""
    for key in keys:
        state.pop(key, None)
