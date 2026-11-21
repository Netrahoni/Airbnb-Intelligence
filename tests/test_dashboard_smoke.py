"""Streamlit AppTest smoke coverage for every dashboard workspace."""

from pathlib import Path
import unittest

from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).resolve().parents[1]
APP_PATH = ROOT / "app.py"
WORKSPACES = (
    "Executive Analytics",
    "Property Recommendations",
    "Price Prediction",
    "Data Analysis",
    "Model Evaluation",
)


class DashboardSmokeTests(unittest.TestCase):
    def test_every_workspace_renders_without_uncaught_exceptions(self):
        self.assertTrue(APP_PATH.is_file(), f"Dashboard entry point missing: {APP_PATH}")

        app = AppTest.from_file(str(APP_PATH), default_timeout=90).run()
        self.assertEqual(
            len(app.exception),
            0,
            "Dashboard failed during its initial render: "
            + "; ".join(str(item.value) for item in app.exception),
        )
        self.assertEqual(len(app.sidebar.radio), 1, "Workspace navigation did not render")

        for workspace in WORKSPACES:
            with self.subTest(workspace=workspace):
                app.sidebar.radio[0].set_value(workspace)
                app.run()
                self.assertEqual(
                    len(app.exception),
                    0,
                    f"{workspace} raised an uncaught exception: "
                    + "; ".join(str(item.value) for item in app.exception),
                )
                self.assertEqual(
                    app.sidebar.radio[0].value,
                    workspace,
                    f"Navigation did not select {workspace}",
                )


if __name__ == "__main__":
    unittest.main()
