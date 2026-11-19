from pathlib import Path

from streamlit.testing.v1 import AppTest

DASHBOARD = str(Path(__file__).resolve().parents[1] / "dashboard" / "dashboard.py")


def test_dashboard_filters_and_prediction():
    at = AppTest.from_file(DASHBOARD, default_timeout=120).run()
    assert not at.exception
    assert [m.value for m in at.metric][0] == "7,043"

    at.sidebar.multiselect[0].set_value(["Month-to-month"]).run()
    assert not at.exception
    assert [m.value for m in at.metric][0] == "3,875"

    at.button[0].click().run()
    assert not at.exception
    results = [e.value for e in at.error] + [s.value for s in at.success]
    assert any("Churn probability" in r for r in results)


def test_dashboard_with_empty_selection():
    at = AppTest.from_file(DASHBOARD, default_timeout=120).run()
    at.sidebar.multiselect[0].set_value([]).run()
    assert not at.exception
    assert any("No customers match" in w.value for w in at.warning)
