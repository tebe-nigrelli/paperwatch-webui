from pathlib import Path


def test_daily_timer_is_persistent():
    root = Path(__file__).parents[1]
    timer = (root / "systemd" / "paperwatch.timer").read_text()
    service = (root / "systemd" / "paperwatch.service").read_text()
    assert "OnCalendar=daily" in timer
    assert "Persistent=true" in timer
    assert "RandomizedDelaySec=30m" in timer
    assert "paperwatch" in service


def test_web_service_is_local_and_restarts():
    root = Path(__file__).parents[1]
    service = (root / "systemd" / "paperwatch-web.service").read_text()
    assert " web --host 127.0.0.1 --port 8765" in service
    assert "Restart=on-failure" in service
    assert "EnvironmentFile=" in service
    # The web service launches desktop applications via xdg-open. Starting it
    # from default.target can happen before Plasma imports DISPLAY/Wayland/DBus
    # variables into the systemd user manager, leaving it permanently headless.
    assert "PartOf=graphical-session.target" in service
    assert "WantedBy=graphical-session.target" in service
    assert "WantedBy=default.target" not in service
