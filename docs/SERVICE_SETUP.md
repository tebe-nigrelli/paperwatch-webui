# Running Paperwatch automatically

Paperwatch uses two commands:

- `paperwatch --config CONFIG web` is a long-running local web server.
- `paperwatch --config CONFIG run` fetches papers and blogs once, runs any automatic queries selected in the Web UI, and exits.

The examples below start the Web UI when you sign in and schedule one daily fetch. Replace paths and usernames where noted. Keep the Web UI bound to `127.0.0.1`; it has no login system.

Before installing automation, run both commands manually and confirm that your configuration, API keys, and model provider work.

## Linux: systemd

The repository includes user-level units in [`../systemd/`](../systemd/). They expect Paperwatch at `~/.local/share/paperwatch`, its configuration at `~/.config/paperwatch/config.toml`, and secrets in `~/.config/paperwatch/env`.

```bash
mkdir -p ~/.local/share ~/.config/paperwatch ~/.config/systemd/user
git clone https://github.com/tebe-nigrelli/paperwatch-webui.git ~/.local/share/paperwatch
cd ~/.local/share/paperwatch
python3 -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/paperwatch init ~/.config/paperwatch/config.toml
cp systemd/paperwatch.service systemd/paperwatch.timer systemd/paperwatch-web.service ~/.config/systemd/user/
printf '%s\n' 'OPENALEX_API_KEY=your-key' > ~/.config/paperwatch/env
chmod 600 ~/.config/paperwatch/env
systemctl --user daemon-reload
systemctl --user enable --now paperwatch.timer paperwatch-web.service
```

Open <http://127.0.0.1:8765>. Check the services with:

```bash
systemctl --user status paperwatch-web.service paperwatch.timer
systemctl --user list-timers paperwatch.timer
journalctl --user -u paperwatch.service -u paperwatch-web.service
```

The web unit is attached to `graphical-session.target` so actions such as **Open PDF** can reach the desktop session. The timer catches missed executions and randomizes its daily start by up to 30 minutes.

After updating the source, reinstall the editable package and restart the Web UI:

```bash
cd ~/.local/share/paperwatch
git pull --ff-only
.venv/bin/python -m pip install -e .
systemctl --user restart paperwatch-web.service
```

## macOS: launchd

The simplest setup uses two user LaunchAgents. In the examples below, replace every `/Users/YOU` with your real home directory. `launchd` does not expand `~` in plist values.

First place and install the application:

```bash
git clone https://github.com/tebe-nigrelli/paperwatch-webui.git ~/Library/Application\ Support/paperwatch
cd ~/Library/Application\ Support/paperwatch
python3 -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/paperwatch init config.toml
mkdir -p ~/Library/LaunchAgents ~/Library/Logs/paperwatch
```

Create `~/Library/LaunchAgents/com.paperwatch.web.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>com.paperwatch.web</string>
  <key>ProgramArguments</key>
  <array>
    <string>/Users/YOU/Library/Application Support/paperwatch/.venv/bin/paperwatch</string>
    <string>--config</string>
    <string>/Users/YOU/Library/Application Support/paperwatch/config.toml</string>
    <string>web</string>
  </array>
  <key>WorkingDirectory</key><string>/Users/YOU/Library/Application Support/paperwatch</string>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>StandardOutPath</key><string>/Users/YOU/Library/Logs/paperwatch/web.log</string>
  <key>StandardErrorPath</key><string>/Users/YOU/Library/Logs/paperwatch/web-error.log</string>
</dict>
</plist>
```

Create `~/Library/LaunchAgents/com.paperwatch.fetch.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>com.paperwatch.fetch</string>
  <key>ProgramArguments</key>
  <array>
    <string>/Users/YOU/Library/Application Support/paperwatch/.venv/bin/paperwatch</string>
    <string>--config</string>
    <string>/Users/YOU/Library/Application Support/paperwatch/config.toml</string>
    <string>run</string>
  </array>
  <key>WorkingDirectory</key><string>/Users/YOU/Library/Application Support/paperwatch</string>
  <key>StartCalendarInterval</key>
  <dict><key>Hour</key><integer>8</integer><key>Minute</key><integer>0</integer></dict>
  <key>StandardOutPath</key><string>/Users/YOU/Library/Logs/paperwatch/fetch.log</string>
  <key>StandardErrorPath</key><string>/Users/YOU/Library/Logs/paperwatch/fetch-error.log</string>
</dict>
</plist>
```

If you use an environment variable for the OpenAlex key, add this inside each plist's top-level `<dict>`:

```xml
<key>EnvironmentVariables</key>
<dict><key>OPENALEX_API_KEY</key><string>your-key</string></dict>
```

Protect the files and load them:

```bash
chmod 600 ~/Library/LaunchAgents/com.paperwatch.*.plist
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.paperwatch.web.plist
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.paperwatch.fetch.plist
launchctl kickstart -k gui/$(id -u)/com.paperwatch.web
```

Open <http://127.0.0.1:8765>. To unload the jobs, use `launchctl bootout gui/$(id -u) PATH_TO_PLIST`.

## Windows: Task Scheduler

Install Paperwatch in a stable directory such as `C:\Users\YOU\paperwatch`, initialize `config.toml`, and verify both commands in PowerShell first.

### Start the Web UI at sign-in

1. Open **Task Scheduler** and select **Create Task**.
2. Name it `Paperwatch Web` and choose **Run only when user is logged on**. This lets **Open PDF** use your desktop application.
3. Add the trigger **At log on** for your user.
4. Add the action **Start a program**:
   - Program: `C:\Users\YOU\paperwatch\.venv\Scripts\paperwatch.exe`
   - Arguments: `--config "C:\Users\YOU\paperwatch\config.toml" web`
   - Start in: `C:\Users\YOU\paperwatch`
5. In **Settings**, enable **Restart the task if it fails** and set **If the task is already running** to **Do not start a new instance**.

### Fetch once per day

Create a second task named `Paperwatch Fetch`:

1. Use a **Daily** trigger at the time you prefer and enable **Run task as soon as possible after a scheduled start is missed**.
2. Use the same program and working directory as above.
3. Set arguments to `--config "C:\Users\YOU\paperwatch\config.toml" run`.
4. Set **If the task is already running** to **Do not start a new instance**.

For unattended tasks, the most predictable place for the OpenAlex key is `config.toml`. If you put the key there, keep the file private and never commit it. A user-level environment variable also works, but Task Scheduler may require sign-out and sign-in before it sees a newly created variable:

```powershell
[Environment]::SetEnvironmentVariable('OPENALEX_API_KEY', 'your-key', 'User')
```

Run each task once from Task Scheduler, then open <http://127.0.0.1:8765> and check **Run** for fetch results.

## Backups and upgrades

Use **Data → Export whole data** before upgrades. The archive contains the SQLite database and PDF store. Configuration and API keys are separate and should be backed up securely.

The database schema migrates automatically when a newer Paperwatch version starts. After pulling an update, reinstall the editable package into the same virtual environment before restarting its service.
