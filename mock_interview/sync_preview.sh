#!/bin/bash
# The Claude Code preview launcher can't read ~/Documents (macOS privacy
# protection), so the preview runs a copy of this app from a stable location
# outside Documents. Run this script after editing app.py / problems.py /
# index.html to update that copy. Debriefs saved by the preview server land in
# the copy's sessions/ dir; mock_interview/sessions here symlinks to it.
set -e
SRC="$(cd "$(dirname "$0")" && pwd)"
DEST="$HOME/Library/Application Support/mock-interview"
mkdir -p "$DEST/sessions"
cp "$SRC/app.py" "$SRC/problems.py" "$SRC/rl_full_loop.py" "$SRC/remediation.py" "$SRC/resume_sessions.py" "$SRC/variants.py" "$SRC/index.html" "$SRC/gap_report.html" "$SRC/study.html" "$SRC/history.html" "$DEST/"
if [ ! -e "$SRC/sessions" ]; then
  ln -s "$DEST/sessions" "$SRC/sessions"
fi
# restart the always-on server (launchd agent) so python changes take effect
launchctl kickstart -k "gui/$(id -u)/com.yiminggu.mock-interview" 2>/dev/null || true
echo "synced -> $DEST"
