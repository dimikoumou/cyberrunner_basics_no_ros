#!/bin/bash
# Waits for the U2D2 USB-serial adapter to reappear (it can drop mid-session
# from the violent-shake vibration -- a known issue, not a code bug), then
# launches the full pd_balance.py run automatically. No need to babysit this
# and re-trigger the run by hand once the cable's reseated.
set -e
cd "$(dirname "$0")"
echo "waiting for /dev/tty.usbserial-* to reappear..."
while ! ls /dev/tty.usbserial-* >/dev/null 2>&1; do
    sleep 2
done
echo "found $(ls /dev/tty.usbserial-*), waiting 3s for it to settle..."
sleep 3
echo "launching pd_balance.py"
exec ../.venv/bin/python3 -u pd_balance.py 800 10
