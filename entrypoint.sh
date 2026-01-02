#!/bin/sh
set -e

# Ensure /data directory has correct ownership for appuser (uid 1000)
# This is needed because Docker volumes are created as root
if [ -d /data ]; then
    chown -R 1000:1000 /data 2>/dev/null || true
fi

# Execute the main command
exec "$@"
