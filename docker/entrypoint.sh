#!/bin/bash
set -euo pipefail

mkdir -p /var/run/sshd /root/.ssh
chmod 700 /root/.ssh


# Generate host keys if missing
if [ ! -f /etc/ssh/ssh_host_rsa_key ]; then
  ssh-keygen -A
fi

# Start sshd in background (needed for Runpod direct TCP / Cursor Remote-SSH)
/usr/sbin/sshd

exec "$@"
