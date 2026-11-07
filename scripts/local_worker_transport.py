#!/usr/bin/env python3
"""SSH-shaped local transport for this project's explicitly named local hosts."""
import os
from pathlib import Path
import shlex
import socket
import sys


def main():
    arguments = sys.argv[1:]
    environment = None
    if arguments[:1] == ['--env']:
        if len(arguments) < 4:
            raise SystemExit('--env requires an absolute environment file, host and command')
        environment = Path(arguments[1])
        if not environment.is_absolute() or not environment.is_file():
            raise SystemExit('Local worker environment must be an existing absolute file')
        arguments = arguments[2:]
    hosts = {'local-v05': 'ZJU3DV-V05', 'local-v02': 'ZJU3DV-V02'}
    if len(arguments) < 2 or hosts.get(arguments[0]) != socket.gethostname():
        raise SystemExit('Local worker transport requires its matching physical host and a command')
    # SSH receives one shell command from the runner/MCP client; rsync supplies
    # a sequence of server arguments. Preserve both forms and streaming stdio.
    command = arguments[1] if len(arguments) == 2 else shlex.join(arguments[1:])
    if environment is not None:
        command = 'set -e\nsource ' + shlex.quote(str(environment)) + '\n' + command
    os.execv('/bin/bash', ['bash', '-lc', command])


if __name__ == '__main__':
    main()
