#!/bin/bash
cd "$(dirname "$0")" && bash installer/install-unix.sh
echo; read -n 1 -s -r -p "Press any key to close..."
