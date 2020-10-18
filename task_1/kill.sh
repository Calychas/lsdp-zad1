#!/usr/bin/env bash

correct_pid=$(pgrep -f "$1" | head -n 1)
echo "Killed $correct_pid"
kill "$correct_pid"
