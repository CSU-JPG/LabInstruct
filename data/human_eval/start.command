#!/bin/bash
# LabInstruct local eval server launcher (macOS) - double-click to start
# First run only: chmod +x start.command  (in Terminal)
# data.js is a build product and is not committed. --if-stale regenerates it
# only when it is missing or older than specs/ and checklists/.
cd "$(dirname "$0")"
node build_data.mjs --if-stale
node server.mjs --open
