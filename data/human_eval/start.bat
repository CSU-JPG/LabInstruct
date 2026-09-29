@echo off
rem LabInstruct local eval server launcher - double-click to start
rem Opens http://localhost:8000/human_eval/survey.html in the browser.
rem Ratings are auto-saved to human_eval/ratings.json.
rem data.js is a build product and is not committed. --if-stale regenerates it
rem only when it is missing or older than specs/ and checklists/.
rem Close the new console window to stop the server.
cd /d "%~dp0"
node build_data.mjs --if-stale
start "LabInstruct Eval Server" cmd /k "chcp 65001 >nul & node server.mjs --open"
exit
