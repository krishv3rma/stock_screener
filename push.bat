@echo off
REM Double-click this to commit and push the README and RESULTS writeup.
cd /d "D:\stock screener"

echo.
echo === Staging README.md and RESULTS.md ===
git add README.md RESULTS.md

echo.
echo === Committing ===
git commit -m "Rewrite README, add backtest results writeup"

echo.
echo === Pushing to GitHub ===
git push

echo.
echo === Done. Check https://github.com/krishv3rma/stock_screener ===
pause
