@echo off
title KOSPI Trading Bot
cd /d E:\workspace\market
set PYTHONPATH=src
python scripts/trading_loop.py
pause
