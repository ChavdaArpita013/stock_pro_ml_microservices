# 📈 Buy/Sell Stock Prediction Microservice

A lightweight ML-powered microservice built using **FastAPI**, designed to predict BUY/HOLD decisions on stocks using **technical analysis indicators** like RSI and historical price data.

> 🎯 Integrated into a larger architecture with:
> - 📦 Spring Boot as the primary backend
> - ⚛️ React frontend interface
> - 🔮 Python ML engine as a microservice

---

## 🚀 Features

- ✅ Fetches live stock market data using `yfinance`
- 📊 Computes technical indicators (e.g., RSI)
- 🤖 Predicts Buy/Hold using a trained scikit-learn model
- 🔌 Exposes `/predict` REST endpoint (POST)
- 🛠 Easily callable from any backend (Spring Boot, Node.js, etc.)

---

## 📁 Project Structure

