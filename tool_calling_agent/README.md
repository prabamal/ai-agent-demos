#### 📂 `tool_calling_agent/README.md`
```markdown
# 🛠️ Tool-Calling Agent

## 📌 Overview
AI agent that dynamically calls external tools (APIs, DB, logs, notifications).

## 🎯 Problem
LLMs need real-world system integration. This project maps natural language → tool execution.

## 🛠️ Design
- ToolRegistry with tools: server status, alerts, restart service, scale app, DB query, notifications, logs
- ToolCallingAgent: analyzes intent, executes tools, generates summaries
- Features: retries, timeouts, permissions, chaining

## 🚀 How to Run
```bash
python tool_calling_agent.py
