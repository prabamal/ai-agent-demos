# ⚡ Event-Driven Agent Orchestration

## 📌 Overview
Implements an event-driven orchestration system where multiple AI agents collaborate via an event bus. Features retries, DLQ handling, metrics, and workflow correlation.

## 🎯 Problem
Traditional IT Ops monitoring systems lack coordination and resilience. This project shows how agents can work together using an event-driven model.

## 🛠️ Design
- EventBus with multiple queues (`alerts`, `metrics`, `deployments`, etc.)
- Agents: AlertCorrelation, MetricEvaluator, RemediationPlanner, DeploymentWatcher, WorkflowManager, UserAction, AuditLogger
- Automatic retries + dead-letter handling

## 🚀 How to Run
```bash
python event_driven_agent_orchestration.py
