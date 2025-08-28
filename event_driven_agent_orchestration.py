# Event-Driven Agent Orchestration System
# (saved from notebook cell)
from __future__ import annotations
import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Callable, Union, Coroutine
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from enum import Enum
import uuid
from collections import defaultdict, deque
import time
import random

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("orchestrator")

class EventType(Enum):
    SYSTEM_ALERT = "system.alert"
    METRIC_THRESHOLD = "metrics.threshold_exceeded"
    SERVICE_STATUS = "service.status_changed"
    DEPLOYMENT = "deployment.completed"
    USER_ACTION = "user.action_requested"
    AGENT_RESPONSE = "agent.response"
    WORKFLOW_STATUS = "workflow.status_changed"
    ERROR = "system.error"

class EventPriority(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class Event:
    id: str
    event_type: EventType
    source: str
    timestamp: datetime
    priority: EventPriority
    payload: Dict[str, Any]
    correlation_id: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    ttl_seconds: int = 300

    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())
        if self.correlation_id is None:
            self.correlation_id = self.id

    def is_expired(self) -> bool:
        return (datetime.now() - self.timestamp).total_seconds() > self.ttl_seconds

    def can_retry(self) -> bool:
        return self.retry_count < self.max_retries

class MockMessageQueue:
    def __init__(self, name: str):
        self.name = name
        self.queue = deque()
        self.dead_letter_queue = deque()
        self.subscribers: List[Callable[[Event], Coroutine[Any, Any, None]]] = []
        self.metrics = {
            "messages_published": 0,
            "messages_consumed": 0,
            "messages_failed": 0,
            "dlq_published": 0,
        }

    async def publish(self, event: Event, routing_key: str = ""):
        if event.is_expired():
            logger.warning(f"[{self.name}] Event {event.id} expired, moving to DLQ")
            self.dead_letter_queue.append(event)
            self.metrics["dlq_published"] += 1
            return
        self.queue.append(event)
        self.metrics["messages_published"] += 1
        asyncio.create_task(self._notify_subscribers(event))

    async def consume(self, timeout: float = 1.0) -> Optional[Event]:
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.queue:
                event = self.queue.popleft()
                self.metrics["messages_consumed"] += 1
                return event
            await asyncio.sleep(0.01)
        return None

    async def subscribe(self, callback: Callable[[Event], Coroutine[Any, Any, None]]):
        self.subscribers.append(callback)

    async def _notify_subscribers(self, event: Event):
        for callback in self.subscribers:
            try:
                await callback(event)
            except Exception as e:
                self.metrics["messages_failed"] += 1
                logger.error(f"[{self.name}] Subscriber callback failed: {e}")

    def get_metrics(self) -> Dict[str, Any]:
        return {
            "queue_name": self.name,
            "queue_size": len(self.queue),
            "dlq_size": len(self.dead_letter_queue),
            **self.metrics,
        }

class EventBus:
    def __init__(self):
        self.queues: Dict[str, MockMessageQueue] = {}
        self.event_history: List[Event] = []
        self.routing_rules: Dict[str, Callable[[Event], List[str]]] = {}
        self._create_default_queues()
        self._setup_default_routing()

    def _create_default_queues(self):
        for q in ["alerts", "metrics", "deployments", "user_actions", "agent_responses", "workflows", "errors"]:
            self.queues[q] = MockMessageQueue(q)

    def _setup_default_routing(self):
        def route_by_event_type(event: Event) -> List[str]:
            routing_map = {
                EventType.SYSTEM_ALERT: ["alerts"],
                EventType.METRIC_THRESHOLD: ["metrics", "alerts"],
                EventType.SERVICE_STATUS: ["alerts", "workflows"],
                EventType.DEPLOYMENT: ["deployments", "workflows"],
                EventType.USER_ACTION: ["user_actions"],
                EventType.AGENT_RESPONSE: ["agent_responses", "workflows"],
                EventType.WORKFLOW_STATUS: ["workflows"],
                EventType.ERROR: ["errors"],
            }
            return routing_map.get(event.event_type, ["errors"])

        def fanout_for_critical(event: Event) -> List[str]:
            if event.priority == EventPriority.CRITICAL:
                return list(self.queues.keys())
            return route_by_event_type(event)

        self.routing_rules["default"] = fanout_for_critical

    async def publish(self, event: Event):
        self.event_history.append(event)
        queues = self.routing_rules["default"](event)
        for qname in queues:
            await self.queues[qname].publish(event)

    async def subscribe(self, queue_name: str, callback: Callable[[Event], Coroutine[Any, Any, None]]):
        await self.queues[queue_name].subscribe(callback)

    def metrics_snapshot(self) -> Dict[str, Any]:
        return {name: q.get_metrics() for name, q in self.queues.items()}

@dataclass
class EventSubscription:
    event_types: List[EventType]
    agent_id: str
    priority_filter: Optional[EventPriority] = None
    source_filter: Optional[str] = None
    active: bool = True

class BaseAgent:
    def __init__(self, agent_id: str, bus: EventBus):
        self.agent_id = agent_id
        self.bus = bus
        self.subscriptions: List[EventSubscription] = []
        self.handled_events = 0
        self.failures = 0

    async def on_event(self, event: Event):
        if not self._is_interested(event):
            return
        try:
            await self.handle_event(event)
            self.handled_events += 1
        except Exception as e:
            self.failures += 1
            logger.exception(f"[{self.agent_id}] error handling event {event.id}: {e}")
            await self.bus.publish(
                Event(
                    id=str(uuid.uuid4()),
                    event_type=EventType.ERROR,
                    source=self.agent_id,
                    timestamp=datetime.now(),
                    priority=EventPriority.HIGH,
                    payload={"failed_event_id": event.id, "error": str(e), "agent": self.agent_id},
                    correlation_id=event.correlation_id,
                )
            )
            if event.can_retry():
                event.retry_count += 1
                await asyncio.sleep(0.1 * min(2 ** event.retry_count, 10))
                await self.bus.publish(event)
            else:
                routed = list(self.bus.routing_rules["default"](event))
                if routed:
                    self.bus.queues[routed[0]].dead_letter_queue.append(event)
                    self.bus.queues[routed[0]].metrics["dlq_published"] += 1

    def _is_interested(self, event: Event) -> bool:
        for sub in self.subscriptions:
            if not sub.active:
                continue
            if event.event_type not in sub.event_types:
                continue
            if sub.priority_filter and event.priority.value < sub.priority_filter.value:
                continue
            if sub.source_filter and sub.source_filter != event.source:
                continue
            return True
        return False

    async def handle_event(self, event: Event):
        raise NotImplementedError

    def subscribe(self, queue_name: str, event_types: List[EventType], priority_filter: Optional[EventPriority] = None, source_filter: Optional[str] = None):
        self.subscriptions.append(EventSubscription(event_types, self.agent_id, priority_filter, source_filter))
        asyncio.create_task(self.bus.subscribe(queue_name, self.on_event))

class AlertCorrelationAgent(BaseAgent):
    async def handle_event(self, event: Event):
        if event.event_type == EventType.SYSTEM_ALERT:
            alert = event.payload
            cpu = alert.get("cpu_percent", 0)
            if cpu >= 90 or alert.get("severity") == "critical":
                await self.bus.publish(
                    Event(
                        id=str(uuid.uuid4()),
                        event_type=EventType.METRIC_THRESHOLD,
                        source=self.agent_id,
                        timestamp=datetime.now(),
                        priority=EventPriority.HIGH,
                        payload={"metric": "cpu_percent", "value": cpu, "threshold": 90, "node": alert.get("node", "unknown")},
                        correlation_id=event.correlation_id,
                    )
                )
            await self.bus.publish(
                Event(
                    id=str(uuid.uuid4()),
                    event_type=EventType.WORKFLOW_STATUS,
                    source=self.agent_id,
                    timestamp=datetime.now(),
                    priority=EventPriority.MEDIUM,
                    payload={"status": "alert_received", "alert_id": alert.get("id")},
                    correlation_id=event.correlation_id,
                )
            )

class MetricEvaluatorAgent(BaseAgent):
    async def handle_event(self, event: Event):
        if event.event_type == EventType.METRIC_THRESHOLD:
            metric = event.payload
            value = metric.get("value", 0)
            threshold = metric.get("threshold", 100)
            if value >= threshold:
                plan = {"action": "scale_up" if value > threshold * 1.2 else "restart_service", "target": metric.get("node", "unknown")}
                await self.bus.publish(
                    Event(
                        id=str(uuid.uuid4()),
                        event_type=EventType.AGENT_RESPONSE,
                        source=self.agent_id,
                        timestamp=datetime.now(),
                        priority=EventPriority.HIGH,
                        payload={"plan": plan, "reason": "threshold_exceeded"},
                        correlation_id=event.correlation_id,
                    )
                )

class RemediationPlannerAgent(BaseAgent):
    async def handle_event(self, event: Event):
        if event.event_type == EventType.AGENT_RESPONSE:
            plan = event.payload.get("plan", {})
            await asyncio.sleep(0.2)
            import random
            success = random.random() > 0.15
            result = {"plan": plan, "success": success, "details": "Scaled instances" if plan.get("action") == "scale_up" else "Service restarted"}
            await self.bus.publish(
                Event(
                    id=str(uuid.uuid4()),
                    event_type=EventType.WORKFLOW_STATUS,
                    source=self.agent_id,
                    timestamp=datetime.now(),
                    priority=EventPriority.MEDIUM if success else EventPriority.HIGH,
                    payload={"status": "remediation_done" if success else "remediation_failed", "result": result},
                    correlation_id=event.correlation_id,
                )
            )

class DeploymentWatcherAgent(BaseAgent):
    async def handle_event(self, event: Event):
        if event.event_type == EventType.DEPLOYMENT:
            payload = event.payload
            await asyncio.sleep(0.1)
            await self.bus.publish(
                Event(
                    id=str(uuid.uuid4()),
                    event_type=EventType.WORKFLOW_STATUS,
                    source=self.agent_id,
                    timestamp=datetime.now(),
                    priority=EventPriority.LOW,
                    payload={"status": "post_deploy_checks_started", "deploy_id": payload.get("deploy_id")},
                    correlation_id=event.correlation_id,
                )
            )

class WorkflowManagerAgent(BaseAgent):
    def __init__(self, agent_id: str, bus: EventBus):
        super().__init__(agent_id, bus)
        self.state: Dict[str, Dict[str, Any]] = defaultdict(dict)

    async def handle_event(self, event: Event):
        if event.event_type == EventType.WORKFLOW_STATUS:
            wf = self.state[event.correlation_id]
            wf["updates"] = wf.get("updates", 0) + 1
            wf["last"] = event.payload.get("status")
            if wf["last"] == "remediation_failed":
                await self.bus.publish(
                    Event(
                        id=str(uuid.uuid4()),
                        event_type=EventType.USER_ACTION,
                        source=self.agent_id,
                        timestamp=datetime.now(),
                        priority=EventPriority.HIGH,
                        payload={"action": "approve_scale_out", "reason": "auto remediation failed", "suggested_instances": 2},
                        correlation_id=event.correlation_id,
                    )
                )

class UserActionAgent(BaseAgent):
    async def handle_event(self, event: Event):
        if event.event_type == EventType.USER_ACTION:
            req = event.payload
            await asyncio.sleep(0.15)
            await self.bus.publish(
                Event(
                    id=str(uuid.uuid4()),
                    event_type=EventType.AGENT_RESPONSE,
                    source=self.agent_id,
                    timestamp=datetime.now(),
                    priority=EventPriority.MEDIUM,
                    payload={"plan": {"action": "scale_up", "instances": req.get("suggested_instances", 2)}, "approved": True},
                    correlation_id=event.correlation_id,
                )
            )

class AuditLoggerAgent(BaseAgent):
    def __init__(self, agent_id: str, bus: EventBus):
        super().__init__(agent_id, bus)
        self.audit_trail: List[Dict[str, Any]] = []

    async def handle_event(self, event: Event):
        if event.event_type == EventType.ERROR:
            self.audit_trail.append({"when": datetime.now().isoformat(), "correlation_id": event.correlation_id, "error": event.payload})
            logger.warning(f"[{self.agent_id}] AUDIT ERROR: {event.payload}")

async def simulate_incoming_events(bus: EventBus, duration_sec: int = 5):
    start = time.time()
    i = 0
    while time.time() - start < duration_sec:
        i += 1
        corr = str(uuid.uuid4())
        import random
        if random.random() > 0.35:
            cpu = random.randint(40, 100)
            evt = Event(
                id=str(uuid.uuid4()),
                event_type=EventType.SYSTEM_ALERT,
                source="monitoring.web",
                timestamp=datetime.now(),
                priority=EventPriority.CRITICAL if cpu >= 95 else (EventPriority.HIGH if cpu >= 90 else EventPriority.MEDIUM),
                payload={"id": f"alert-{i}", "node": f"web-{random.randint(1,3)}", "cpu_percent": cpu, "severity": "high" if cpu >= 90 else "medium"},
                correlation_id=corr,
                ttl_seconds=120,
            )
        else:
            evt = Event(
                id=str(uuid.uuid4()),
                event_type=EventType.DEPLOYMENT,
                source="ci.cd",
                timestamp=datetime.now(),
                priority=EventPriority.MEDIUM,
                payload={"deploy_id": f"deploy-{i}", "service": "api", "version": f"v{random.randint(1,3)}.{random.randint(0,9)}.{random.randint(0,9)}"},
                correlation_id=corr,
                ttl_seconds=120,
            )
        await bus.publish(evt)
        await asyncio.sleep(0.1)

async def main_demo(runtime_sec: int = 6):
    bus = EventBus()
    a1 = AlertCorrelationAgent("alert-correlation", bus); a1.subscribe("alerts", [EventType.SYSTEM_ALERT], priority_filter=EventPriority.MEDIUM)
    a2 = MetricEvaluatorAgent("metric-evaluator", bus); a2.subscribe("metrics", [EventType.METRIC_THRESHOLD])
    a3 = RemediationPlannerAgent("remediation-planner", bus); a3.subscribe("agent_responses", [EventType.AGENT_RESPONSE])
    a4 = DeploymentWatcherAgent("deployment-watcher", bus); a4.subscribe("deployments", [EventType.DEPLOYMENT])
    a5 = WorkflowManagerAgent("workflow-manager", bus); a5.subscribe("workflows", [EventType.WORKFLOW_STATUS])
    a6 = UserActionAgent("user-action", bus); a6.subscribe("user_actions", [EventType.USER_ACTION])
    a7 = AuditLoggerAgent("audit-logger", bus); a7.subscribe("errors", [EventType.ERROR])

    producer = asyncio.create_task(simulate_incoming_events(bus, duration_sec=runtime_sec))
    await producer
    await asyncio.sleep(1.0)

    print("\n" + "=" * 70)
    print("📊 QUEUE METRICS")
    print("=" * 70)
    for name, snapshot in bus.metrics_snapshot().items():
        print(f"\nQueue: {name}")
        for k, v in snapshot.items():
            print(f"  {k}: {v}")

    print("\n" + "=" * 70)
    print("🧾 DEMO SUMMARY")
    print("=" * 70)
    total_events = sum(q.metrics["messages_published"] for q in bus.queues.values())
    total_consumed = sum(q.metrics["messages_consumed"] for q in bus.queues.values())
    total_dlq = sum(q.metrics["dlq_published"] for q in bus.queues.values())
    print(f"Total published: {total_events}")
    print(f"Total consumed:  {total_consumed}")
    print(f"Total to DLQ:   {total_dlq}")
    print("\n✅ Event-Driven Orchestration demo completed.\n")

if __name__ == "__main__":
    try:
        asyncio.run(main_demo(runtime_sec=6))
    except RuntimeError:
        loop = asyncio.get_event_loop()
        loop.run_until_complete(main_demo(runtime_sec=6))
