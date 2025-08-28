# Multi-Agent ITOps Monitoring System
# Demonstrates supervisor-worker pattern for autonomous IT operations

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import time
from datetime import datetime

# Mock dependencies - in real implementation you'd use:
# from langchain.agents import AgentExecutor
# from langchain.tools import Tool
# from openai import OpenAI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AlertSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class AlertStatus(Enum):
    OPEN = "open"
    IN_PROGRESS = "in_progress" 
    RESOLVED = "resolved"
    ESCALATED = "escalated"

@dataclass
class SystemAlert:
    id: str
    timestamp: datetime
    severity: AlertSeverity
    source: str
    message: str
    metrics: Dict[str, Any]
    status: AlertStatus = AlertStatus.OPEN
    assigned_agent: Optional[str] = None
    resolution_steps: List[str] = None
    
    def __post_init__(self):
        if self.resolution_steps is None:
            self.resolution_steps = []

class AgentCapability(Enum):
    MONITORING = "monitoring"
    ANALYSIS = "analysis"
    REMEDIATION = "remediation"
    ESCALATION = "escalation"

class BaseAgent:
    def __init__(self, name: str, capabilities: List[AgentCapability]):
        self.name = name
        self.capabilities = capabilities
        self.active_tasks = []
        self.memory = {}
        
    async def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Override in subclasses"""
        raise NotImplementedError
        
    def log_action(self, action: str, details: Dict[str, Any] = None):
        timestamp = datetime.now().isoformat()
        log_entry = {
            "timestamp": timestamp,
            "agent": self.name,
            "action": action,
            "details": details or {}
        }
        logger.info(f"[{self.name}] {action}: {details}")
        return log_entry

class MonitoringAgent(BaseAgent):
    def __init__(self):
        super().__init__("MonitoringAgent", [AgentCapability.MONITORING])
        self.thresholds = {
            "cpu_usage": 85.0,
            "memory_usage": 90.0,
            "disk_usage": 95.0,
            "response_time": 5000  # ms
        }
        
    async def collect_metrics(self, system: str) -> Dict[str, float]:
        """Simulate metric collection"""
        await asyncio.sleep(0.5)  # Simulate network delay
        
        # Mock metrics - in reality would call monitoring APIs
        import random
        metrics = {
            "cpu_usage": random.uniform(30, 95),
            "memory_usage": random.uniform(40, 95),
            "disk_usage": random.uniform(20, 98),
            "response_time": random.uniform(100, 8000)
        }
        
        self.log_action("metrics_collected", {"system": system, "metrics": metrics})
        return metrics
    
    async def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        system = task.get("system")
        metrics = await self.collect_metrics(system)
        
        alerts = []
        for metric_name, value in metrics.items():
            threshold = self.thresholds.get(metric_name)
            if threshold and value > threshold:
                severity = self._determine_severity(metric_name, value, threshold)
                alert = SystemAlert(
                    id=f"{system}_{metric_name}_{int(time.time())}",
                    timestamp=datetime.now(),
                    severity=severity,
                    source=system,
                    message=f"{metric_name} threshold exceeded: {value:.1f}% (threshold: {threshold}%)",
                    metrics=metrics
                )
                alerts.append(alert)
                
        return {"system": system, "alerts": alerts, "metrics": metrics}
    
    def _determine_severity(self, metric: str, value: float, threshold: float) -> AlertSeverity:
        if value > threshold * 1.2:
            return AlertSeverity.CRITICAL
        elif value > threshold * 1.1:
            return AlertSeverity.HIGH
        else:
            return AlertSeverity.MEDIUM

class AnalysisAgent(BaseAgent):
    def __init__(self):
        super().__init__("AnalysisAgent", [AgentCapability.ANALYSIS])
        self.knowledge_base = {
            "cpu_usage": {
                "common_causes": ["High load processes", "Memory leaks", "Inefficient queries"],
                "investigation_steps": ["Check top processes", "Analyze CPU patterns", "Review recent deployments"]
            },
            "memory_usage": {
                "common_causes": ["Memory leaks", "Large datasets", "Caching issues"],
                "investigation_steps": ["Check memory allocation", "Analyze heap dumps", "Review garbage collection"]
            }
        }
    
    async def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        alert = task.get("alert")
        if not alert:
            return {"error": "No alert provided"}
            
        # Extract metric type from alert message
        metric_type = self._extract_metric_type(alert.message)
        
        analysis = await self._analyze_alert(alert, metric_type)
        
        self.log_action("alert_analyzed", {
            "alert_id": alert.id,
            "analysis": analysis
        })
        
        return {
            "alert_id": alert.id,
            "analysis": analysis,
            "recommended_actions": analysis.get("recommended_actions", []),
            "escalation_needed": analysis.get("escalation_needed", False)
        }
    
    def _extract_metric_type(self, message: str) -> str:
        for metric in ["cpu_usage", "memory_usage", "disk_usage", "response_time"]:
            if metric.replace("_", " ") in message.lower():
                return metric
        return "unknown"
    
    async def _analyze_alert(self, alert: SystemAlert, metric_type: str) -> Dict[str, Any]:
        await asyncio.sleep(1)  # Simulate analysis time
        
        knowledge = self.knowledge_base.get(metric_type, {})
        
        analysis = {
            "root_cause_candidates": knowledge.get("common_causes", ["Unknown causes"]),
            "investigation_steps": knowledge.get("investigation_steps", ["Manual investigation required"]),
            "severity_assessment": alert.severity.value,
            "escalation_needed": alert.severity in [AlertSeverity.HIGH, AlertSeverity.CRITICAL],
            "recommended_actions": []
        }
        
        # Generate context-specific recommendations
        if metric_type == "cpu_usage":
            if alert.metrics.get("cpu_usage", 0) > 95:
                analysis["recommended_actions"] = [
                    "Scale up resources immediately",
                    "Identify and terminate high-CPU processes",
                    "Enable auto-scaling if available"
                ]
        elif metric_type == "memory_usage":
            analysis["recommended_actions"] = [
                "Check for memory leaks",
                "Restart services if safe to do so",
                "Increase memory allocation"
            ]
            
        return analysis

class RemediationAgent(BaseAgent):
    def __init__(self):
        super().__init__("RemediationAgent", [AgentCapability.REMEDIATION])
        self.remediation_playbooks = {
            "cpu_high": [
                "identify_high_cpu_processes",
                "check_system_load",
                "restart_problematic_services",
                "scale_resources_if_needed"
            ],
            "memory_high": [
                "check_memory_leaks",
                "clear_unnecessary_caches",
                "restart_memory_intensive_services",
                "garbage_collection_optimization"
            ]
        }
    
    async def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        alert = task.get("alert")
        analysis = task.get("analysis", {})
        
        playbook = self._select_playbook(alert, analysis)
        execution_results = await self._execute_playbook(playbook, alert)
        
        self.log_action("remediation_executed", {
            "alert_id": alert.id,
            "playbook": playbook,
            "results": execution_results
        })
        
        return {
            "alert_id": alert.id,
            "playbook_executed": playbook,
            "results": execution_results,
            "success": execution_results.get("success", False)
        }
    
    def _select_playbook(self, alert: SystemAlert, analysis: Dict[str, Any]) -> List[str]:
        message_lower = alert.message.lower()
        
        if "cpu" in message_lower:
            return self.remediation_playbooks["cpu_high"]
        elif "memory" in message_lower:
            return self.remediation_playbooks["memory_high"]
        else:
            return ["manual_intervention_required"]
    
    async def _execute_playbook(self, playbook: List[str], alert: SystemAlert) -> Dict[str, Any]:
        results = {"steps": [], "success": True, "errors": []}
        
        for step in playbook:
            step_result = await self._execute_step(step, alert)
            results["steps"].append(step_result)
            
            if not step_result["success"]:
                results["success"] = False
                results["errors"].append(step_result["error"])
                
        return results
    
    async def _execute_step(self, step: str, alert: SystemAlert) -> Dict[str, Any]:
        await asyncio.sleep(0.5)  # Simulate execution time
        
        # Mock step execution - in reality would call APIs/run commands
        step_mappings = {
            "identify_high_cpu_processes": self._mock_process_identification,
            "check_system_load": self._mock_load_check,
            "restart_problematic_services": self._mock_service_restart,
            "scale_resources_if_needed": self._mock_resource_scaling
        }
        
        executor = step_mappings.get(step, self._mock_default_step)
        return await executor(step, alert)
    
    async def _mock_process_identification(self, step: str, alert: SystemAlert) -> Dict[str, Any]:
        return {
            "step": step,
            "success": True,
            "output": "Identified processes: java (45% CPU), python (20% CPU)",
            "action_taken": "Process identification completed"
        }
    
    async def _mock_load_check(self, step: str, alert: SystemAlert) -> Dict[str, Any]:
        return {
            "step": step, 
            "success": True,
            "output": "Load average: 4.2, 3.8, 3.5",
            "action_taken": "System load assessed"
        }
    
    async def _mock_service_restart(self, step: str, alert: SystemAlert) -> Dict[str, Any]:
        # Simulate potential failure
        import random
        success = random.choice([True, True, True, False])  # 75% success rate
        
        if success:
            return {
                "step": step,
                "success": True,
                "output": "Services restarted successfully",
                "action_taken": "Restarted high-CPU services"
            }
        else:
            return {
                "step": step,
                "success": False,
                "error": "Failed to restart service - permission denied",
                "action_taken": "Attempted service restart"
            }
    
    async def _mock_resource_scaling(self, step: str, alert: SystemAlert) -> Dict[str, Any]:
        return {
            "step": step,
            "success": True,
            "output": "Scaled from 2 to 4 instances",
            "action_taken": "Auto-scaling triggered"
        }
    
    async def _mock_default_step(self, step: str, alert: SystemAlert) -> Dict[str, Any]:
        return {
            "step": step,
            "success": False,
            "error": f"Unknown remediation step: {step}",
            "action_taken": "No action taken"
        }

class SupervisorAgent:
    """Orchestrates the multi-agent system"""
    
    def __init__(self):
        self.name = "SupervisorAgent"
        self.agents = {
            "monitoring": MonitoringAgent(),
            "analysis": AnalysisAgent(), 
            "remediation": RemediationAgent()
        }
        self.active_workflows = {}
        self.execution_history = []
        
    async def process_system_monitoring(self, systems: List[str]) -> Dict[str, Any]:
        """Main orchestration method"""
        results = {"workflows": [], "summary": {}}
        
        for system in systems:
            workflow_id = f"workflow_{system}_{int(time.time())}"
            workflow_result = await self._execute_monitoring_workflow(workflow_id, system)
            results["workflows"].append(workflow_result)
            
        results["summary"] = self._generate_summary(results["workflows"])
        self.log_action("monitoring_cycle_completed", results["summary"])
        
        return results
    
    async def _execute_monitoring_workflow(self, workflow_id: str, system: str) -> Dict[str, Any]:
        """Execute complete monitoring -> analysis -> remediation workflow"""
        
        workflow = {
            "id": workflow_id,
            "system": system,
            "start_time": datetime.now(),
            "steps": [],
            "status": "running"
        }
        
        try:
            # Step 1: Monitoring
            monitoring_task = {"system": system}
            monitoring_result = await self.agents["monitoring"].process_task(monitoring_task)
            workflow["steps"].append({"step": "monitoring", "result": monitoring_result})
            
            # Process each alert found
            alerts = monitoring_result.get("alerts", [])
            
            for alert in alerts:
                # Step 2: Analysis
                analysis_task = {"alert": alert}
                analysis_result = await self.agents["analysis"].process_task(analysis_task)
                workflow["steps"].append({"step": "analysis", "result": analysis_result})
                
                # Step 3: Remediation (if needed)
                if analysis_result.get("escalation_needed", False):
                    remediation_task = {"alert": alert, "analysis": analysis_result}
                    remediation_result = await self.agents["remediation"].process_task(remediation_task)
                    workflow["steps"].append({"step": "remediation", "result": remediation_result})
                    
                    # Update alert status based on remediation success
                    if remediation_result.get("success", False):
                        alert.status = AlertStatus.RESOLVED
                    else:
                        alert.status = AlertStatus.ESCALATED
                else:
                    alert.status = AlertStatus.RESOLVED
                    
            workflow["status"] = "completed"
            workflow["end_time"] = datetime.now()
            
        except Exception as e:
            workflow["status"] = "failed"
            workflow["error"] = str(e)
            workflow["end_time"] = datetime.now()
            logger.error(f"Workflow {workflow_id} failed: {e}")
            
        return workflow
    
    def _generate_summary(self, workflows: List[Dict[str, Any]]) -> Dict[str, Any]:
        summary = {
            "total_workflows": len(workflows),
            "successful_workflows": 0,
            "failed_workflows": 0,
            "total_alerts": 0,
            "resolved_alerts": 0,
            "escalated_alerts": 0,
            "systems_monitored": set()
        }
        
        for workflow in workflows:
            if workflow["status"] == "completed":
                summary["successful_workflows"] += 1
            else:
                summary["failed_workflows"] += 1
                
            summary["systems_monitored"].add(workflow["system"])
            
            # Count alerts by status
            for step in workflow.get("steps", []):
                if step["step"] == "monitoring":
                    alerts = step["result"].get("alerts", [])
                    summary["total_alerts"] += len(alerts)
                    
                    for alert in alerts:
                        if alert.status == AlertStatus.RESOLVED:
                            summary["resolved_alerts"] += 1
                        elif alert.status == AlertStatus.ESCALATED:
                            summary["escalated_alerts"] += 1
        
        summary["systems_monitored"] = list(summary["systems_monitored"])
        return summary
    
    def log_action(self, action: str, details: Dict[str, Any] = None):
        timestamp = datetime.now().isoformat()
        log_entry = {
            "timestamp": timestamp,
            "agent": self.name,
            "action": action,
            "details": details or {}
        }
        logger.info(f"[{self.name}] {action}")
        self.execution_history.append(log_entry)

# Demo execution
async def main():
    """Demonstrate the multi-agent ITOps system"""
    
    print("🚀 Starting Multi-Agent ITOps Monitoring System")
    print("=" * 60)
    
    supervisor = SupervisorAgent()
    
    # Systems to monitor
    systems = ["web-server-01", "database-cluster", "api-gateway"]
    
    print(f"📊 Monitoring systems: {', '.join(systems)}")
    print("⏳ Executing monitoring workflows...\n")
    
    # Execute monitoring cycle
    results = await supervisor.process_system_monitoring(systems)
    
    # Display results
    print("\n" + "=" * 60)
    print("📋 EXECUTION SUMMARY")
    print("=" * 60)
    
    summary = results["summary"]
    print(f"Systems Monitored: {len(summary['systems_monitored'])}")
    print(f"Total Workflows: {summary['total_workflows']}")
    print(f"Successful: {summary['successful_workflows']}")
    print(f"Failed: {summary['failed_workflows']}")
    print(f"Alerts Generated: {summary['total_alerts']}")
    print(f"Alerts Resolved: {summary['resolved_alerts']}")
    print(f"Alerts Escalated: {summary['escalated_alerts']}")
    
    print("\n" + "=" * 60)
    print("🔍 WORKFLOW DETAILS")
    print("=" * 60)
    
    for workflow in results["workflows"]:
        print(f"\nWorkflow: {workflow['id']}")
        print(f"System: {workflow['system']}")
        print(f"Status: {workflow['status']}")
        print(f"Steps executed: {len(workflow['steps'])}")
        
        for i, step in enumerate(workflow["steps"], 1):
            step_name = step["step"].title()
            if step["step"] == "monitoring":
                alerts_found = len(step["result"].get("alerts", []))
                print(f"  {i}. {step_name}: {alerts_found} alerts found")
            elif step["step"] == "analysis":
                escalation = step["result"].get("escalation_needed", False)
                print(f"  {i}. {step_name}: Escalation needed: {escalation}")
            elif step["step"] == "remediation":
                success = step["result"].get("success", False)
                print(f"  {i}. {step_name}: Success: {success}")

if __name__ == "__main__":
    asyncio.run(main())