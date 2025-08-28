# Tool-Calling Agent for API Integration
# Demonstrates dynamic tool usage and external system integration

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import aiohttp
import time

# Mock dependencies - in real implementation you'd use:
# from openai import OpenAI
# import requests
# from langchain.tools import Tool

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ToolType(Enum):
    API_CALL = "api_call"
    SYSTEM_COMMAND = "system_command"
    DATABASE_QUERY = "database_query"
    FILE_OPERATION = "file_operation"
    NOTIFICATION = "notification"

@dataclass
class ToolDefinition:
    name: str
    description: str
    parameters: Dict[str, Any]
    tool_type: ToolType
    required_permissions: List[str] = None
    timeout_seconds: int = 30
    retry_count: int = 3
    
    def __post_init__(self):
        if self.required_permissions is None:
            self.required_permissions = []

@dataclass
class ToolCall:
    tool_name: str
    parameters: Dict[str, Any]
    call_id: str
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

@dataclass
class ToolResult:
    call_id: str
    success: bool
    result: Any
    error_message: Optional[str] = None
    execution_time_ms: float = 0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class MockAPIClient:
    """Mock API client for external system integration"""
    
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.session = None
        
    async def get(self, endpoint: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Mock GET request"""
        await asyncio.sleep(0.2)  # Simulate network delay
        
        # Mock responses based on endpoint
        if "servers" in endpoint:
            return {
                "servers": [
                    {"id": "srv-001", "name": "web-server-01", "status": "running", "cpu": 45.2, "memory": 67.8},
                    {"id": "srv-002", "name": "db-server-01", "status": "running", "cpu": 78.5, "memory": 82.1},
                    {"id": "srv-003", "name": "api-gateway", "status": "warning", "cpu": 92.3, "memory": 45.6}
                ]
            }
        elif "alerts" in endpoint:
            return {
                "alerts": [
                    {"id": "alert-001", "severity": "high", "message": "CPU usage > 90%", "server": "srv-003"},
                    {"id": "alert-002", "severity": "medium", "message": "Memory usage > 80%", "server": "srv-002"}
                ]
            }
        elif "metrics" in endpoint:
            server_id = params.get("server_id") if params else "srv-001"
            return {
                "server_id": server_id,
                "timestamp": datetime.now().isoformat(),
                "metrics": {
                    "cpu_percent": 67.4,
                    "memory_percent": 78.2,
                    "disk_percent": 34.7,
                    "network_in_mbps": 12.4,
                    "network_out_mbps": 8.9
                }
            }
        else:
            return {"message": "Mock API response", "endpoint": endpoint, "params": params}
    
    async def post(self, endpoint: str, data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Mock POST request"""
        await asyncio.sleep(0.3)
        
        if "restart" in endpoint:
            return {"status": "success", "message": "Service restart initiated", "job_id": f"job-{int(time.time())}"}
        elif "scale" in endpoint:
            return {"status": "success", "message": "Scaling operation started", "new_instances": data.get("instances", 1)}
        elif "notifications" in endpoint:
            return {"status": "sent", "message_id": f"msg-{int(time.time())}", "recipients": data.get("recipients", [])}
        else:
            return {"status": "success", "message": "Operation completed", "data": data}

class ToolRegistry:
    """Registry for managing available tools"""
    
    def __init__(self):
        self.tools: Dict[str, ToolDefinition] = {}
        self.tool_functions: Dict[str, Callable] = {}
        self._register_default_tools()
    
    def register_tool(self, tool_def: ToolDefinition, func: Callable):
        """Register a new tool"""
        self.tools[tool_def.name] = tool_def
        self.tool_functions[tool_def.name] = func
        logger.info(f"Registered tool: {tool_def.name}")
    
    def get_tool(self, name: str) -> Optional[ToolDefinition]:
        """Get tool definition by name"""
        return self.tools.get(name)
    
    def get_available_tools(self) -> List[ToolDefinition]:
        """Get all available tools"""
        return list(self.tools.values())
    
    def _register_default_tools(self):
        """Register default ITOps tools"""
        
        # Server monitoring tools
        self.register_tool(
            ToolDefinition(
                name="get_server_status",
                description="Get status and metrics for servers",
                parameters={
                    "server_ids": {"type": "array", "items": {"type": "string"}, "description": "List of server IDs to check"},
                    "include_metrics": {"type": "boolean", "default": True, "description": "Include detailed metrics"}
                },
                tool_type=ToolType.API_CALL,
                required_permissions=["monitoring.read"]
            ),
            self._get_server_status
        )
        
        self.register_tool(
            ToolDefinition(
                name="get_system_alerts",
                description="Retrieve active system alerts",
                parameters={
                    "severity": {"type": "string", "enum": ["low", "medium", "high", "critical"], "description": "Filter by severity"},
                    "limit": {"type": "integer", "default": 50, "description": "Maximum number of alerts to return"}
                },
                tool_type=ToolType.API_CALL,
                required_permissions=["alerts.read"]
            ),
            self._get_system_alerts
        )
        
        # System control tools
        self.register_tool(
            ToolDefinition(
                name="restart_service",
                description="Restart a specific service on a server",
                parameters={
                    "server_id": {"type": "string", "description": "Server identifier"},
                    "service_name": {"type": "string", "description": "Name of service to restart"},
                    "force": {"type": "boolean", "default": False, "description": "Force restart even if service is healthy"}
                },
                tool_type=ToolType.SYSTEM_COMMAND,
                required_permissions=["system.restart"],
                timeout_seconds=60
            ),
            self._restart_service
        )
        
        self.register_tool(
            ToolDefinition(
                name="scale_application",
                description="Scale application instances up or down",
                parameters={
                    "app_name": {"type": "string", "description": "Application name"},
                    "instances": {"type": "integer", "minimum": 1, "maximum": 20, "description": "Target number of instances"},
                    "environment": {"type": "string", "enum": ["dev", "staging", "prod"], "description": "Environment to scale"}
                },
                tool_type=ToolType.API_CALL,
                required_permissions=["scaling.write"],
                timeout_seconds=120
            ),
            self._scale_application
        )
        
        # Database tools
        self.register_tool(
            ToolDefinition(
                name="execute_query",
                description="Execute a database query for monitoring purposes",
                parameters={
                    "query": {"type": "string", "description": "SQL query to execute (read-only)"},
                    "database": {"type": "string", "description": "Database name"},
                    "timeout": {"type": "integer", "default": 30, "description": "Query timeout in seconds"}
                },
                tool_type=ToolType.DATABASE_QUERY,
                required_permissions=["database.read"]
            ),
            self._execute_query
        )
        
        # Notification tools
        self.register_tool(
            ToolDefinition(
                name="send_notification",
                description="Send notifications to team members",
                parameters={
                    "message": {"type": "string", "description": "Notification message"},
                    "recipients": {"type": "array", "items": {"type": "string"}, "description": "List of recipient IDs"},
                    "priority": {"type": "string", "enum": ["low", "normal", "high", "urgent"], "default": "normal"},
                    "channels": {"type": "array", "items": {"type": "string"}, "description": "Notification channels (email, slack, sms)"}
                },
                tool_type=ToolType.NOTIFICATION,
                required_permissions=["notifications.send"]
            ),
            self._send_notification
        )
        
        # File operations
        self.register_tool(
            ToolDefinition(
                name="read_log_file",
                description="Read and analyze log files",
                parameters={
                    "file_path": {"type": "string", "description": "Path to log file"},
                    "lines": {"type": "integer", "default": 100, "description": "Number of recent lines to read"},
                    "filter_pattern": {"type": "string", "description": "Regex pattern to filter lines"}
                },
                tool_type=ToolType.FILE_OPERATION,
                required_permissions=["logs.read"]
            ),
            self._read_log_file
        )
    
    # Tool implementations
    async def _get_server_status(self, **kwargs) -> Dict[str, Any]:
        """Get server status from monitoring API"""
        api_client = MockAPIClient("https://monitoring-api.example.com")
        
        server_ids = kwargs.get("server_ids", [])
        include_metrics = kwargs.get("include_metrics", True)
        
        # Get server list
        servers_response = await api_client.get("/api/v1/servers")
        
        if server_ids:
            # Filter by requested server IDs
            filtered_servers = [s for s in servers_response["servers"] if s["id"] in server_ids]
        else:
            filtered_servers = servers_response["servers"]
        
        # Enhance with detailed metrics if requested
        if include_metrics:
            for server in filtered_servers:
                metrics_response = await api_client.get(f"/api/v1/metrics", {"server_id": server["id"]})
                server["detailed_metrics"] = metrics_response["metrics"]
        
        return {
            "servers": filtered_servers,
            "total_count": len(filtered_servers),
            "timestamp": datetime.now().isoformat()
        }
    
    async def _get_system_alerts(self, **kwargs) -> Dict[str, Any]:
        """Get system alerts from alerting API"""
        api_client = MockAPIClient("https://alerts-api.example.com")
        
        severity_filter = kwargs.get("severity")
        limit = kwargs.get("limit", 50)
        
        alerts_response = await api_client.get("/api/v1/alerts")
        alerts = alerts_response["alerts"]
        
        # Apply severity filter
        if severity_filter:
            alerts = [a for a in alerts if a["severity"] == severity_filter]
        
        # Apply limit
        alerts = alerts[:limit]
        
        return {
            "alerts": alerts,
            "count": len(alerts),
            "filtered_by_severity": severity_filter,
            "timestamp": datetime.now().isoformat()
        }
    
    async def _restart_service(self, **kwargs) -> Dict[str, Any]:
        """Restart service via system API"""
        api_client = MockAPIClient("https://system-api.example.com")
        
        server_id = kwargs["server_id"]
        service_name = kwargs["service_name"]
        force = kwargs.get("force", False)
        
        # Simulate pre-restart checks
        if not force:
            await asyncio.sleep(1)  # Health check simulation
        
        restart_response = await api_client.post(
            f"/api/v1/servers/{server_id}/services/{service_name}/restart",
            {"force": force}
        )
        
        return {
            "server_id": server_id,
            "service_name": service_name,
            "status": restart_response["status"],
            "job_id": restart_response.get("job_id"),
            "message": restart_response["message"]
        }
    
    async def _scale_application(self, **kwargs) -> Dict[str, Any]:
        """Scale application instances"""
        api_client = MockAPIClient("https://orchestration-api.example.com")
        
        app_name = kwargs["app_name"]
        instances = kwargs["instances"]
        environment = kwargs["environment"]
        
        scale_response = await api_client.post(
            f"/api/v1/apps/{app_name}/scale",
            {
                "instances": instances,
                "environment": environment
            }
        )
        
        return {
            "app_name": app_name,
            "target_instances": instances,
            "environment": environment,
            "status": scale_response["status"],
            "message": scale_response["message"]
        }
    
    async def _execute_query(self, **kwargs) -> Dict[str, Any]:
        """Execute database query"""
        query = kwargs["query"]
        database = kwargs["database"]
        timeout = kwargs.get("timeout", 30)
        
        # Simulate query execution
        await asyncio.sleep(0.5)  # Query execution time
        
        # Mock query results based on query type
        if "SELECT" in query.upper():
            if "performance" in query.lower() or "slow" in query.lower():
                return {
                    "query": query,
                    "database": database,
                    "results": [
                        {"query_time": "2.3s", "query": "SELECT * FROM users WHERE...", "executions": 45},
                        {"query_time": "1.8s", "query": "SELECT * FROM orders JOIN...", "executions": 23}
                    ],
                    "execution_time_ms": 487,
                    "rows_returned": 2
                }
            elif "count" in query.lower():
                return {
                    "query": query,
                    "database": database,
                    "results": [{"count": 15642}],
                    "execution_time_ms": 123,
                    "rows_returned": 1
                }
            else:
                return {
                    "query": query,
                    "database": database,
                    "results": [
                        {"id": 1, "name": "sample_data", "status": "active"},
                        {"id": 2, "name": "test_data", "status": "inactive"}
                    ],
                    "execution_time_ms": 234,
                    "rows_returned": 2
                }
        else:
            return {
                "error": "Only SELECT queries are allowed for monitoring purposes",
                "query": query,
                "database": database
            }
    
    async def _send_notification(self, **kwargs) -> Dict[str, Any]:
        """Send notification via messaging API"""
        api_client = MockAPIClient("https://notifications-api.example.com")
        
        message = kwargs["message"]
        recipients = kwargs["recipients"]
        priority = kwargs.get("priority", "normal")
        channels = kwargs.get("channels", ["email"])
        
        notification_response = await api_client.post(
            "/api/v1/notifications/send",
            {
                "message": message,
                "recipients": recipients,
                "priority": priority,
                "channels": channels
            }
        )
        
        return {
            "message_id": notification_response["message_id"],
            "status": notification_response["status"],
            "recipients_count": len(recipients),
            "channels": channels,
            "priority": priority
        }
    
    async def _read_log_file(self, **kwargs) -> Dict[str, Any]:
        """Read log file contents"""
        file_path = kwargs["file_path"]
        lines = kwargs.get("lines", 100)
        filter_pattern = kwargs.get("filter_pattern")
        
        # Simulate log file reading
        await asyncio.sleep(0.3)
        
        # Mock log entries
        mock_logs = [
            "2024-01-15 10:30:22 INFO Starting application server",
            "2024-01-15 10:30:23 INFO Database connection established",
            "2024-01-15 10:30:45 WARN High memory usage detected: 85%",
            "2024-01-15 10:31:12 ERROR Failed to process request: timeout",
            "2024-01-15 10:31:15 INFO Request completed successfully",
            "2024-01-15 10:31:22 WARN CPU usage spike: 92%",
            "2024-01-15 10:31:45 ERROR Database connection lost",
            "2024-01-15 10:32:01 INFO Database connection restored"
        ]
        
        # Apply line limit
        log_entries = mock_logs[-lines:] if lines < len(mock_logs) else mock_logs
        
        # Apply filter pattern if provided
        if filter_pattern:
            import re
            pattern = re.compile(filter_pattern, re.IGNORECASE)
            log_entries = [entry for entry in log_entries if pattern.search(entry)]
        
        return {
            "file_path": file_path,
            "lines_requested": lines,
            "lines_returned": len(log_entries),
            "filter_pattern": filter_pattern,
            "log_entries": log_entries,
            "summary": {
                "info_count": len([l for l in log_entries if "INFO" in l]),
                "warn_count": len([l for l in log_entries if "WARN" in l]),
                "error_count": len([l for l in log_entries if "ERROR" in l])
            }
        }

class ToolCallingAgent:
    """AI Agent that can dynamically call tools to interact with external systems"""
    
    def __init__(self):
        self.name = "ToolCallingAgent"
        self.tool_registry = ToolRegistry()
        self.execution_history = []
        self.permissions = [
            "monitoring.read", "alerts.read", "system.restart",
            "scaling.write", "database.read", "notifications.send", "logs.read"
        ]
        
    async def process_request(self, request: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process a request that may require tool usage"""
        
        # Analyze request and determine required tools
        tool_plan = self._analyze_request(request, context or {})
        
        if not tool_plan["tools_needed"]:
            return {
                "response": "I understand your request, but I don't need to use any tools to answer it. Could you provide more specific information about what system actions you need?",
                "tools_used": [],
                "execution_time_ms": 0
            }
        
        # Execute tool chain
        execution_start = time.time()
        results = await self._execute_tool_chain(tool_plan["tools_needed"])
        execution_time = (time.time() - execution_start) * 1000
        
        # Generate response based on tool results
        response = self._generate_response(request, results, tool_plan)
        
        # Log execution
        self.log_action("request_processed", {
            "request": request,
            "tools_used": [t["tool_name"] for t in tool_plan["tools_needed"]],
            "execution_time_ms": execution_time,
            "success": all(r.success for r in results)
        })
        
        return {
            "response": response,
            "tools_used": [
                {
                    "tool_name": result.call_id.split("_")[0],
                    "success": result.success,
                    "execution_time_ms": result.execution_time_ms
                }
                for result in results
            ],
            "execution_time_ms": execution_time
        }
    
    def _analyze_request(self, request: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze request and determine which tools to use"""
        request_lower = request.lower()
        tools_needed = []
        
        # Pattern matching for different request types
        if any(word in request_lower for word in ["server status", "server health", "check servers"]):
            tools_needed.append({
                "tool_name": "get_server_status",
                "parameters": {"include_metrics": True}
            })
        
        if any(word in request_lower for word in ["alerts", "incidents", "problems"]):
            tools_needed.append({
                "tool_name": "get_system_alerts",
                "parameters": {"limit": 20}
            })
        
        if any(word in request_lower for word in ["restart", "reboot", "reload"]):
            # Extract service/server info from context or request
            server_id = context.get("server_id", "srv-001")  # Default fallback
            service_name = context.get("service_name", "application")
            
            tools_needed.append({
                "tool_name": "restart_service",
                "parameters": {
                    "server_id": server_id,
                    "service_name": service_name,
                    "force": "force" in request_lower
                }
            })
        
        if any(word in request_lower for word in ["scale", "instances", "scale up", "scale down"]):
            app_name = context.get("app_name", "web-app")
            instances = context.get("instances", 3)
            environment = context.get("environment", "prod")
            
            tools_needed.append({
                "tool_name": "scale_application", 
                "parameters": {
                    "app_name": app_name,
                    "instances": instances,
                    "environment": environment
                }
            })
        
        if any(word in request_lower for word in ["logs", "log file", "check logs"]):
            file_path = context.get("log_path", "/var/log/application.log")
            tools_needed.append({
                "tool_name": "read_log_file",
                "parameters": {
                    "file_path": file_path,
                    "lines": 50,
                    "filter_pattern": context.get("log_filter")
                }
            })
        
        if any(word in request_lower for word in ["notify", "alert team", "send notification"]):
            tools_needed.append({
                "tool_name": "send_notification",
                "parameters": {
                    "message": context.get("message", "System alert notification"),
                    "recipients": context.get("recipients", ["admin@company.com"]),
                    "priority": "high" if "urgent" in request_lower else "normal",
                    "channels": ["email", "slack"]
                }
            })
        
        if any(word in request_lower for word in ["database", "query", "db performance"]):
            query = context.get("query", "SELECT COUNT(*) FROM active_connections")
            database = context.get("database", "production")
            
            tools_needed.append({
                "tool_name": "execute_query",
                "parameters": {
                    "query": query,
                    "database": database
                }
            })
        
        return {
            "tools_needed": tools_needed,
            "analysis": {
                "intent": self._classify_intent(request),
                "urgency": "high" if any(word in request_lower for word in ["urgent", "critical", "emergency"]) else "normal",
                "action_type": self._classify_action_type(request)
            }
        }
    
    def _classify_intent(self, request: str) -> str:
        """Classify the intent of the request"""
        request_lower = request.lower()
        
        if any(word in request_lower for word in ["status", "check", "monitor", "health"]):
            return "monitoring"
        elif any(word in request_lower for word in ["restart", "fix", "resolve", "repair"]):
            return "remediation"
        elif any(word in request_lower for word in ["scale", "increase", "decrease", "optimize"]):
            return "scaling"
        elif any(word in request_lower for word in ["notify", "alert", "inform"]):
            return "communication"
        elif any(word in request_lower for word in ["logs", "investigate", "analyze"]):
            return "investigation"
        else:
            return "general"
    
    def _classify_action_type(self, request: str) -> str:
        """Classify the type of action needed"""
        request_lower = request.lower()
        
        if any(word in request_lower for word in ["read", "check", "status", "get"]):
            return "read"
        elif any(word in request_lower for word in ["restart", "scale", "change", "modify"]):
            return "write"
        elif any(word in request_lower for word in ["notify", "send", "alert"]):
            return "notify"
        else:
            return "mixed"
    
    async def _execute_tool_chain(self, tool_calls: List[Dict[str, Any]]) -> List[ToolResult]:
        """Execute a chain of tool calls"""
        results = []
        
        for i, tool_call in enumerate(tool_calls):
            call_id = f"{tool_call['tool_name']}_{int(time.time())}_{i}"
            
            try:
                start_time = time.time()
                
                # Check permissions
                tool_def = self.tool_registry.get_tool(tool_call['tool_name'])
                if not tool_def:
                    result = ToolResult(
                        call_id=call_id,
                        success=False,
                        result=None,
                        error_message=f"Tool '{tool_call['tool_name']}' not found"
                    )
                    results.append(result)
                    continue
                
                if not self._check_permissions(tool_def.required_permissions):
                    result = ToolResult(
                        call_id=call_id,
                        success=False,
                        result=None,
                        error_message=f"Insufficient permissions for tool '{tool_call['tool_name']}'"
                    )
                    results.append(result)
                    continue
                
                # Execute tool
                tool_func = self.tool_registry.tool_functions[tool_call['tool_name']]
                
                # Add timeout handling
                try:
                    tool_result = await asyncio.wait_for(
                        tool_func(**tool_call['parameters']),
                        timeout=tool_def.timeout_seconds
                    )
                    
                    execution_time = (time.time() - start_time) * 1000
                    
                    result = ToolResult(
                        call_id=call_id,
                        success=True,
                        result=tool_result,
                        execution_time_ms=execution_time,
                        metadata={
                            "tool_type": tool_def.tool_type.value,
                            "parameters": tool_call['parameters']
                        }
                    )
                    
                except asyncio.TimeoutError:
                    result = ToolResult(
                        call_id=call_id,
                        success=False,
                        result=None,
                        error_message=f"Tool execution timed out after {tool_def.timeout_seconds}s",
                        execution_time_ms=(time.time() - start_time) * 1000
                    )
                
            except Exception as e:
                result = ToolResult(
                    call_id=call_id,
                    success=False,
                    result=None,
                    error_message=str(e),
                    execution_time_ms=(time.time() - start_time) * 1000
                )
            
            results.append(result)
            
            # Add small delay between tool calls to prevent overwhelming systems
            await asyncio.sleep(0.1)
        
        return results
    
    def _check_permissions(self, required_permissions: List[str]) -> bool:
        """Check if agent has required permissions"""
        return all(perm in self.permissions for perm in required_permissions)
    
    def _generate_response(self, request: str, results: List[ToolResult], 
                          tool_plan: Dict[str, Any]) -> str:
        """Generate response based on tool execution results"""
        
        successful_results = [r for r in results if r.success]
        failed_results = [r for r in results if not r.success]
        
        if not successful_results and failed_results:
            return f"I encountered errors while processing your request: {'; '.join([r.error_message for r in failed_results])}"
        
        # Generate response based on intent
        intent = tool_plan["analysis"]["intent"]
        
        if intent == "monitoring":
            return self._generate_monitoring_response(successful_results)
        elif intent == "remediation":
            return self._generate_remediation_response(successful_results)
        elif intent == "scaling":
            return self._generate_scaling_response(successful_results)
        elif intent == "communication":
            return self._generate_communication_response(successful_results)
        elif intent == "investigation":
            return self._generate_investigation_response(successful_results)
        else:
            return self._generate_general_response(successful_results)
    
    def _generate_monitoring_response(self, results: List[ToolResult]) -> str:
        """Generate response for monitoring requests"""
        response_parts = []
        
        for result in results:
            if "server_status" in result.call_id:
                servers = result.result.get("servers", [])
                healthy_count = len([s for s in servers if s["status"] == "running"])
                total_count = len(servers)
                
                response_parts.append(f"Server Status: {healthy_count}/{total_count} servers are running normally.")
                
                # Highlight any issues
                problematic_servers = [s for s in servers if s["status"] != "running"]
                if problematic_servers:
                    response_parts.append(f"⚠️ Servers needing attention: {', '.join([s['name'] for s in problematic_servers])}")
            
            elif "system_alerts" in result.call_id:
                alerts = result.result.get("alerts", [])
                if alerts:
                    high_priority = len([a for a in alerts if a["severity"] in ["high", "critical"]])
                    response_parts.append(f"🚨 Found {len(alerts)} active alerts ({high_priority} high priority)")
                else:
                    response_parts.append("✅ No active alerts found")
        
        return " ".join(response_parts) if response_parts else "Monitoring check completed successfully."
    
    def _generate_remediation_response(self, results: List[ToolResult]) -> str:
        """Generate response for remediation actions"""
        response_parts = []
        
        for result in results:
            if "restart_service" in result.call_id:
                job_id = result.result.get("job_id")
                service = result.result.get("service_name")
                response_parts.append(f"✅ Service '{service}' restart initiated (Job ID: {job_id})")
        
        return " ".join(response_parts) if response_parts else "Remediation actions completed."
    
    def _generate_scaling_response(self, results: List[ToolResult]) -> str:
        """Generate response for scaling operations"""
        response_parts = []
        
        for result in results:
            if "scale_application" in result.call_id:
                app_name = result.result.get("app_name")
                instances = result.result.get("target_instances")
                response_parts.append(f"🔧 Scaling '{app_name}' to {instances} instances")
        
        return " ".join(response_parts) if response_parts else "Scaling operations completed."
    
    def _generate_communication_response(self, results: List[ToolResult]) -> str:
        """Generate response for communication actions"""
        response_parts = []
        
        for result in results:
            if "send_notification" in result.call_id:
                recipients_count = result.result.get("recipients_count")
                message_id = result.result.get("message_id")
                response_parts.append(f"📧 Notification sent to {recipients_count} recipients (ID: {message_id})")
        
        return " ".join(response_parts) if response_parts else "Notifications sent successfully."
    
    def _generate_investigation_response(self, results: List[ToolResult]) -> str:
        """Generate response for investigation requests"""
        response_parts = []
        
        for result in results:
            if "read_log_file" in result.call_id:
                lines_returned = result.result.get("lines_returned")
                summary = result.result.get("summary", {})
                error_count = summary.get("error_count", 0)
                warn_count = summary.get("warn_count", 0)
                
                response_parts.append(f"📋 Analyzed {lines_returned} log entries: {error_count} errors, {warn_count} warnings")
            
            elif "execute_query" in result.call_id:
                rows = result.result.get("rows_returned")
                exec_time = result.result.get("execution_time_ms")
                response_parts.append(f"🗃️ Database query returned {rows} rows in {exec_time}ms")
        
        return " ".join(response_parts) if response_parts else "Investigation completed."
    
    def _generate_general_response(self, results: List[ToolResult]) -> str:
        """Generate general response"""
        successful_tools = len(results)
        total_time = sum(r.execution_time_ms for r in results)
        
        return f"✅ Successfully executed {successful_tools} operations in {total_time:.1f}ms"
    
    def get_available_tools(self) -> List[Dict[str, Any]]:
        """Get list of available tools"""
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "type": tool.tool_type.value,
                "parameters": tool.parameters,
                "permissions": tool.required_permissions
            }
            for tool in self.tool_registry.get_available_tools()
        ]
    
    def log_action(self, action: str, details: Dict[str, Any] = None):
        timestamp = datetime.now().isoformat()
        log_entry = {
            "timestamp": timestamp,
            "agent": self.name,
            "action": action,
            "details": details or {}
        }
        logger.info(f"[{self.name}] {action}: {details}")
        self.execution_history.append(log_entry)

# Demo execution
async def main():
    """Demonstrate the Tool-Calling Agent"""
    
    print("🛠️  Starting Tool-Calling Agent for API Integration")
    print("=" * 60)
    
    agent = ToolCallingAgent()
    
    # Display available tools
    print("🔧 Available Tools:")
    tools = agent.get_available_tools()
    for tool in tools:
        print(f"  - {tool['name']}: {tool['description']}")
    print()
    
    # Sample requests to demonstrate different capabilities
    sample_requests = [
        {
            "request": "Check the status of all servers and show me any alerts",
            "context": {}
        },
        {
            "request": "Restart the application service on server srv-003 due to high CPU",
            "context": {"server_id": "srv-003", "service_name": "web-application"}
        },
        {
            "request": "Scale up the web-app to 5 instances in production",
            "context": {"app_name": "web-app", "instances": 5, "environment": "prod"}
        },
        {
            "request": "Check the application logs for any errors in the last 50 lines",
            "context": {"log_path": "/var/log/app.log", "log_filter": "ERROR"}
        },
        {
            "request": "Send an urgent notification to the team about the server issues",
            "context": {
                "message": "Critical: Server srv-003 experiencing high CPU usage",
                "recipients": ["admin@company.com", "oncall@company.com"],
                "priority": "urgent"
            }
        },
        {
            "request": "Query the database to check active connections",
            "context": {"query": "SELECT COUNT(*) as active_connections FROM pg_stat_activity WHERE state = 'active'", "database": "production"}
        }
    ]
    
    print("🚀 Processing sample requests...\n")
    
    for i, request_data in enumerate(sample_requests, 1):
        print(f"Request {i}: {request_data['request']}")
        print("-" * 50)
        
        # Process request
        result = await agent.process_request(
            request_data['request'],
            request_data['context']
        )
        
        # Display results
        print(f"Response: {result['response']}")
        print(f"Tools Used: {len(result['tools_used'])}")
        print(f"Execution Time: {result['execution_time_ms']:.1f}ms")
        
        if result['tools_used']:
            print("Tool Details:")
            for tool in result['tools_used']:
                status = "✅" if tool['success'] else "❌"
                print(f"  {status} {tool['tool_name']} ({tool['execution_time_ms']:.1f}ms)")
        
        print()
    
    print("=" * 60)
    print("📊 EXECUTION SUMMARY")
    print("=" * 60)
    
    history = agent.execution_history
    total_requests = len(history)
    total_tools_used = sum(len(entry['details']['tools_used']) for entry in history)
    avg_execution_time = sum(entry['details']['execution_time_ms'] for entry in history) / total_requests if total_requests > 0 else 0
    
    print(f"Total Requests Processed: {total_requests}")
    print(f"Total Tools Executed: {total_tools_used}")
    print(f"Average Execution Time: {avg_execution_time:.1f}ms")
    print(f"Available Tools: {len(tools)}")
    print(f"Agent Permissions: {len(agent.permissions)}")

if __name__ == "__main__":
    asyncio.run(main())