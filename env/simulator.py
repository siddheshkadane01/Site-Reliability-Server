import copy
import random
from datetime import datetime, timedelta, timezone

from .models import Alert, HealthSummary, LogEntry, SystemMetrics

SERVICES = [
    "api-gateway",
    "auth-service",
    "user-service",
    "order-service",
    "db-proxy",
    "cache-service",
]

SERVICE_GRAPH = {
    "api-gateway": ["auth-service", "order-service", "user-service"],
    "auth-service": ["db-proxy", "cache-service"],
    "user-service": ["db-proxy", "cache-service"],
    "order-service": ["db-proxy"],
    "db-proxy": [],
    "cache-service": [],
}

HEALTH_THRESHOLDS = {
    "cpu_pct": 70.0,
    "mem_pct": 80.0,
    "error_rate": 0.01,
    "latency_ms": 200.0,
}

# Optional per-step jitter for non-evaluation mode. Evaluation mode disables it
# so the same action sequence always yields the same trajectory and score.
_DRIFT_SIGMA = {
    "cpu_pct": 0.6,
    "mem_pct": 0.4,
    "error_rate": 0.002,
    "latency_ms": 3.0,
}
_DRIFT_CLAMP = {
    "cpu_pct": (0.0, 100.0),
    "mem_pct": (0.0, 100.0),
    "error_rate": (0.0, 1.0),
    "latency_ms": (5.0, 5000.0),
}


class VirtualDataCentre:
    """Pure in-memory simulation of a 6-service data centre."""

    def __init__(
        self,
        scenario: dict,
        *,
        enable_drift: bool = False,
        seed: int = 42,
    ):
        self.scenario = copy.deepcopy(scenario)
        self.state = copy.deepcopy(scenario["initial_state"])
        self.config = copy.deepcopy(scenario["initial_config"])
        self.replicas = {s: 1 for s in SERVICES}
        self.deploy_history = copy.deepcopy(scenario.get("deploy_history", []))
        self.enable_drift = enable_drift
        self._rng = random.Random(seed)
        self.alerts: list[Alert] = []
        self.logs: list[LogEntry] = []
        self._step_index = 0
        self._base_time = self._derive_base_time()
        self._traffic_drains = {service: 0 for service in SERVICES}
        # Track which services have been genuinely fixed (health restored) for
        # the SILENCE_ALERT bonus reward.
        self._silenced_fixed: set[str] = set()
        self._refresh_alerts()

    # ------------------------------------------------------------------
    # Public read API
    # ------------------------------------------------------------------

    def get_metrics(self) -> SystemMetrics:
        return SystemMetrics(
            cpu_pct={s: self.state[s]["cpu_pct"] for s in SERVICES},
            mem_pct={s: self.state[s]["mem_pct"] for s in SERVICES},
            error_rate={s: self.state[s]["error_rate"] for s in SERVICES},
            latency_ms={s: self.state[s]["latency_ms"] for s in SERVICES},
            timestamp=self._now(),
        )

    def health_score(self) -> HealthSummary:
        scores: dict[str, float] = {}
        for s in SERVICES:
            st = self.state[s]
            cpu_score = max(0.0, min(1.0, (HEALTH_THRESHOLDS["cpu_pct"] - st["cpu_pct"]) / 30.0 + 1.0))
            mem_score = max(0.0, min(1.0, (HEALTH_THRESHOLDS["mem_pct"] - st["mem_pct"]) / 20.0 + 1.0))
            err_score = max(0.0, min(1.0, 1.0 - st["error_rate"] / 1.0))
            lat_score = max(
                0.0,
                min(1.0, (HEALTH_THRESHOLDS["latency_ms"] - st["latency_ms"]) / 1800.0 + 1.0),
            )
            scores[s] = round((cpu_score + mem_score + err_score + lat_score) / 4.0, 4)

        overall = round(sum(scores.values()) / len(scores), 4)
        return HealthSummary(per_service=scores, overall=overall)

    def is_healthy(self) -> bool:
        for s in SERVICES:
            st = self.state[s]
            if (
                st["cpu_pct"] >= HEALTH_THRESHOLDS["cpu_pct"]
                or st["mem_pct"] >= HEALTH_THRESHOLDS["mem_pct"]
                or st["error_rate"] >= HEALTH_THRESHOLDS["error_rate"]
                or st["latency_ms"] >= HEALTH_THRESHOLDS["latency_ms"]
            ):
                return False
        return True

    # ------------------------------------------------------------------
    # Action dispatch
    # ------------------------------------------------------------------

    def apply_action(
        self, action_type: str, target: str, config_key=None, config_value=None
    ) -> dict:
        """Apply an action to the simulation. Returns result info dict."""
        result: dict = {"valid": True, "changed": False, "details": "", "silence_bonus": False}
        self._step_index += 1

        if target not in SERVICES:
            result["valid"] = False
            result["details"] = f"Unknown service: {target}"
            return result

        if action_type == "CHECK_LOGS":
            self._handle_check_logs(target, result)

        elif action_type == "INSPECT_SERVICE":
            self._handle_inspect(target, result)

        elif action_type == "DRAIN_TRAFFIC":
            self._handle_drain_traffic(target, result)

        elif action_type == "RESTART_SERVICE":
            self._handle_restart(target, result)

        elif action_type == "SCALE_UP":
            self._handle_scale_up(target, result)

        elif action_type == "SCALE_DOWN":
            self._handle_scale_down(target, result)

        elif action_type == "ROLLBACK":
            self._handle_rollback(target, result)

        elif action_type == "UPDATE_CONFIG":
            self._handle_update_config(target, config_key, config_value, result)

        elif action_type == "SILENCE_ALERT":
            self._handle_silence_alert(target, result)

        # Drift is disabled in evaluation mode to keep trajectories reproducible.
        if self.enable_drift:
            self._apply_drift()
        self._refresh_alerts()
        return result

    # ------------------------------------------------------------------
    # Action handlers
    # ------------------------------------------------------------------

    def _handle_check_logs(self, target: str, result: dict) -> None:
        db_timeout = int(self.config.get("db_timeout", 5000))
        pool_size = int(self.config.get("pool_size", 10))
        hard_root_key = self.scenario.get("ground_truth", {}).get("correct_config_key")
        if target == "db-proxy" and db_timeout < 500:
            self._add_log(
                target,
                "ERROR",
                f"Database timeout misconfigured: db_timeout={db_timeout}ms causing upstream request failures",
            )
            result["details"] = "Retrieved logs for db-proxy: db timeout appears misconfigured"
        elif target == "db-proxy" and hard_root_key == "pool_size" and pool_size < 10:
            self._add_log(
                target,
                "ERROR",
                f"Connection pool saturation detected: pool_size={pool_size} is exhausting database worker slots",
            )
            result["details"] = "Retrieved logs for db-proxy: connection pool saturation detected"
        elif target in ["order-service", "user-service", "auth-service"] and db_timeout < 500:
            self._add_log(
                target,
                "WARN",
                "Frequent dependency timeout from db-proxy; check database client timeout configuration",
            )
            result["details"] = f"Retrieved logs for {target}: dependency timeout errors observed"
        elif target in ["order-service", "user-service", "auth-service"] and hard_root_key == "pool_size" and pool_size < 10:
            self._add_log(
                target,
                "WARN",
                "Database connection pool exhausted downstream; requests are queueing behind the proxy",
            )
            result["details"] = f"Retrieved logs for {target}: downstream connection pool saturation observed"
        elif target == "cache-service" and self.state[target]["error_rate"] > 0.05:
            self._add_log(
                target,
                "ERROR",
                "Cache eviction storm detected: connection pool exhausted, serving stale keys",
            )
            result["details"] = f"Retrieved logs for {target}: cache pool exhaustion detected"
        else:
            self._add_log(target, "INFO", f"Log fetch requested for {target}")
            result["details"] = f"Retrieved logs for {target}"

    def _handle_inspect(self, target: str, result: dict) -> None:
        st = self.state[target]
        if target == "db-proxy":
            result["details"] = (
                f"Inspected db-proxy: cpu={st['cpu_pct']:.1f}% mem={st['mem_pct']:.1f}% "
                f"err={st['error_rate']:.3f} lat={st['latency_ms']:.0f}ms "
                f"db_timeout={self.config.get('db_timeout')} pool_size={self.config.get('pool_size')}"
            )
        elif target == "cache-service":
            result["details"] = (
                f"Inspected cache-service: cpu={st['cpu_pct']:.1f}% mem={st['mem_pct']:.1f}% "
                f"err={st['error_rate']:.3f} lat={st['latency_ms']:.0f}ms "
                f"ttl={self.config.get('ttl')} pool_size={self.config.get('pool_size')}"
            )
        else:
            result["details"] = (
                f"Inspected {target}: cpu={st['cpu_pct']:.1f}% mem={st['mem_pct']:.1f}% "
                f"err={st['error_rate']:.3f} lat={st['latency_ms']:.0f}ms"
            )

    def _handle_drain_traffic(self, target: str, result: dict) -> None:
        self._traffic_drains[target] += 1
        drain_count = self._traffic_drains[target]
        svc = self.state[target]

        if self._service_is_healthy(target):
            result["risk_penalty"] = result.get("risk_penalty", 0.0) + 0.06
            result["details"] = f"Traffic was drained away from {target}, but the service was already healthy"
            self._add_log(
                target,
                "WARN",
                "Traffic drain was applied to a healthy service, creating avoidable customer impact",
            )
            return

        severity_scale = 0.85 if drain_count == 1 else 0.93 if drain_count == 2 else 0.98
        svc["cpu_pct"] = max(15.0, svc["cpu_pct"] * severity_scale)
        svc["mem_pct"] = max(20.0, svc["mem_pct"] * (severity_scale + 0.05))
        svc["error_rate"] = max(0.0, svc["error_rate"] * 0.75)
        svc["latency_ms"] = max(40.0, svc["latency_ms"] * 0.72)

        if target == "db-proxy":
            for upstream in ("auth-service", "user-service", "order-service"):
                self.state[upstream]["error_rate"] = max(0.0, self.state[upstream]["error_rate"] * 0.88)
                self.state[upstream]["latency_ms"] = max(60.0, self.state[upstream]["latency_ms"] * 0.90)

        if self.scenario["task_id"] == "medium":
            result["risk_penalty"] = result.get("risk_penalty", 0.0) + 0.01
            result["details"] = f"Traffic drained away from {target}; saturation eased but capacity work is still required"
            self._add_log(
                target,
                "INFO",
                "Traffic drain reduced live load temporarily, but the hot path still needs durable remediation",
            )
        elif self.scenario["task_id"] == "expert":
            result["risk_penalty"] = result.get("risk_penalty", 0.0) + 0.015
            result["details"] = f"Traffic drained away from {target}; downstream pressure eased temporarily"
            self._add_log(
                target,
                "INFO",
                "Traffic drain activated to buy time while the cascade is stabilized",
            )
        else:
            result["risk_penalty"] = result.get("risk_penalty", 0.0) + 0.02
            result["details"] = f"Traffic drained away from {target}; load reduced temporarily"
            self._add_log(
                target,
                "INFO",
                "Traffic was diverted away from the degraded service",
            )

        if drain_count >= 3:
            result["risk_penalty"] = result.get("risk_penalty", 0.0) + 0.03
            self._add_log(
                target,
                "WARN",
                "Repeated traffic drains are causing sustained brownout risk without fixing the root cause",
            )
        result["changed"] = True

    def _handle_restart(self, target: str, result: dict) -> None:
        if self._restart_disrupts_healthy_service(target, result):
            return
        if self._expert_wrong_order_restart(target, result):
            return

        svc = self.state[target]
        svc["cpu_pct"] = max(5.0, svc["cpu_pct"] * 0.4)
        svc["mem_pct"] = max(5.0, svc["mem_pct"] * 0.3)
        svc["error_rate"] = max(0.0, svc["error_rate"] * 0.2)
        svc["latency_ms"] = max(10.0, svc["latency_ms"] * 0.5)
        self._propagate_recovery(target)
        self._medium_restart_only_masks_capacity(target, result)
        self._add_log(target, "INFO", f"{target} restarted successfully")
        result["changed"] = True
        if not result["details"]:
            result["details"] = f"{target} restarted"

    def _handle_scale_up(self, target: str, result: dict) -> None:
        if self.replicas[target] >= 5:
            result["valid"] = False
            result["details"] = f"{target} already at max replicas (5)"
            return
        self.replicas[target] += 1
        svc = self.state[target]
        svc["cpu_pct"] = max(5.0, svc["cpu_pct"] * 0.7)
        svc["mem_pct"] = max(5.0, svc["mem_pct"] * 0.85)
        svc["latency_ms"] = max(10.0, svc["latency_ms"] * 0.8)
        if self.scenario["task_id"] in {"hard", "expert"} and svc["error_rate"] >= 0.04:
            svc["error_rate"] = max(0.02, svc["error_rate"] * 0.95)
            result["risk_penalty"] = result.get("risk_penalty", 0.0) + 0.03
            result["details"] = (
                f"{target} scaled to {self.replicas[target]} replicas, but the underlying incident persists"
            )
        else:
            result["details"] = f"{target} scaled to {self.replicas[target]} replicas"
        result["changed"] = True

    def _handle_scale_down(self, target: str, result: dict) -> None:
        if self.replicas[target] <= 1:
            result["valid"] = False
            result["details"] = f"{target} already at minimum replicas (1)"
            return
        self.replicas[target] -= 1
        svc = self.state[target]
        svc["cpu_pct"] = min(100.0, svc["cpu_pct"] * 1.3)
        svc["error_rate"] = min(1.0, svc["error_rate"] + 0.015)
        svc["latency_ms"] = min(5000.0, svc["latency_ms"] * 1.2)
        result["risk_penalty"] = result.get("risk_penalty", 0.0) + 0.06
        result["changed"] = True
        result["details"] = f"{target} scaled down to {self.replicas[target]} replicas"

    def _handle_rollback(self, target: str, result: dict) -> None:
        if len(self.deploy_history) >= 2:
            prev = self.deploy_history[-2]
            for k, v in prev.get("changes", {}).items():
                if k in self.config:
                    self.config[k] = v
            self._apply_config_effects()
            self._add_log(target, "INFO", f"Rolled back {target} to deploy {prev['deploy_id']}")
            result["changed"] = True
            result["details"] = f"Rolled back to deploy {prev['deploy_id']}"
            if self.scenario["task_id"] == "hard":
                result["risk_penalty"] = result.get("risk_penalty", 0.0) + 0.08
                self._add_log(
                    target,
                    "WARN",
                    "Rollback restored an older config snapshot but left the timeout regression in place",
                )
        else:
            result["details"] = "No previous deployment to roll back to"

    def _handle_update_config(
        self, target: str, config_key, config_value, result: dict
    ) -> None:
        if not config_key or config_value is None:
            result["valid"] = False
            result["details"] = "UPDATE_CONFIG requires config_key and config_value"
            return
        self.config[config_key] = config_value
        self._apply_config_effects()
        self._add_log(target, "INFO", f"Config updated: {config_key}={config_value}")
        result["changed"] = True
        result["details"] = f"Set {config_key}={config_value}"
        if self.scenario["task_id"] == "hard":
            hard_truth = self.scenario.get("ground_truth", {})
            correct_key = hard_truth.get("correct_config_key", "db_timeout")
            correct_value = int(hard_truth.get("correct_config_value", 5000))
            if config_key != correct_key:
                result["risk_penalty"] = result.get("risk_penalty", 0.0) + 0.05
                self._add_log(
                    target,
                    "WARN",
                    f"Updated {config_key}, but the active {correct_key} regression is still unresolved",
                )
            else:
                try:
                    val_int = int(config_value)
                except (TypeError, ValueError):
                    val_int = 0
                if correct_key == "db_timeout":
                    if val_int < 1000:
                        result["risk_penalty"] = result.get("risk_penalty", 0.0) + 0.08
                    elif 1000 <= val_int < correct_value:
                        result["risk_penalty"] = result.get("risk_penalty", 0.0) + 0.03
                elif correct_key == "pool_size":
                    if val_int < 6:
                        result["risk_penalty"] = result.get("risk_penalty", 0.0) + 0.08
                    elif 6 <= val_int < correct_value:
                        result["risk_penalty"] = result.get("risk_penalty", 0.0) + 0.03

    def _handle_silence_alert(self, target: str, result: dict) -> None:
        """Silence alert. Grants a +0.02 bonus if the service is now healthy (already fixed)."""
        silenced_any = False
        for alert in self.alerts:
            if alert.service == target and not alert.silenced:
                alert.silenced = True
                silenced_any = True

        svc = self.state[target]
        service_is_healthy = (
            svc["cpu_pct"] < HEALTH_THRESHOLDS["cpu_pct"]
            and svc["mem_pct"] < HEALTH_THRESHOLDS["mem_pct"]
            and svc["error_rate"] < HEALTH_THRESHOLDS["error_rate"]
            and svc["latency_ms"] < HEALTH_THRESHOLDS["latency_ms"]
        )
        # Award bonus only once per service (avoid farming)
        if silenced_any and service_is_healthy and target not in self._silenced_fixed:
            self._silenced_fixed.add(target)
            result["silence_bonus"] = True
            result["details"] = f"Alerts silenced for {target} (service healthy — cleanup bonus awarded)"
        elif silenced_any:
            result["details"] = f"Alerts silenced for {target}"
            result["risk_penalty"] = result.get("risk_penalty", 0.0) + 0.05
            self._add_log(
                target,
                "WARN",
                "Alert was silenced before the service recovered; customer impact may still be active",
            )
        else:
            result["details"] = f"No active alerts to silence for {target}"

    # ------------------------------------------------------------------
    # Internal mechanics
    # ------------------------------------------------------------------

    def _propagate_recovery(self, service: str) -> None:
        """When a service recovers, partially improve its dependents."""
        for svc, deps in SERVICE_GRAPH.items():
            if service in deps:
                self.state[svc]["error_rate"] = max(0.0, self.state[svc]["error_rate"] * 0.7)
                self.state[svc]["latency_ms"] = max(10.0, self.state[svc]["latency_ms"] * 0.85)

    def _apply_config_effects(self) -> None:
        """Apply config changes to service state — key mechanic for Task 3."""
        db_timeout = self.config.get("db_timeout", 5000)
        pool_size = self.config.get("pool_size", 10)
        if db_timeout < 500:
            for svc in ["order-service", "user-service", "auth-service"]:
                self.state[svc]["error_rate"] = min(1.0, self.state[svc]["error_rate"] + 0.35)
                self.state[svc]["latency_ms"] = min(5000.0, self.state[svc]["latency_ms"] * 2.2)
            self.state["db-proxy"]["error_rate"] = min(
                1.0, self.state["db-proxy"]["error_rate"] + 0.20
            )
            self.state["db-proxy"]["latency_ms"] = min(
                5000.0, self.state["db-proxy"]["latency_ms"] * 1.6
            )
            self.state["api-gateway"]["latency_ms"] = min(
                5000.0, self.state["api-gateway"]["latency_ms"] * 1.25
            )
        else:
            for svc in ["order-service", "user-service", "auth-service"]:
                self.state[svc]["error_rate"] = max(0.0, self.state[svc]["error_rate"] * 0.25)
                self.state[svc]["latency_ms"] = max(40.0, self.state[svc]["latency_ms"] * 0.30)
            self.state["db-proxy"]["error_rate"] = max(
                0.0, self.state["db-proxy"]["error_rate"] * 0.35
            )
            self.state["db-proxy"]["latency_ms"] = max(
                30.0, self.state["db-proxy"]["latency_ms"] * 0.35
            )
            self.state["api-gateway"]["latency_ms"] = max(
                40.0, self.state["api-gateway"]["latency_ms"] * 0.75
            )

        if pool_size < 10:
            pressure = (10 - pool_size) / 10.0
            self.state["db-proxy"]["cpu_pct"] = min(100.0, self.state["db-proxy"]["cpu_pct"] + 18.0 * pressure)
            self.state["db-proxy"]["mem_pct"] = min(100.0, self.state["db-proxy"]["mem_pct"] + 12.0 * pressure)
            self.state["db-proxy"]["error_rate"] = min(1.0, self.state["db-proxy"]["error_rate"] + 0.12 * pressure)
            self.state["db-proxy"]["latency_ms"] = min(5000.0, self.state["db-proxy"]["latency_ms"] * (1.0 + 0.9 * pressure))
            for svc in ["order-service", "user-service", "auth-service"]:
                self.state[svc]["error_rate"] = min(1.0, self.state[svc]["error_rate"] + 0.08 * pressure)
                self.state[svc]["latency_ms"] = min(5000.0, self.state[svc]["latency_ms"] * (1.0 + 0.5 * pressure))
        else:
            self.state["db-proxy"]["cpu_pct"] = max(15.0, self.state["db-proxy"]["cpu_pct"] * 0.82)
            self.state["db-proxy"]["mem_pct"] = max(20.0, self.state["db-proxy"]["mem_pct"] * 0.88)
            for svc in ["order-service", "user-service", "auth-service"]:
                self.state[svc]["error_rate"] = max(0.0, self.state[svc]["error_rate"] * 0.82)
                self.state[svc]["latency_ms"] = max(60.0, self.state[svc]["latency_ms"] * 0.85)

    def _apply_drift(self) -> None:
        """Add seeded per-step noise for non-evaluation mode."""
        for svc in SERVICES:
            for metric, sigma in _DRIFT_SIGMA.items():
                lo, hi = _DRIFT_CLAMP[metric]
                self.state[svc][metric] = round(
                    max(lo, min(hi, self.state[svc][metric] + self._rng.gauss(0, sigma))),
                    4,
                )

    def _add_log(self, service: str, severity: str, message: str) -> None:
        self.logs.append(
            LogEntry(
                timestamp=self._now(),
                service=service,
                severity=severity,
                message=message,
            )
        )
        self.logs = self.logs[-10:]

    def _restart_disrupts_healthy_service(self, target: str, result: dict) -> bool:
        if target in self._ground_truth_services():
            return False
        if not self._service_is_healthy(target):
            return False

        svc = self.state[target]
        svc["cpu_pct"] = min(100.0, svc["cpu_pct"] * 1.10)
        svc["mem_pct"] = min(100.0, svc["mem_pct"] * 1.05)
        svc["error_rate"] = min(1.0, svc["error_rate"] + 0.04)
        svc["latency_ms"] = min(5000.0, svc["latency_ms"] * 1.40)
        for downstream, deps in SERVICE_GRAPH.items():
            if target in deps:
                self.state[downstream]["error_rate"] = min(
                    1.0, self.state[downstream]["error_rate"] + 0.015
                )
                self.state[downstream]["latency_ms"] = min(
                    5000.0, self.state[downstream]["latency_ms"] * 1.10
                )

        result["changed"] = True
        result["risk_penalty"] = result.get("risk_penalty", 0.0) + 0.08
        result["details"] = f"{target} restart caused avoidable disruption on a healthy service"
        self._add_log(target, "WARN", "Manual restart caused a brief availability dip on a healthy service")
        return True

    def _expert_wrong_order_restart(self, target: str, result: dict) -> bool:
        if self.scenario["task_id"] != "expert" or target != "db-proxy":
            return False
        if self._service_is_healthy("cache-service"):
            return False

        svc = self.state[target]
        svc["cpu_pct"] = max(50.0, svc["cpu_pct"] * 0.92)
        svc["mem_pct"] = max(50.0, svc["mem_pct"] * 0.95)
        svc["error_rate"] = max(0.12, svc["error_rate"] * 0.95)
        svc["latency_ms"] = max(500.0, svc["latency_ms"] * 0.92)
        self.state["cache-service"]["error_rate"] = min(
            1.0, self.state["cache-service"]["error_rate"] + 0.05
        )
        self.state["cache-service"]["latency_ms"] = min(
            5000.0, self.state["cache-service"]["latency_ms"] * 1.10
        )

        result["changed"] = True
        result["risk_penalty"] = result.get("risk_penalty", 0.0) + 0.10
        result["details"] = (
            "db-proxy restart was attempted before cache recovery; the database remains saturated"
        )
        self._add_log(
            target,
            "WARN",
            "db-proxy restarted, but cache misses immediately saturated the connection pool again",
        )
        return True

    def _medium_restart_only_masks_capacity(self, target: str, result: dict) -> None:
        if self.scenario["task_id"] != "medium":
            return

        root_service = self.scenario.get("ground_truth", {}).get("root_cause_service")
        if target != root_service:
            return

        svc = self.state[target]
        svc["cpu_pct"] = max(svc["cpu_pct"], 76.0)
        svc["mem_pct"] = max(svc["mem_pct"], 82.0)
        svc["error_rate"] = max(svc["error_rate"], 0.015)
        svc["latency_ms"] = max(svc["latency_ms"], 240.0)
        result["risk_penalty"] = result.get("risk_penalty", 0.0) + 0.04
        result["details"] = (
            f"{target} restarted, but the traffic surge still requires additional capacity"
        )
        self._add_log(
            target,
            "WARN",
            "Service restart reduced process-local errors, but saturation returned because capacity was not increased",
        )

    def _ground_truth_services(self) -> set[str]:
        ground_truth = self.scenario.get("ground_truth", {})
        services = {
            ground_truth.get("root_cause_service"),
            ground_truth.get("secondary_cause_service"),
        }
        return {service for service in services if service}

    def _service_is_healthy(self, service: str) -> bool:
        svc = self.state[service]
        return (
            svc["cpu_pct"] < HEALTH_THRESHOLDS["cpu_pct"]
            and svc["mem_pct"] < HEALTH_THRESHOLDS["mem_pct"]
            and svc["error_rate"] < HEALTH_THRESHOLDS["error_rate"]
            and svc["latency_ms"] < HEALTH_THRESHOLDS["latency_ms"]
        )

    def _refresh_alerts(self) -> None:
        """Rebuild the active alert list covering all four tracked metrics."""
        existing_silenced = {
            (a.service, a.metric): a.silenced
            for a in self.alerts
        }
        self.alerts = []
        for svc in SERVICES:
            st = self.state[svc]

            if st["cpu_pct"] >= HEALTH_THRESHOLDS["cpu_pct"]:
                key = (svc, "cpu_pct")
                self.alerts.append(
                    Alert(
                        alert_id=f"{svc}-cpu",
                        service=svc,
                        metric="cpu_pct",
                        threshold=HEALTH_THRESHOLDS["cpu_pct"],
                        current=st["cpu_pct"],
                        severity="WARN" if st["cpu_pct"] < 90 else "CRITICAL",
                        triggered_at=self._now(),
                        silenced=existing_silenced.get(key, False),
                    )
                )

            if st["mem_pct"] >= HEALTH_THRESHOLDS["mem_pct"]:
                key = (svc, "mem_pct")
                self.alerts.append(
                    Alert(
                        alert_id=f"{svc}-mem",
                        service=svc,
                        metric="mem_pct",
                        threshold=HEALTH_THRESHOLDS["mem_pct"],
                        current=st["mem_pct"],
                        severity="WARN" if st["mem_pct"] < 92 else "CRITICAL",
                        triggered_at=self._now(),
                        silenced=existing_silenced.get(key, False),
                    )
                )

            if st["error_rate"] >= HEALTH_THRESHOLDS["error_rate"]:
                key = (svc, "error_rate")
                self.alerts.append(
                    Alert(
                        alert_id=f"{svc}-err",
                        service=svc,
                        metric="error_rate",
                        threshold=HEALTH_THRESHOLDS["error_rate"],
                        current=st["error_rate"],
                        severity="CRITICAL" if st["error_rate"] > 0.1 else "WARN",
                        triggered_at=self._now(),
                        silenced=existing_silenced.get(key, False),
                    )
                )

            if st["latency_ms"] >= HEALTH_THRESHOLDS["latency_ms"]:
                key = (svc, "latency_ms")
                self.alerts.append(
                    Alert(
                        alert_id=f"{svc}-lat",
                        service=svc,
                        metric="latency_ms",
                        threshold=HEALTH_THRESHOLDS["latency_ms"],
                        current=st["latency_ms"],
                        severity="WARN" if st["latency_ms"] < 1000 else "CRITICAL",
                        triggered_at=self._now(),
                        silenced=existing_silenced.get(key, False),
                    )
                )

    def _derive_base_time(self) -> datetime:
        deploy_events = self.scenario.get("deploy_history", [])
        if deploy_events:
            latest = max(
                datetime.fromisoformat(event["timestamp"].replace("Z", "+00:00"))
                for event in deploy_events
            )
            return latest + timedelta(minutes=5)
        return datetime(2025, 1, 1, tzinfo=timezone.utc)

    def _now(self) -> datetime:
        return self._base_time + timedelta(minutes=self._step_index)
