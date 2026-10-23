from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import subprocess
import tempfile
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import yaml
import asyncio

logger = logging.getLogger(__name__)


class DeploymentStage(Enum):
    PREPARE = "prepare"
    BUILD = "build"
    TEST = "test"
    DEPLOY = "deploy"
    VERIFY = "verify"
    ROLLBACK = "rollback"
    COMPLETE = "complete"


class DeploymentStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"
    CANCELLED = "cancelled"


class Environment(Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


@dataclass(frozen=True, slots=True)
class DeploymentTarget:
    name: str
    environment: Environment
    hosts: List[str]
    variables: Dict[str, str] = field(default_factory=dict)
    secrets: Dict[str, str] = field(default_factory=dict)
    health_check_url: str = ""
    rollback_on_failure: bool = True


@dataclass
class DeploymentStep:
    name: str
    stage: DeploymentStage
    command: str
    working_dir: str = "."
    environment: Dict[str, str] = field(default_factory=dict)
    timeout: int = 300
    retries: int = 0
    condition: Optional[str] = None
    rollback_command: Optional[str] = None
    depends_on: List[str] = field(default_factory=list)


@dataclass
class Deployment:
    id: str
    name: str
    version: str
    target: DeploymentTarget
    steps: List[DeploymentStep]
    status: DeploymentStatus = DeploymentStatus.PENDING
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    current_step: int = 0
    logs: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class DeploymentEngine:
    def __init__(self):
        self.deployments: Dict[str, Deployment] = {}
        self.targets: Dict[str, DeploymentTarget] = {}
        self.templates: Dict[str, List[DeploymentStep]] = {}
        self._running = False

    def register_target(self, target: DeploymentTarget):
        self.targets[target.name] = target

    def register_template(self, name: str, steps: List[DeploymentStep]):
        self.templates[name] = steps

    def create_deployment(
        self,
        name: str,
        version: str,
        target_name: str,
        template: str = None,
        custom_steps: List[DeploymentStep] = None,
    ) -> Deployment:
        target = self.targets.get(target_name)
        if not target:
            raise ValueError(f"Target {target_name} not found")

        steps = []
        if template and template in self.templates:
            steps.extend(self.templates[template])
        if custom_steps:
            steps.extend(custom_steps)

        deployment = Deployment(
            id=str(uuid.uuid4()),
            name=name,
            version=version,
            target=target,
            steps=steps,
        )
        self.deployments[deployment.id] = deployment
        return deployment

    async def execute(self, deployment_id: str) -> Deployment:
        deployment = self.deployments.get(deployment_id)
        if not deployment:
            raise ValueError(f"Deployment {deployment_id} not found")

        deployment.status = DeploymentStatus.RUNNING
        deployment.started_at = datetime.utcnow()

        try:
            for i, step in enumerate(deployment.steps):
                deployment.current_step = i
                await self._execute_step(deployment, step)

            deployment.status = DeploymentStatus.SUCCESS
        except Exception as e:
            deployment.status = DeploymentStatus.FAILED
            deployment.logs.append(f"Deployment failed: {str(e)}")
            if deployment.target.rollback_on_failure:
                await self._rollback(deployment)
            raise
        finally:
            deployment.completed_at = datetime.utcnow()

        return deployment

    async def _execute_step(self, deployment: Deployment, step: DeploymentStep):
        deployment.logs.append(f"[{datetime.utcnow().isoformat()}] Starting step: {step.name}")

        if step.condition:
            if not self._evaluate_condition(step.condition, deployment):
                deployment.logs.append(f"Step {step.name} skipped (condition not met)")
                return

        for attempt in range(step.retries + 1):
            try:
                deployment.logs.append(f"Executing: {step.command} (attempt {attempt + 1})")
                await self._run_command(
                    step.command,
                    step.working_dir,
                    {**deployment.target.variables, **step.environment},
                    step.timeout,
                )
                deployment.logs.append(f"Step {step.name} completed successfully")
                return
            except Exception as e:
                if attempt < step.retries:
                    deployment.logs.append(f"Attempt {attempt + 1} failed: {str(e)}, retrying...")
                    await asyncio.sleep(2 ** attempt)
                else:
                    raise

    def _evaluate_condition(self, condition: str, deployment: Deployment) -> bool:
        try:
            return eval(condition, {"deployment": deployment})
        except Exception:
            return False

    async def _run_command(
        self,
        command: str,
        working_dir: str,
        environment: Dict[str, str],
        timeout: int,
    ) -> Tuple[int, str, str]:
        env = {**os.environ, **environment}
        process = await asyncio.create_subprocess_shell(
            command,
            cwd=working_dir,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout,
            )
            return process.returncode, stdout.decode(), stderr.decode()
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            raise TimeoutError(f"Command timed out after {timeout}s")

    async def _rollback(self, deployment: Deployment):
        deployment.status = DeploymentStatus.ROLLBACK
        deployment.logs.append("Starting rollback...")

        for step in reversed(deployment.steps[:deployment.current_step]):
            if step.rollback_command:
                try:
                    deployment.logs.append(f"Rolling back: {step.name}")
                    await self._run_command(
                        step.rollback_command,
                        step.working_dir,
                        deployment.target.variables,
                        60,
                    )
                    deployment.logs.append(f"Rollback {step.name} completed")
                except Exception as e:
                    deployment.logs.append(f"Rollback {step.name} failed: {e}")

        deployment.status = DeploymentStatus.ROLLED_BACK


class DeploymentPipeline:
    def __init__(self, engine: DeploymentEngine):
        self.engine = engine

    def add_stage(self, name: str, steps: List[DeploymentStep]):
        self.engine.register_template(name, steps)

    async def deploy(
        self,
        name: str,
        version: str,
        target: str,
        environment: Environment = Environment.PRODUCTION,
    ) -> Deployment:
        target_obj = self.engine.targets.get(target)
        if not target_obj:
            raise ValueError(f"Target {target} not found")

        target_obj.environment = environment.value

        deployment = self.engine.create_deployment(
            name=name,
            version=version,
            target_name=target,
            template=environment.value,
        )

        return await self.engine.execute(deployment.id)


class CI_CDPipeline:
    def __init__(self):
        self.stages: List[Dict[str, Any]] = []
        self.artifacts: Dict[str, str] = {}
        self.cache: Dict[str, Any] = {}

    def add_stage(
        self,
        name: str,
        script: str,
        image: str = "python:3.12",
        needs: List[str] = None,
        artifacts: List[str] = None,
        cache_key: str = None,
        only: List[str] = None,
        except_refs: List[str] = None,
    ):
        self.stages.append({
            "name": name,
            "script": script,
            "image": image,
            "needs": needs or [],
            "artifacts": artifacts or [],
            "cache_key": cache_key,
            "only": only or [],
            "except": except_refs or [],
        })

    def generate_gitlab_ci(self) -> str:
        yaml_data = {
            "stages": [s["name"] for s in self.stages],
            "variables": {
                "PIP_CACHE_DIR": "$CI_PROJECT_DIR/.pip-cache",
                "DOCKER_DRIVER": "overlay2",
            },
            "cache": {
                "paths": [".pip-cache/", "__pycache__/", ".mypy_cache/", ".pytest_cache/"],
            },
            "before_script": [
                "python -m pip install --upgrade pip",
                "pip install -r requirements.txt",
            ],
        }

        for stage in self.stages:
            job = {
                "stage": stage["name"],
                "image": stage["image"],
                "script": stage["script"].split("\n"),
            }
            if stage["needs"]:
                job["needs"] = stage["needs"]
            if stage["artifacts"]:
                job["artifacts"] = {"paths": stage["artifacts"], "expire_in": "1 week"}
            if stage["cache_key"]:
                job["cache"] = {"key": stage["cache_key"], "paths": [".pip-cache/"]}
            if stage["only"]:
                job["only"] = stage["only"]
            if stage["except"]:
                job["except"] = stage["except"]

            yaml_data[stage["name"]] = job

        return yaml.dump(yaml_data, default_flow_style=False)

    def generate_github_actions(self) -> str:
        yaml_data = {
            "name": "CI/CD Pipeline",
            "on": {
                "push": {"branches": ["main", "develop"]},
                "pull_request": {"branches": ["main"]},
            },
            "jobs": {},
        }

        for stage in self.stages:
            job_name = stage["name"].replace(" ", "-").lower()
            yaml_data["jobs"][job_name] = {
                "runs-on": "ubuntu-latest",
                "container": {"image": stage["image"]},
                "steps": [
                    {"uses": "actions/checkout@v4"},
                    {"name": "Set up Python", "uses": "actions/setup-python@v5", "with": {"python-version": "3.12"}},
                    {"name": "Install dependencies", "run": "pip install -r requirements.txt"},
                    {"name": "Run script", "run": stage["script"]},
                ],
            }
            if stage["needs"]:
                yaml_data["jobs"][job_name]["needs"] = stage["needs"]

        return yaml.dump(yaml_data, default_flow_style=False)


class InfrastructureProvisioner:
    def __init__(self):
        self.resources: Dict[str, "InfrastructureResource"] = {}
        self.state: Dict[str, Any] = {}

    def add_resource(self, resource: "InfrastructureResource"):
        self.resources[resource.name] = resource

    async def plan(self) -> List[Dict[str, Any]]:
        plan = []
        for name, resource in self.resources.items():
            plan.append({
                "action": "create",
                "resource": name,
                "type": resource.type,
                "config": resource.config,
            })
        return plan

    async def apply(self, plan: List[Dict[str, Any]] = None):
        if plan is None:
            plan = await self.plan()

        for item in plan:
            resource = self.resources[item["resource"]]
            await resource.create(item["config"])

    async def destroy(self):
        for name in reversed(list(self.resources.keys())):
            resource = self.resources[name]
            await resource.destroy()


class InfrastructureResource:
    def __init__(self, name: str, type: str, config: Dict[str, Any]):
        self.name = name
        self.type = type
        self.config = config

    async def create(self, config: Dict[str, Any]):
        pass

    async def destroy(self):
        pass


class DockerResource(InfrastructureResource):
    def __init__(self, name: str, image: str, ports: Dict[int, int] = None, volumes: Dict[str, str] = None, env: Dict[str, str] = None):
        super().__init__(name, "docker", {"image": image, "ports": ports or {}, "volumes": volumes or {}, "env": env or {}})
        self.container_id: Optional[str] = None

    async def create(self, config: Dict[str, Any]):
        cmd = ["docker", "run", "-d", "--name", self.name]
        for host_port, container_port in self.config.get("ports", {}).items():
            cmd.extend(["-p", f"{host_port}:{container_port}"])
        for host_path, container_path in self.config.get("volumes", {}).items():
            cmd.extend(["-v", f"{host_path}:{container_path}"])
        for key, value in self.config.get("env", {}).items():
            cmd.extend(["-e", f"{key}={value}"])
        cmd.append(self.config["image"])

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Docker run failed: {result.stderr}")
        self.container_id = result.stdout.strip()

    async def destroy(self):
        if self.container_id:
            subprocess.run(["docker", "rm", "-f", self.container_id])


class KubernetesResource(InfrastructureResource):
    def __init__(self, name: str, manifests: List[str], namespace: str = "default"):
        super().__init__(name, "kubernetes", {"manifests": manifests, "namespace": namespace})

    async def create(self, config: Dict[str, Any]):
        for manifest in self.config["manifests"]:
            subprocess.run(["kubectl", "apply", "-f", manifest, "-n", self.config["namespace"]], check=True)

    async def destroy(self):
        for manifest in reversed(self.config["manifests"]):
            subprocess.run(["kubectl", "delete", "-f", manifest, "-n", self.config["namespace"]])


class TerraformResource(InfrastructureResource):
    def __init__(self, name: str, working_dir: str, variables: Dict[str, str] = None):
        super().__init__(name, "terraform", {"working_dir": working_dir, "variables": variables or {}})

    async def create(self, config: Dict[str, Any]):
        working_dir = self.config["working_dir"]
        subprocess.run(["terraform", "init"], cwd=working_dir, check=True)
        var_args = []
        for k, v in self.config.get("variables", {}).items():
            var_args.extend(["-var", f"{k}={v}"])
        subprocess.run(["terraform", "apply", "-auto-approve"] + var_args, cwd=working_dir, check=True)

    async def destroy(self):
        working_dir = self.config["working_dir"]
        var_args = []
        for k, v in self.config.get("variables", {}).items():
            var_args.extend(["-var", f"{k}={v}"])
        subprocess.run(["terraform", "destroy", "-auto-approve"] + var_args, cwd=working_dir, check=True)


class HelmResource(InfrastructureResource):
    def __init__(self, name: str, chart: str, values: Dict[str, Any] = None, namespace: str = "default"):
        super().__init__(name, "helm", {"chart": chart, "values": values or {}, "namespace": namespace})
        self.release_name = name

    async def create(self, config: Dict[str, Any]):
        cmd = ["helm", "upgrade", "--install", self.release_name, self.config["chart"], "-n", self.config["namespace"]]
        for key, value in self.config.get("values", {}).items():
            cmd.extend(["--set", f"{key}={value}"])
        subprocess.run(cmd, check=True)

    async def destroy(self):
        subprocess.run(["helm", "uninstall", self.release_name, "-n", self.config["namespace"]])


class DeploymentOrchestrator:
    def __init__(self):
        self.engine = DeploymentEngine()
        self.pipeline = CI_CDPipeline()
        self.provisioner = InfrastructureProvisioner()
        self.hooks: Dict[str, List[Callable]] = defaultdict(list)

    def add_hook(self, event: str, hook: Callable):
        self.hooks[event].append(hook)

    async def _run_hooks(self, event: str, *args, **kwargs):
        for hook in self.hooks.get(event, []):
            if inspect.iscoroutinefunction(hook):
                await hook(*args, **kwargs)
            else:
                hook(*args, **kwargs)

    async def full_deploy(
        self,
        app_name: str,
        version: str,
        environment: Environment,
        infrastructure: List[InfrastructureResource] = None,
    ) -> Deployment:
        await self._run_hooks("pre_deploy", app_name, version, environment)

        if infrastructure:
            for resource in infrastructure:
                self.provisioner.add_resource(resource)
            await self.provisioner.apply()

        target_name = environment.value
        if target_name not in self.engine.targets:
            self.engine.register_target(DeploymentTarget(
                name=target_name,
                environment=environment,
                hosts=[],
            ))

        deployment = self.engine.create_deployment(
            name=app_name,
            version=version,
            target_name=target_name,
            template=environment.value,
        )

        result = await self.engine.execute(deployment.id)

        await self._run_hooks("post_deploy", app_name, version, environment, result)

        return result

    async def rollback(self, deployment_id: str):
        deployment = self.engine.deployments.get(deployment_id)
        if deployment:
            await self.engine._rollback(deployment)


def create_default_pipeline() -> CI_CDPipeline:
    pipeline = CI_CDPipeline()

    pipeline.add_stage(
        name="lint",
        script="ruff check . && mypy . --ignore-missing-imports",
        image="python:3.12",
    )

    pipeline.add_stage(
        name="test",
        script="pytest -q --cov=core --cov-report=xml",
        image="python:3.12",
        needs=["lint"],
        artifacts=["coverage.xml"],
    )

    pipeline.add_stage(
        name="build",
        script="docker build -t $IMAGE_NAME:$VERSION .",
        image="docker:latest",
        needs=["test"],
        only=["main", "develop"],
    )

    pipeline.add_stage(
        name="deploy-staging",
        script="helm upgrade --install $APP_NAME ./helm -n staging --set image.tag=$VERSION",
        image="alpine/helm:latest",
        needs=["build"],
        only=["develop"],
    )

    pipeline.add_stage(
        name="deploy-production",
        script="helm upgrade --install $APP_NAME ./helm -n production --set image.tag=$VERSION",
        image="alpine/helm:latest",
        needs=["build"],
        only=["main"],
    )

    return pipeline


def create_production_deployment(
    app_name: str,
    version: str,
    hosts: List[str],
    docker_image: str,
    env_vars: Dict[str, str],
) -> Deployment:
    engine = DeploymentEngine()

    target = DeploymentTarget(
        name="production",
        environment=Environment.PRODUCTION,
        hosts=hosts,
        variables=env_vars,
        health_check_url=f"http://{hosts[0]}/health",
        rollback_on_failure=True,
    )
    engine.register_target(target)

    steps = [
        DeploymentStep(
            name="pull_image",
            stage=DeploymentStage.PREPARE,
            command=f"docker pull {docker_image}:{version}",
        ),
        DeploymentStep(
            name="stop_old_container",
            stage=DeploymentStage.DEPLOY,
            command=f"docker stop {app_name} || true",
            rollback_command=f"docker start {app_name}_old || true",
        ),
        DeploymentStep(
            name="rename_old_container",
            stage=DeploymentStage.DEPLOY,
            command=f"docker rename {app_name} {app_name}_old || true",
            rollback_command=f"docker rename {app_name}_old {app_name} || true",
        ),
        DeploymentStep(
            name="start_new_container",
            stage=DeploymentStage.DEPLOY,
            command=f"docker run -d --name {app_name} --restart unless-stopped "
                    f"{' '.join(f'-e {k}={v}' for k, v in env_vars.items())} "
                    f"{docker_image}:{version}",
            rollback_command=f"docker stop {app_name} && docker rename {app_name}_old {app_name}",
        ),
        DeploymentStep(
            name="health_check",
            stage=DeploymentStage.VERIFY,
            command=f"curl -f http://localhost/health || exit 1",
            timeout=60,
            retries=5,
        ),
        DeploymentStep(
            name="cleanup_old",
            stage=DeploymentStage.COMPLETE,
            command=f"docker rm {app_name}_old || true",
        ),
    ]

    deployment = Deployment(
        id=str(uuid.uuid4()),
        name=app_name,
        version=version,
        target=target,
        steps=steps,
    )

    return deployment