"""Sandbox Runner module.

Executes code in isolated environments (Docker or Kubernetes).
"""

import base64
import logging
import time
import uuid
from typing import Optional

logger = logging.getLogger("Sandbox")

try:
    import docker
    from docker.errors import ImageNotFound
    DOCKER_AVAILABLE = True
except ImportError:
    DOCKER_AVAILABLE = False

try:
    from kubernetes import client, config
    from kubernetes.client.rest import ApiException
    K8S_AVAILABLE = True
except ImportError:
    K8S_AVAILABLE = False


# Constants
K8S_NAMESPACE = "default"
K8S_JOB_TIMEOUT_SECONDS = 300
K8S_POD_START_TIMEOUT = 30
K8S_EXECUTION_TIMEOUT = 60
DEFAULT_DOCKER_IMAGE = "python:3.9-slim"
SANDBOX_IMAGE = "pr-agent-sandbox:latest"


class SandboxRunner:
    """Executes arbitrary code in a sandboxed environment."""

    def run_tests(self, code: str, runner_type: str = "docker") -> str:
        """Executes the provided code in the specified runner.

        Args:
            code: The Python code to execute.
            runner_type: 'docker' or 'kubernetes'.

        Returns:
            The execution output or error message.
        """
        if runner_type == "kubernetes":
            return self._run_k8s_job(code)
        else:
            return self._run_docker(code)

    def _run_docker(self, code: str) -> str:
        """Runs code in a hardened Docker container."""
        if not DOCKER_AVAILABLE:
            return "Docker not installed."

        try:
            docker_client = docker.from_env()

            # Attempt to use local sandbox image, fallback to standard python
            image = SANDBOX_IMAGE
            try:
                docker_client.images.get(image)
            except ImageNotFound:
                image = DEFAULT_DOCKER_IMAGE

            # Encode code to avoid escaping issues in shell
            b64_code = base64.b64encode(code.encode("utf-8")).decode("utf-8")
            cmd = (
                f'python -c "import base64; '
                f"exec(base64.b64decode('{b64_code}').decode('utf-8'))\""
            )

            container = docker_client.containers.run(
                image,
                cmd,
                network_mode="none",
                read_only=True,
                cap_drop=["ALL"],
                mem_limit="128m",
                detach=False,
                remove=True,
            )
            return container.decode("utf-8")
        except Exception as e:
            return f"Sandbox Execution Failed (Docker not running?): {e}"

    def _run_k8s_job(self, code: str) -> str:
        """Submits the code verification as a Kubernetes Job."""
        if not K8S_AVAILABLE:
            return "Error: kubernetes python client not installed."

        if not self._load_k8s_config():
            return "Error: Could not load KubeConfig (Cluster Unreachable)."

        unique_id = str(uuid.uuid4())[:8]
        job_name = f"pr-agent-verify-{unique_id}"

        logger.info(f"Orchestrating K8s Job: {job_name}")

        try:
            batch_v1 = client.BatchV1Api()
            job = self._create_k8s_job_object(job_name, code)
            batch_v1.create_namespaced_job(body=job, namespace=K8S_NAMESPACE)
            
            logger.info(f"Dispatched Job '{job_name}'. Waiting for logs...")
            return self._wait_for_k8s_result(job_name)

        except ApiException as e:
            return f"K8s API Error: {e}"

    def _load_k8s_config(self) -> bool:
        """Loads kube config from local file or in-cluster config."""
        try:
            config.load_kube_config()
            return True
        except Exception:
            try:
                config.load_incluster_config()
                return True
            except Exception:
                return False

    def _create_k8s_job_object(self, job_name: str, code: str):
        """Creates the V1Job object."""
        return client.V1Job(
            api_version="batch/v1",
            kind="Job",
            metadata=client.V1ObjectMeta(name=job_name),
            spec=client.V1JobSpec(
                ttl_seconds_after_finished=K8S_JOB_TIMEOUT_SECONDS,
                backoff_limit=0,
                template=client.V1PodTemplateSpec(
                    spec=client.V1PodSpec(
                        containers=[
                            client.V1Container(
                                name="runner",
                                image="python:3.10-slim",
                                image_pull_policy="IfNotPresent",
                                command=["python3", "-c", code],
                                resources=client.V1ResourceRequirements(
                                    limits={"memory": "256Mi", "cpu": "500m"},
                                    requests={"memory": "128Mi", "cpu": "100m"},
                                ),
                            )
                        ],
                        restart_policy="Never",
                    )
                ),
            ),
        )

    def _wait_for_k8s_result(self, job_name: str) -> str:
        """Polls for the K8s job completion and returns logs."""
        core_v1 = client.CoreV1Api()
        pod_name = None

        # 1. Find the Pod
        for _ in range(K8S_POD_START_TIMEOUT):
            pods = core_v1.list_namespaced_pod(
                namespace=K8S_NAMESPACE, label_selector=f"job-name={job_name}"
            )
            if pods.items:
                pod_name = pods.items[0].metadata.name
                break
            time.sleep(1)

        if not pod_name:
            return "Error: K8s Pod failed to start."

        # 2. Wait for completion
        for _ in range(K8S_EXECUTION_TIMEOUT):
            pod = core_v1.read_namespaced_pod_status(
                name=pod_name, namespace=K8S_NAMESPACE
            )
            phase = pod.status.phase
            if phase in ["Succeeded", "Failed"]:
                logs = core_v1.read_namespaced_pod_log(
                    name=pod_name, namespace=K8S_NAMESPACE
                )
                return f"K8s Execution ({phase}):\n{logs}"
            time.sleep(1)

        return "Error: K8s Job Timed Out."
