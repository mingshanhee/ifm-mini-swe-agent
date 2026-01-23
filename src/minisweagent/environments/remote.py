import httpx
import logging
import uuid
from typing import Any
from pydantic import BaseModel


class RemoteEnvironmentConfig(BaseModel):
    url: str
    """Base URL of the remote service."""
    container_type: str
    """Type of container to use."""
    image: str
    """Name of the image to use."""
    timeout: int = 300
    """Timeout for HTTP requests and command execution (remote side should also respect this)."""


class RemoteEnvironment:
    def __init__(
        self,
        *,
        config_class: type = RemoteEnvironmentConfig,
        logger: logging.Logger | None = None,
        **kwargs,
    ):
        """This class executes bash commands in a remote environment via HTTP."""
        self.logger = logger or logging.getLogger("minisweagent.environment")
        self.config = config_class(**kwargs)
        self.run_id = f"run-{uuid.uuid4().hex[:8]}"
        self.running = False
        self._start_instance()

    def _start_instance(self):
        """Start the remote instance and verify success."""
        self.logger.info(f"Starting remote instance at {self.config.url} with run_id {self.run_id}")
        
        payload = {
            "container_name": self.config.image,
            "run_id": self.run_id,
            "environment_config": {
                "container_type": self.config.container_type,
                "image": self.config.image,
            },
        }
        print(payload)

        with httpx.Client(timeout=self.config.timeout) as client:
            response = client.post(
                f"{self.config.url.rstrip('/')}/start_instance",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
            if data.get("status") != "success":
                raise RuntimeError(f"Failed to start instance: {data}")
            
            self.running = True
            self.logger.info(f"Successfully started remote instance for run_id {self.run_id}")

    def execute(self, command: str, cwd: str = "", *, timeout: int | None = None) -> dict[str, Any]:
        """Execute a command in the remote instance and return the result."""
        if not self.running:
            raise RuntimeError("Instance not running")
            
        self.logger.debug(f"Executing command in {self.run_id}: {command}")
        
        payload = {
            "run_id": self.run_id,
            "cmd": command,
        }

        with httpx.Client(timeout=timeout or self.config.timeout) as client:
            response = client.post(
                f"{self.config.url.rstrip('/')}/execute_command",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
            if data.get("status") != "success":
                raise RuntimeError(f"Failed to execute command: {data}")
            return data["result"]

    def cleanup(self):
        """Stop and clean up the remote instance."""
        if self.running:
            try:
                with httpx.Client(timeout=10) as client:
                    response = client.post(
                        f"{self.config.url.rstrip('/')}/close_instance",
                        json={"run_id": self.run_id},
                    )
                    response.raise_for_status()
                self.logger.info(f"Closed remote instance for run_id {self.run_id}")
                self.running = False
            except Exception as e:
                self.logger.error(f"Failed to close remote instance for run_id {self.run_id}: {e}")

    def get_available_resources(self) -> dict[str, Any]:
        """Check current resource usage on the remote service."""
        with httpx.Client(timeout=10) as client:
            response = client.get(f"{self.config.url.rstrip('/')}/get_available_resources")
            response.raise_for_status()
            return response.json()

    def get_template_vars(self) -> dict[str, Any]:
        """Return variables for templating."""
        return self.config.model_dump() | {"run_id": self.run_id, "running": self.running}

    def __del__(self):
        """Cleanup instance when object is destroyed."""
        self.cleanup()
