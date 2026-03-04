#!/usr/bin/env python3

"""Run mini-SWE-agent on SWE-bench instances in batch mode."""
# Read this first: https://mini-swe-agent.com/latest/usage/swebench/  (usage docs)

import concurrent.futures
import json
import random
import re
import threading
import subprocess
import time
import traceback
from pathlib import Path
from typing import cast
import uuid

import typer
from jinja2 import StrictUndefined, Template
from rich.live import Live

from minisweagent.harness.constants import SWEbenchInstance
from minisweagent.harness.grading import get_eval_report
from minisweagent.harness.test_spec import make_test_spec

from minisweagent import Environment
from minisweagent.agents.default import DefaultAgent
from minisweagent.config import builtin_config_dir, get_config_from_spec
from minisweagent.environments import get_environment
from minisweagent.models import get_model
from minisweagent.run.benchmarks.utils.batch_progress import RunBatchProgressManager
from minisweagent.utils.log import add_file_handler, logger
from minisweagent.utils.serialize import UNSET, recursive_merge

_HELP_TEXT = """Run mini-SWE-agent on SWEBench instances.

[not dim]
More information about the usage: [bold green]https://mini-swe-agent.com/latest/usage/swebench/[/bold green]
[/not dim]
"""

_CONFIG_SPEC_HELP_TEXT = """Path to config files, filenames, or key-value pairs.

[bold red]IMPORTANT:[/bold red] [red]If you set this option, the default config file will not be used.[/red]
So you need to explicitly set it e.g., with [bold green]-c swebench.yaml <other options>[/bold green]

Multiple configs will be recursively merged.

Examples:

[bold red]-c model.model_kwargs.temperature=0[/bold red] [red]You forgot to add the default config file! See above.[/red]

[bold green]-c swebench.yaml -c model.model_kwargs.temperature=0.5[/bold green]

[bold green]-c swebench.yaml -c agent.max_iterations=50[/bold green]
"""

DEFAULT_CONFIG_FILE = builtin_config_dir / "benchmarks" / "swebench.yaml"

DATASET_MAPPING = {
    "gym": "SWE-Gym/SWE-Gym",
    "full": "princeton-nlp/SWE-Bench",
    "verified": "princeton-nlp/SWE-Bench_Verified",
    "lite": "princeton-nlp/SWE-Bench_Lite",
    "multimodal": "princeton-nlp/SWE-Bench_Multimodal",
    "multilingual": "swe-bench/SWE-Bench_Multilingual",
    "smith": "SWE-bench/SWE-smith",
    "_test": "klieret/swe-bench-dummy-test-dataset",
    "rebench": "nebius/SWE-rebench",
}

app = typer.Typer(rich_markup_mode="rich", add_completion=False)
_OUTPUT_FILE_LOCK = threading.Lock()


class ProgressTrackingAgent(DefaultAgent):
    """Simple wrapper around DefaultAgent that provides progress updates."""

    def __init__(self, *args, progress_manager: RunBatchProgressManager, instance_id: str = "", **kwargs):
        super().__init__(*args, **kwargs)
        self.progress_manager: RunBatchProgressManager = progress_manager
        self.instance_id = instance_id

    def step(self) -> dict:
        """Override step to provide progress updates."""
        self.progress_manager.update_instance_status(self.instance_id, f"Step {self.n_calls + 1:3d} (${self.cost:.2f})")
        return super().step()


def check_docker_image_exists(image_name: str) -> bool:
    """Check if a docker image exists locally or on remote."""
    try:
        # Check locally first
        res = subprocess.run(["docker", "image", "inspect", image_name], capture_output=True, text=True)
        if res.returncode == 0:
            return True
        # Check remote (only manifest, won't pull)
        res = subprocess.run(["docker", "manifest", "inspect", image_name], capture_output=True, text=True)
        return res.returncode == 0
    except Exception:
        return False


def get_swebench_docker_image_name(instance: dict) -> str:
    """Get the image name for a SWEBench instance."""
    image_name = instance.get("image_name", None) or instance.get("docker_image", None)
    if image_name is None:
        # Docker doesn't allow double underscore, so we replace them with a magic token
        iid = instance["instance_id"]

        if instance.get("subset", "swebench") == "gym":
            id_docker_compatible = iid.replace("__", "_s_")
            image_name = f"xingyaoww/sweb.eval.x86_64.{id_docker_compatible}:latest".lower()
        else:
            id_docker_compatible = iid.replace("__", "_1776_")
            image_name = f"swebench/sweb.eval.x86_64.{id_docker_compatible}:latest".lower()
            
    return image_name


def get_sb_environment(config: dict, instance: dict) -> Environment:
    env_config = config.setdefault("environment", {})
    env_config["environment_class"] = env_config.get("environment_class", "docker")
    image_name = get_swebench_docker_image_name(instance)
    
    if env_config["environment_class"] in ["docker", "swerex_modal"]:
        env_config["image"] = image_name
    elif env_config["environment_class"] in ["singularity", "contree"]:
        env_config["image"] = "docker://" + image_name
    elif env_config["environment_class"] == "enroot":
        env_config["image"] = "docker://" + image_name.replace("docker.io/", "")
    elif env_config["environment_class"] == "remote":
        env_config["container_type"] = "docker"
        env_config["image"] = image_name
        

    env = get_environment(env_config)
    if startup_command := config.get("run", {}).get("env_startup_command"):
        startup_command = Template(startup_command, undefined=StrictUndefined).render(**instance)
        out = env.execute(startup_command)
        if out["returncode"] != 0:
            raise RuntimeError(f"Error executing startup command: {out}")
    return env


def update_preds_file(output_path: Path, instance_id: str, model_name: str, result: str):
    """Update the output JSON file with results from a single instance."""
    with _OUTPUT_FILE_LOCK:
        output_data = {}
        if output_path.exists():
            output_data = json.loads(output_path.read_text())
        output_data[instance_id] = {
            "model_name_or_path": model_name,
            "instance_id": instance_id,
            "model_patch": result,
        }
        output_path.write_text(json.dumps(output_data, indent=2))


def remove_from_preds_file(output_path: Path, instance_id: str):
    """Remove an instance from the predictions file."""
    if not output_path.exists():
        return
    with _OUTPUT_FILE_LOCK:
        output_data = json.loads(output_path.read_text())
        if instance_id in output_data:
            del output_data[instance_id]
            output_path.write_text(json.dumps(output_data, indent=2))


def process_instance(
    instance: dict,
    output_dir: Path,
    config: dict,
    progress_manager: RunBatchProgressManager,
    run_id: str,
) -> None:
    """Process a single SWEBench instance."""
    instance_id = instance["instance_id"]
    instance_dir = output_dir / instance_id
    # avoid inconsistent state if something here fails and there's leftover previous files
    remove_from_preds_file(output_dir / "preds.json", instance_id)
    (instance_dir / f"{instance_id}.traj.json").unlink(missing_ok=True)
    model = get_model(config=config.get("model", {}))
    task = instance["problem_statement"]

    progress_manager.on_instance_start(instance_id)
    progress_manager.update_instance_status(instance_id, "Pulling/starting environment")

    agent = None
    exit_status = None
    result = None
    extra_info = {}

    try:
        env = get_sb_environment(config, instance)
        agent = ProgressTrackingAgent(
            model,
            env,
            progress_manager=progress_manager,
            instance_id=instance_id,
            **config.get("agent", {}),
        )
        info = agent.run(task)
        exit_status = info.get("exit_status")
        result = info.get("submission")
    except Exception as e:
        logger.error(f"Error processing instance {instance_id}: {e}", exc_info=True)
        exit_status, result = type(e).__name__, ""
        extra_info = {"traceback": traceback.format_exc(), "exception_str": str(e)}
    finally:
        logger.info(f"[EVAL]{instance_id} Running eval")

        eval_report = run_eval(
            instance=instance,
            env=env,
            model_patch=result,
            instance_dir=instance_dir,
            run_id=run_id
        )
        
        if env and hasattr(env, "stop"):
            env.stop()
            
        logger.info(f"[EVAL]{instance_id} Eval completed")
        
        data = save_traj(
            agent,
            instance_dir / f"{instance_id}.traj.json",
            exit_status=exit_status,
            result=result,
            extra_info=extra_info,
            instance_id=instance_id,
            eval_report=eval_report.get("eval_report", {}).get(instance_id, {}),
            print_fct=logger.info,
        )

        update_preds_file(output_dir / "preds.json", instance_id, model.config.model_name, result)
        
        progress_manager.on_instance_end(instance_id, exit_status)
        
        return data, eval_report


def filter_instances(
    instances: list[dict], *, filter_spec: str, slice_spec: str = "", shuffle: bool = False
) -> list[dict]:
    """Filter and slice a list of SWEBench instances."""
    if shuffle:
        instances = sorted(instances.copy(), key=lambda x: x["instance_id"])
        random.seed(42)
        random.shuffle(instances)
    before_filter = len(instances)
    instances = [instance for instance in instances if re.match(filter_spec, instance["instance_id"])]
    if (after_filter := len(instances)) != before_filter:
        logger.info(f"Instance filter: {before_filter} -> {after_filter} instances")
    if slice_spec:
        values = [int(x) if x else None for x in slice_spec.split(":")]
        instances = instances[slice(*values)]
        if (after_slice := len(instances)) != before_filter:
            logger.info(f"Instance slice: {before_filter} -> {after_slice} instances")
    return instances


# Custom Functions
def run_eval(
    instance: SWEbenchInstance,
    env: Environment,
    model_patch: str,
    instance_dir: str,
    run_id: str,
    is_golden: bool = False,
):
    instances = [cast(SWEbenchInstance, instance)]
    test_spec = list(map(make_test_spec, instances))[0]

    pred = {"instance_id": test_spec.instance_id, "model_patch": model_patch}

    instance_id = test_spec.instance_id

    log_dir = instance_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    report_path = log_dir / f"report_{run_id}.json"
    patch_file = log_dir / f"patch_{run_id}.diff"
    with open(patch_file, "w") as f:
        f.write(model_patch)

    logger.info(f"DEBUG test_spec {test_spec}")
    logger.info(f"DEBUG eval_script {test_spec.eval_script}")

    if is_golden:
        res = env.execute(command=f"cat > patch.diff <<'EOF'\n{model_patch}\n\nEOF")
        res = env.execute(command="git status --porcelain")
        res = env.execute(command="git apply --check patch.diff")
        res = env.execute(command="git apply patch.diff")

    eval_script = test_spec.eval_script.replace("#!/bin/bash", "")
    res = env.execute(command=eval_script)

    test_output, returncode = res["output"], res["returncode"]
    logger.info(f"[EVAL]{instance_id} returncode: {returncode}")

    test_output_path = log_dir / f"test_output_{run_id}.txt"
    with open(test_output_path, "w") as f:
        f.write(test_output)
        logger.info(f"[EVAL]{instance_id} Test output written to {test_output_path}")

    report = get_eval_report(
        test_spec=test_spec,
        prediction=pred,
        log_path=test_output_path,
        include_tests_status=True,
    )
    logger.info(f"[EVAL]{instance_id} Result: resolved: {report[instance_id]['resolved']}")

    with open(report_path, "w") as f:
        f.write(json.dumps(report, indent=4))

    return {
        "instance_id": instance_id,
        "model_patch": model_patch,
        "eval_report": report,
    }

# fmt: off
@app.command(help=_HELP_TEXT)
def main(
    subset: str = typer.Option("lite", "--subset", help="SWEBench subset to use or path to a dataset", rich_help_panel="Data selection"),
    split: str = typer.Option("dev", "--split", help="Dataset split", rich_help_panel="Data selection"),
    slice_spec: str = typer.Option("", "--slice", help="Slice specification (e.g., '0:5' for first 5 instances)", rich_help_panel="Data selection"),
    filter_spec: str = typer.Option("", "--filter", help="Filter instance IDs by regex", rich_help_panel="Data selection"),
    shuffle: bool = typer.Option(False, "--shuffle", help="Shuffle instances", rich_help_panel="Data selection"),
    output: str = typer.Option("testfolder", "-o", "--output", help="Output directory", rich_help_panel="Basic"),
    workers: int = typer.Option(1, "-w", "--workers", help="Number of worker threads for parallel processing", rich_help_panel="Basic"),
    model: str | None = typer.Option(None, "-m", "--model", help="Model to use", rich_help_panel="Basic"),
    model_class: str | None = typer.Option(None, "-c", "--model-class", help="Model class to use (e.g., 'anthropic' or 'minisweagent.models.anthropic.AnthropicModel')", rich_help_panel="Advanced"),
    run_id: str | None = typer.Option(None, "-r", "--run-id", help="Run ID", rich_help_panel="Advanced"),
    redo_existing: bool = typer.Option(False, "--redo-existing", help="Redo existing instances", rich_help_panel="Data selection"),
    config_spec: Path = typer.Option( builtin_config_dir / "extra" / "swebench.yaml", "-c", "--config", help="Path to a config file", rich_help_panel="Basic"),
    environment_class: str | None = typer.Option( None, "--environment-class", help="Environment type to use. Recommended are docker or singularity", rich_help_panel="Advanced"),
    remote_url: str = typer.Option("http://localhost:8008", "--remote-url", help="URL for RemoteEnvironment", rich_help_panel="Advanced"),
) -> None:
    # fmt: on
    output_path = Path(output) / run_id / model
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Results will be saved to {output_path}")
    add_file_handler(output_path / "minisweagent.log")

    from datasets import load_dataset

    dataset_path = DATASET_MAPPING.get(subset, subset)
    logger.info(f"Loading dataset {dataset_path}, split {split}...")
    instances = list(load_dataset(dataset_path, split=split))

    if subset == "gym":
        logger.info("Filtering gym instances by docker image availability...")
        available_instances = []
        for instance in instances:
            instance["subset"] = "gym"
            image_name = get_swebench_docker_image_name(instance)
            if check_docker_image_exists(image_name):
                available_instances.append(instance)
        
        logger.info(f"Found {len(available_instances)} available instances out of {len(instances)}")
        if len(available_instances) > 50:
            random.seed(42)
            instances = random.sample(available_instances, 50)
            logger.info("Sampled 50 random instances.")
        else:
            instances = available_instances

    instances = filter_instances(instances, filter_spec=filter_spec, slice_spec=slice_spec, shuffle=shuffle)
    if not redo_existing and (output_path / "preds.json").exists():
        existing_instances = list(json.loads((output_path / "preds.json").read_text()).keys())
        logger.info(f"Skipping {len(existing_instances)} existing instances")
        instances = [instance for instance in instances if instance["instance_id"] not in existing_instances]
    logger.info(f"Running on {len(instances)} instances...")

    logger.info(f"Building agent config from specs: {config_spec}")
    configs = [get_config_from_spec(spec) for spec in config_spec]
    configs.append({
        "environment": {"environment_class": environment_class or UNSET},
        "model": {"model_name": model or UNSET, "model_class": model_class or UNSET},
    })
    if environment_class == "remote":
        configs[-1]["environment"]["url"] = remote_url
    
    config = recursive_merge(*configs)
    
    if run_id is None:
        run_id = f"{int(time.time())}_{str(uuid.uuid4())}"

    progress_manager = RunBatchProgressManager(len(instances), output_path / f"exit_statuses_{time.time()}_{run_id}.yaml")
    results = {}

    def process_futures(futures: dict[concurrent.futures.Future, str]):
        completed = 0
        total = len(futures)
        for future in concurrent.futures.as_completed(futures):
            try:
                data, eval_report = future.result()
                completed += 1
                logger.info(f"Progress: {completed}/{total} instances completed")
                
                if data is None:
                    continue
                
                results[data["instance_id"]] = data
                results[data["instance_id"]]["eval_report"] = eval_report
            except concurrent.futures.CancelledError:
                pass
            except Exception as e:
                instance_id = futures[future]
                logger.error(f"Error in future for instance {instance_id}: {e}", exc_info=True)
                progress_manager.on_uncaught_exception(instance_id, e)

    # Run the following code if the script is called directly
    if __name__ == "__main__":
        with Live(progress_manager.render_group, refresh_per_second=4):
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(process_instance, instance, output_path, config, progress_manager, run_id): instance[
                        "instance_id"
                    ]
                    for instance in instances
                }
                
                try:
                    process_futures(futures)
                except KeyboardInterrupt:
                    logger.info("Cancelling all pending jobs. Press ^C again to exit immediately.")
                    for future in futures:
                        if not future.running() and not future.done():
                            future.cancel()
                    process_futures(futures)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(process_instance, instance, output_path, config, progress_manager, run_id): instance[
                    "instance_id"
                ]
                for instance in instances
            }
            
            try:
                process_futures(futures)
            except KeyboardInterrupt:
                logger.info("Cancelling all pending jobs. Press ^C again to exit immediately.")
                for future in futures:
                    if not future.running() and not future.done():
                        future.cancel()
                process_futures(futures)
                
    return results


if __name__ == "__main__":
    app()