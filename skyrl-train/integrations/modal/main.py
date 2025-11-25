import modal
import os
from pathlib import Path


def _find_local_repo_root() -> Path:
    """Computes full path of local SkyRL repo robustly."""
    if "SKYRL_REPO_ROOT" in os.environ:
        return Path(os.environ["SKYRL_REPO_ROOT"])

    candidates = [Path(__file__).resolve(), Path.cwd()]
    for start in candidates:
        for base in [start] + list(start.parents):
            if (base / "skyrl-train").exists() and (base / "skyrl-gym").exists():
                return base
    raise Exception("SkyRL root repo path not found")

def create_modal_volume(
    data_volume_name: str = "skyrl-data",
    cache_volume_name: str = "skyrl-cache",
):
    data_vol = modal.Volume.from_name(data_volume_name, create_if_missing=True)
    cache_vol = modal.Volume.from_name(cache_volume_name, create_if_missing=True)
    return {
        "/root/data": data_vol,
        "/root/.cache/uv": cache_vol,
        "/root/.cache/pip": cache_vol,
        "/root/.cache/huggingface": cache_vol,
    }



def create_modal_image() -> modal.Image:
    """
    Creates a Modal image using SkyRL base, mounts local repo, and installs
    lightweight extras needed for dataset preparation.
    """
    local_repo_path = _find_local_repo_root()
    print(f"Root path: {local_repo_path}")

    envs = {
        "SKYRL_REPO_ROOT": "/root/SkyRL",
        "UV_CACHE_DIR": "/root/.cache/uv",
        "PIP_CACHE_DIR": "/root/.cache/pip",
        "HF_HOME": "/root/.cache/huggingface",
        "TRANSFORMERS_CACHE": "/root/.cache/huggingface",
    }

    return (
        modal.Image.from_registry("novaskyai/skyrl-train-ray-2.51.1-py3.12-cu12.8")
        .env(envs)
        # 👇 these are missing in the base image, needed for prepare_modal_text_split.py
        .pip_install("datasets", "transformers")
        .add_local_dir(
            local_path=str(local_repo_path),
            remote_path="/root/SkyRL",
            ignore=[
                ".venv",
                "*.pyc",
                "__pycache__",
                ".git",
                "*.egg-info",
                ".pytest_cache",
                "node_modules",
                ".DS_Store",
            ],
        )
    )


def create_modal_volume(volume_name: str = "skyrl-data") -> dict[str, modal.Volume]:
    """Creates volume to attach to container."""
    data_volume = modal.Volume.from_name(volume_name, create_if_missing=True)
    return {"/root/data": data_volume}


app = modal.App(os.getenv("MODAL_APP_NAME", "my_skyrl_app"))
image = create_modal_image()
volume = create_modal_volume()


def _run_in_skyrl_train(command: str, *, start_ray: bool):
    """Shared runner used by both CPU and GPU functions."""
    import subprocess
    import os

    repo_root = os.environ.get("SKYRL_REPO_ROOT", "/root/SkyRL")
    print(f"Container repo root: {repo_root}")
    print(f"Initial working directory: {os.getcwd()}")

    run_command_dir = os.path.join(repo_root, "skyrl-train")
    os.chdir(run_command_dir)
    print(f"Changed to directory: {os.getcwd()}")

    if start_ray:
        # Ensure skyrl-gym exists so uv editable resolution works
        gym_src = os.path.join("..", "skyrl-gym")
        gym_dst = os.path.join(".", "skyrl-gym")
        if not os.path.exists(gym_dst):
            if os.path.exists(gym_src):
                print("Copying ../skyrl-gym into working_dir for uv packaging")
                subprocess.run(f"cp -r {gym_src} {gym_dst}", shell=True, check=True)
            else:
                raise Exception("Cannot find skyrl-gym source")

        print("Initializing ray cluster in command line")
        subprocess.run("ray start --head", shell=True, check=True)
        os.environ["RAY_ADDRESS"] = "auto"

    print(f"Running command: {command}")
    print(f"Working directory: {os.getcwd()}")
    print("=" * 60)

    process = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )

    for line in process.stdout:
        print(line, end="")

    returncode = process.wait()
    print("=" * 60)

    if returncode != 0:
        raise Exception(f"Command failed with exit code {returncode}")


# ---- CPU-only runner (no Ray, no GPU) ----
@app.function(
    image=image,
    gpu=None,
    volumes=volume,
    timeout=int(os.environ.get("MODAL_CPU_TIMEOUT_SEC", 60 * 60)),  # 1h default
    cpu=int(os.environ.get("MODAL_CPU", 8)),
    memory=int(os.environ.get("MODAL_CPU_MEM_MIB", 32 * 1024)),
)
def run_cpu(command: str):
    _run_in_skyrl_train(command, start_ray=False)


# ---- GPU runner (Ray on, for training) ----
@app.function(
    image=image,
    gpu=os.environ.get("MODAL_GPU", "L4:1"),
    volumes=volume,
    timeout=int(os.environ.get("MODAL_GPU_TIMEOUT_SEC", 60 * 60 * 24)),  # 24h default
    cpu=int(os.environ.get("MODAL_GPU_CPU", 16)),
    memory=int(os.environ.get("MODAL_GPU_MEM_MIB", 128 * 1024)),
    secrets=[modal.Secret.from_name("wandb")]
)
def run_gpu(command: str):
    _run_in_skyrl_train(command, start_ray=True)


@app.local_entrypoint()
def main(
    command: str = "nvidia-smi",
    use_gpu: bool = True,
):
    """
    Run a command on Modal inside skyrl-train/.

    Examples:
      # CPU command
      modal run main.py --use-gpu false --command "python ..."

      # GPU command
      MODAL_GPU=L4:4 modal run main.py --command "bash ..."
    """
    print(f"{'=' * 5} Submitting to Modal (use_gpu={use_gpu}): {command} {'=' * 5}")
    if use_gpu:
        run_gpu.remote(command)
    else:
        run_cpu.remote(command)
    print(f"\n{'=' * 5} Command completed successfully {'=' * 5}")
