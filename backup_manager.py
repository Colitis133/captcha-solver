import json
import os
import subprocess
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

try:  # Optional dependency when backups are enabled
    from azure.storage.blob import BlobServiceClient
    from azure.core.exceptions import ResourceExistsError
except ImportError:  # pragma: no cover - handled at runtime if backups disabled
    BlobServiceClient = None
    ResourceExistsError = None


class BackupManager:
    """Handles baseline and incremental backups to Azure Blob Storage."""

    def __init__(
        self,
        base_dir: Path,
        logger,
        enabled: bool = False,
        run_folder_prefix: str = "run",
        baseline_epoch: int = 3,
        incremental_interval: int = 10,
        incremental_directories: Optional[List[str]] = None,
        incremental_files: Optional[List[str]] = None,
        azure_config: Optional[Dict[str, str]] = None,
    ) -> None:
        self.base_dir = Path(base_dir)
        self.logger = logger
        self.enabled = enabled
        self.run_folder_prefix = run_folder_prefix
        self.baseline_epoch = baseline_epoch
        self.incremental_interval = incremental_interval
        self.incremental_directories = incremental_directories or ["checkpoints", "models", "logs"]
        self.incremental_files = incremental_files or ["config.json", "backup_state.json", "requirements-lock.txt"]

        azure_config = azure_config or {}
        self.azure_connection_string = azure_config.get("connection_string")
        connection_env = azure_config.get("connection_string_env")
        if not self.azure_connection_string and connection_env:
            self.azure_connection_string = os.getenv(connection_env)
        self.azure_container = azure_config.get("container", "captcha-backups")
        self.azure_path_prefix = azure_config.get("path_prefix", "").strip("/")

        self.backup_dir = self.base_dir / "backups"
        self.backup_dir.mkdir(exist_ok=True)

        self.state_path = self.base_dir / "backup_state.json"
        self.state: Dict[str, object] = {
            "baseline_uploaded": False,
            "baseline_epoch": self.baseline_epoch,
            "uploaded_epochs": [],
            "run_label": None,
        }
        self._blob_service_client = None
        self._container_client = None

        if self.enabled:
            self._load_state()
            self._save_state()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def maybe_backup(self, epoch_number: int, checkpoint_path: Optional[Path], best_model_path: Optional[Path]) -> None:
        """Trigger baseline or incremental backups when criteria are met."""
        if not self.enabled:
            return

        if BlobServiceClient is None:
            raise ImportError(
                "Azure Storage libraries missing. Install azure-storage-blob to enable backups."
            )

        self._ensure_storage_ready()

        if not self.state.get("run_label"):
            self.state["run_label"] = f"{self.run_folder_prefix}-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"
            self._save_state()

        if not self.state["baseline_uploaded"] and epoch_number >= self.baseline_epoch:
            self._perform_baseline_backup(epoch_number)
            return

        if (
            self.state["baseline_uploaded"]
            and epoch_number % self.incremental_interval == 0
            and epoch_number not in self.state["uploaded_epochs"]
        ):
            if not checkpoint_path or not Path(checkpoint_path).exists():
                self.logger.warning(
                    "Skipping incremental backup for epoch %s (checkpoint missing).", epoch_number
                )
                return
            self._perform_incremental_backup(epoch_number, checkpoint_path, best_model_path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _perform_baseline_backup(self, epoch_number: int) -> None:
        self.logger.info("Creating baseline backup for epoch %s", epoch_number)
        zip_path = self._build_zip_path(f"baseline_epoch{epoch_number}.zip")
        self._generate_requirements_lock()
        self._save_state()
        self._zip_baseline(zip_path)
        self._upload_zip(zip_path, remote_name=zip_path.name)
        self.state["baseline_uploaded"] = True
        self.state["baseline_epoch"] = epoch_number
        self._save_state()
        self._cleanup(zip_path)

    def _perform_incremental_backup(
        self,
        epoch_number: int,
        checkpoint_path: Path,
        best_model_path: Optional[Path],
    ) -> None:
        self.logger.info("Creating incremental backup for epoch %s", epoch_number)
        zip_path = self._build_zip_path(f"increment_epoch{epoch_number}.zip")
        self._generate_requirements_lock()
        self._save_state()
        include_dirs = [d for d in self.incremental_directories if (self.base_dir / d).exists()]
        include_files = [f for f in self.incremental_files if (self.base_dir / f).exists()]

        if checkpoint_path:
            include_files.append(os.path.relpath(Path(checkpoint_path), self.base_dir))
        if best_model_path and Path(best_model_path).exists():
            rel_best = os.path.relpath(Path(best_model_path), self.base_dir)
            if rel_best not in include_files:
                include_files.append(rel_best)

        unique_files = list(dict.fromkeys(include_files))
        self._zip_selected(zip_path, include_dirs, unique_files)
        self._upload_zip(zip_path, remote_name=zip_path.name)
        uploaded_epochs: List[int] = self.state.get("uploaded_epochs", [])
        uploaded_epochs.append(epoch_number)
        self.state["uploaded_epochs"] = uploaded_epochs
        self._save_state()
        self._cleanup(zip_path)

    def _ensure_storage_ready(self) -> None:
        if self._blob_service_client is None:
            if not self.azure_connection_string:
                raise ValueError(
                    "Azure storage connection string not provided. Set it in config or via environment variable."
                )
            self._blob_service_client = BlobServiceClient.from_connection_string(self.azure_connection_string)

        if self._container_client is None:
            if not self.azure_container:
                raise ValueError("Azure container name is required for backups.")
            self._container_client = self._blob_service_client.get_container_client(self.azure_container)
            try:
                self._container_client.create_container()
                self.logger.info("Created Azure container '%s'.", self.azure_container)
            except ResourceExistsError:
                pass

    def _upload_zip(self, local_path: Path, remote_name: str) -> None:
        blob_name = self._build_blob_name(remote_name)
        blob_client = self._container_client.get_blob_client(blob_name)
        with open(local_path, "rb") as data:
            blob_client.upload_blob(data, overwrite=True)
        self.logger.info(
            "Uploaded backup %s to container '%s' as blob '%s'.",
            remote_name,
            self.azure_container,
            blob_name,
        )

    def _build_zip_path(self, filename: str) -> Path:
        timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        run_label = self.state.get("run_label") or f"{self.run_folder_prefix}-{timestamp}"
        return self.backup_dir / f"{run_label}_{filename}"

    def _build_blob_name(self, filename: str) -> str:
        parts = []
        if self.azure_path_prefix:
            parts.append(self.azure_path_prefix)
        if self.state.get("run_label"):
            parts.append(self.state["run_label"])
        parts.append(filename)
        return "/".join(parts)

    def _zip_baseline(self, destination: Path) -> None:
        with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(self.base_dir):
                rel_root = os.path.relpath(root, self.base_dir)
                if self._should_skip(rel_root):
                    dirs[:] = []
                    continue
                for file_name in files:
                    rel_path = os.path.normpath(os.path.join(rel_root, file_name))
                    if self._should_skip(rel_path):
                        continue
                    abs_path = os.path.join(root, file_name)
                    arcname = rel_path if rel_path != "." else file_name
                    zipf.write(abs_path, arcname)

    def _zip_selected(self, destination: Path, directories: List[str], files: List[str]) -> None:
        with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as zipf:
            for rel_dir in directories:
                abs_dir = self.base_dir / rel_dir
                if not abs_dir.exists():
                    continue
                for file_path in abs_dir.rglob("*"):
                    if file_path.is_dir():
                        continue
                    rel_path = file_path.relative_to(self.base_dir)
                    zipf.write(file_path, rel_path.as_posix())
            for rel_file in files:
                abs_file = self.base_dir / rel_file
                if abs_file.exists() and abs_file.is_file():
                    rel_path = abs_file.relative_to(self.base_dir)
                    zipf.write(abs_file, rel_path.as_posix())

    def _should_skip(self, rel_path: str) -> bool:
        if rel_path in (".", ""):
            return False
        normalized = rel_path.strip("./")
        skip_prefixes = {".git", "backups", "__pycache__", "dataset"}
        return any(normalized.startswith(prefix) for prefix in skip_prefixes)

    def _generate_requirements_lock(self) -> None:
        lock_path = self.base_dir / "requirements-lock.txt"
        try:
            result = subprocess.run(
                ["pip", "freeze"],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            lock_path.write_text(result.stdout)
        except subprocess.CalledProcessError as exc:
            self.logger.warning("pip freeze failed: %s", exc)

    def _load_state(self) -> None:
        if self.state_path.exists():
            try:
                data = json.loads(self.state_path.read_text())
                self.state.update(data)
            except json.JSONDecodeError:
                self.logger.warning("backup_state.json is corrupt; starting with fresh state")

    def _save_state(self) -> None:
        self.state_path.write_text(json.dumps(self.state, indent=2))

    def _cleanup(self, path: Path) -> None:
        try:
            path.unlink()
        except OSError:
            self.logger.warning("Could not delete temporary backup %s", path)
