from datetime import date
from citylearn.utils.paths import PROJECT_ROOT
from pathlib import Path
import os, sys
from abc import ABC, abstractmethod
import logging

logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)


class Renderer(ABC):


    _directory : Path
    _flag : str
    _session_name : str
    _start_date : date
    _enabled : bool
    _FALLBACK_DIR = "simulator_output"


    def __init__(self, directory: str, flag : str, session_name : str,
                start_date : date, enabled = True):
        
        self._directory = Path(PROJECT_ROOT, directory).expanduser()
        self._create_output_dir()
        
        self._flag = flag
        self._session_name = session_name
        self._start_date = start_date
        self._enabled = enabled

    @abstractmethod
    def export_csv(self, filename : str, data : dict):
        pass

    def _create_output_dir(self):

        if self._directory.exists() and os.access(self._directory, os.W_OK):

            logger.info(f"[{self._directory}] directory already exists and is being reused by the renderer.")

            for csv_file in self._directory.glob(".csv"):
                csv_file.unlink()
            return

        try:
            self._directory.mkdir(parents=True, exist_ok=True)
        except PermissionError:
            logger.warning(f"User lacks permission to create [{self._directory}] directory.")

            self._directory = (Path.cwd() / self._FALLBACK_DIR).expanduser()
            self._directory.mkdir(parents=True, exist_ok=True)

        logger.info(f"Renderer will provide output under [{self._directory}] directory.")


    def export_final_kpis():
        pass

    def parse_start_date():
        pass
    
    def get_iso_timestamp():
        pass
    def override_timestep():
        pass
    def restore_timestemp():
        pass
    def reset_time_tracking():
        pass

print(PROJECT_ROOT)