import logging
import os
import pathlib

from napistu.gcs.constants import INIT_DATA_DIR_MSG

logger = logging.getLogger(__name__)


def _initialize_data_dir(data_dir: str, init_msg: str = INIT_DATA_DIR_MSG) -> None:
    """Create a data directory if it doesn't exist."""

    if not os.path.isdir(data_dir):

        logger.warning(init_msg.format(data_dir=data_dir))

        # Artifact directory not found; creating {parentdir}")
        logger.warning(f"Trying to create {data_dir}")
        pathlib.Path(data_dir).mkdir(parents=True, exist_ok=True)

    return None
