# utils/logger.py
import logging
import os

def setup_logging(log_file: str, level=logging.INFO):
    """
    配置日志输出到指定文件
    """
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logging.basicConfig(
        filename=log_file,
        level=level,
        format='%(asctime)s %(levelname)s: %(message)s'
    )
