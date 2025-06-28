# utils/logger.py
import logging
import os
from logging.handlers import RotatingFileHandler

def setup_logging(log_file: str, level=logging.INFO):
    """
    配置日志输出到指定文件，支持多线程环境
    """
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # 创建logger
    logger = logging.getLogger()
    logger.setLevel(level)
    
    # 清除现有的handlers，避免重复配置
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # 创建线程安全的文件处理器
    file_handler = RotatingFileHandler(
        log_file, 
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(level)
    
    # 创建格式器
    formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
    file_handler.setFormatter(formatter)
    
    # 添加处理器到logger
    logger.addHandler(file_handler)
    
    # 注释掉控制台输出，只输出到文件
    # console_handler = logging.StreamHandler()
    # console_handler.setLevel(level)
    # console_handler.setFormatter(formatter)
    # logger.addHandler(console_handler)
