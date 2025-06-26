import os
import yaml
import logging
import json
import time
import argparse
from typing import List, Dict, Any, Set
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

from api.batch_processor import process_batch
from utils.logger import setup_logging

def load_config(path: str = "config/config.yaml") -> Dict[str, Any]:
    """加载配置文件，支持环境变量替换"""
    with open(path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 替换环境变量
    if isinstance(config.get('openai_api_key'), str) and config['openai_api_key'].startswith('${'):
        env_var = config['openai_api_key'][2:-1]  # 移除 ${}
        config['openai_api_key'] = os.getenv(env_var)
        if not config['openai_api_key']:
            raise ValueError(f"环境变量 {env_var} 未设置")
    
    return config

def print_progress_bar(current: int, total: int, prefix: str = "Progress", length: int = 50):
    """打印进度条"""
    filled_length = int(length * current // total)
    bar = '█' * filled_length + '-' * (length - filled_length)
    percent = current / total * 100
    print(f'\r{prefix}: |{bar}| {current}/{total} ({percent:.1f}%)', end='', flush=True)
    if current == total:
        print()  # 换行

def load_completed_files(saves_file: str) -> Set[str]:
    """加载已完成的文件列表"""
    if not os.path.exists(saves_file):
        return set()
    
    try:
        with open(saves_file, 'r', encoding='utf-8') as f:
            completed_files = set(line.strip() for line in f if line.strip())
        return completed_files
    except Exception as e:
        logging.warning(f"读取进度文件失败: {e}")
        return set()

def save_completed_files(saves_file: str, completed_files: Set[str]):
    """保存已完成的文件列表"""
    try:
        with open(saves_file, 'w', encoding='utf-8') as f:
            for file_name in sorted(completed_files):
                f.write(f"{file_name}\n")
    except Exception as e:
        logging.error(f"保存进度文件失败: {e}")

def get_file_batch_key(file_paths: List[Path]) -> str:
    """生成批次标识符"""
    return ",".join(sorted(str(f.name) for f in file_paths))

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Financial Report QA Generator')
    parser.add_argument('-r', '--resume', type=str, help='Resume from saves file')
    args = parser.parse_args()
    
    try:
        # 加载配置
        config = load_config()
        
        # 设置日志
        setup_logging(config['log_file'])
        
        if os.path.exists(config['log_file']): # 清空上次日志文件
            os.remove(config['log_file'])

        logging.info("=== Report QA Pipeline Start ===")
        
        # 检查输入目录
        input_dir = Path(config['input_dir'])
        if not input_dir.exists():
            logging.error(f"输入目录不存在: {input_dir}")
            return
        
        # 获取文件列表
        file_list = list(input_dir.glob('*.txt'))
        if not file_list:
            logging.warning(f"在 {input_dir} 中未找到 .txt 文件")
            return
        
        # Resume功能：加载已完成的文件
        completed_files = set()
        if args.resume:
            completed_files = load_completed_files(args.resume)
            logging.info(f"Resume模式：已加载 {len(completed_files)} 个已完成文件")
        
        # 过滤出未完成的文件
        remaining_files = [f for f in file_list if f.name not in completed_files]
        
        if not remaining_files:
            logging.info("所有文件都已处理完成")
            return
        
        logging.info(f"找到 {len(file_list)} 个文件，剩余 {len(remaining_files)} 个待处理")
        
        # 配置参数
        batch_size = config.get('batch_size', 2)  # 每个进程处理的文本数量
        max_workers = config.get('max_workers', 4)  # 最大进程数
        
        # 计算总批次数
        total_batches = (len(remaining_files) + batch_size - 1) // batch_size
        logging.info(f"总共 {total_batches} 个批次，最多使用 {max_workers} 个worker")
        
        # 准备输出文件
        output_path = Path(config['output_file'])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 准备进度文件
        saves_file = args.resume if args.resume else "output/saves.txt"
        
        # 清空输出文件（仅在非resume模式）
        if not args.resume:
            with open(output_path, 'w', encoding='utf-8') as f:
                pass
        
        # 滚动窗口式处理
        total_qa_count = 0
        completed_batches = 0
        
        print(f"开始处理 {len(remaining_files)} 个文件，共 {total_batches} 个批次...")
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # 提交第一批任务
            futures = {}
            current_batch = 0
            
            # 初始提交任务
            for i in range(0, min(max_workers * batch_size, len(remaining_files)), batch_size):
                batch_files = remaining_files[i:i+batch_size]
                batch_texts = []
                
                for file_path in batch_files:
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            text = f.read().strip()
                        if text:
                            batch_texts.append(text)
                    except Exception as e:
                        logging.error(f"读取文件失败 {file_path}: {e}")
                
                if batch_texts:
                    current_batch += 1
                    future = executor.submit(process_batch, batch_texts, config, current_batch, str(output_path))
                    futures[future] = (current_batch, batch_files)
            
            # 处理完成的任务并提交新任务
            file_index = len(futures) * batch_size
            
            while futures:
                # 等待一个任务完成
                done_future = next(as_completed(futures.keys()))
                batch_id, batch_files = futures.pop(done_future)
                
                try:
                    qa_count = done_future.result()
                    total_qa_count += qa_count
                    completed_batches += 1
                    # logging.info(f"批次 {batch_id} 完成，生成 {qa_count} 个QA")
                    
                    # 保存已完成的文件到进度文件
                    for file_path in batch_files:
                        completed_files.add(file_path.name)
                    save_completed_files(saves_file, completed_files)
                    
                    # 更新进度条
                    print_progress_bar(completed_batches, total_batches, "处理进度")
                    
                except Exception as e:
                    logging.error(f"批次 {batch_id} 处理失败: {e}")
                
                # 提交新任务（如果还有文件）
                if file_index < len(remaining_files):
                    batch_files = remaining_files[file_index:file_index + batch_size]
                    batch_texts = []
                    
                    for file_path in batch_files:
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                text = f.read().strip()
                            if text:
                                batch_texts.append(text)
                        except Exception as e:
                            logging.error(f"读取文件失败 {file_path}: {e}")
                    
                    if batch_texts:
                        current_batch += 1
                        future = executor.submit(process_batch, batch_texts, config, current_batch, str(output_path))
                        futures[future] = (current_batch, batch_files)
                    
                    file_index += batch_size
        
        print()  # 进度条完成后换行
        logging.info(f"Pipeline finished, total items: {total_qa_count}")
        
        # 正常结束时删除进度文件（除非是resume模式）
        if not args.resume and os.path.exists(saves_file):
            try:
                os.remove(saves_file)
                logging.info(f"已删除进度文件: {saves_file}")
            except Exception as e:
                logging.warning(f"删除进度文件失败: {e}")
        else:
            logging.info(f"进度已保存到: {saves_file}")
        
        logging.info("=== End ===")
        
    except Exception as e:
        logging.error(f"程序执行失败: {e}")
        raise

if __name__ == "__main__":
    main()