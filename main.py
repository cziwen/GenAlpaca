import os
import yaml
import logging
import argparse
import sys
import threading
from typing import Dict, Any, Set
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from datetime import datetime

from api.processor import process_single_file
from utils.logger import setup_logging

# 全局API失败计数器
_api_failure_count = 0
_api_failure_lock = threading.Lock()
_api_failure_threshold = 5  # 默认连续失败5次后终止


def get_api_failure_count() -> int:
    """获取API失败次数"""
    global _api_failure_count
    with _api_failure_lock:
        return _api_failure_count


def increment_api_failure_count():
    """增加API失败次数"""
    global _api_failure_count
    with _api_failure_lock:
        _api_failure_count += 1
        logging.warning(f"API调用失败，当前连续失败次数: {_api_failure_count}")


def reset_api_failure_count():
    """重置API失败次数"""
    global _api_failure_count
    with _api_failure_lock:
        _api_failure_count = 0


def set_api_failure_threshold(threshold: int):
    """设置API失败阈值"""
    global _api_failure_threshold
    _api_failure_threshold = threshold


def should_terminate() -> bool:
    """检查是否应该终止程序"""
    return get_api_failure_count() >= _api_failure_threshold


def load_config(path: str = "config/config.yaml") -> Dict[str, Any]:
    """加载配置文件，支持环境变量替换"""
    with open(path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 替换环境变量
    for key in ['openai_api_key', 'deepseek_api_key']:
        if isinstance(config.get(key), str) and config[key].startswith('${'):
            env_var = config[key][2:-1]  # 移除 ${}
            config[key] = os.getenv(env_var)
    
    if not (config.get('openai_api_key') or config.get('deepseek_api_key')):
        raise ValueError("openai_api_key 和 deepseek_api_key 至少需要设置一个")
    
    return config

def create_backup_dir() -> str:
    """创建带时间戳的备份目录"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = f"output/{timestamp}"
    os.makedirs(backup_dir, exist_ok=True)
    return backup_dir

def load_completed_files(progress_file: str) -> Set[str]:
    """加载已完成的文件列表"""
    if not os.path.exists(progress_file):
        return set()
    
    try:
        with open(progress_file, 'r', encoding='utf-8') as f:
            completed_files = set(line.strip() for line in f if line.strip())
        return completed_files
    except Exception as e:
        logging.warning(f"读取进度文件失败: {e}")
        return set()

def get_backup_dir_from_progress(progress_file: str) -> str:
    """从进度文件路径推断出备份目录"""
    return os.path.dirname(progress_file)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Financial Report QA Generator')
    parser.add_argument('-r', '--resume', type=str, help='Resume from progress file')
    args = parser.parse_args()
    
    try:
        # 创建或获取备份目录
        if args.resume:
            backup_dir = get_backup_dir_from_progress(args.resume)
            print(f"Resume模式：使用原备份目录: {backup_dir}")
        else:
            backup_dir = create_backup_dir()
            print(f"备份目录: {backup_dir}")
        
        # 加载配置
        config = load_config()
        
        # 设置API失败阈值
        api_failure_threshold = config.get('api_failure_threshold', 5)
        set_api_failure_threshold(api_failure_threshold)
        
        # 更新配置文件中的路径到备份目录
        config['output_file'] = os.path.join(backup_dir, 'result.jsonl')
        config['log_file'] = os.path.join(backup_dir, 'logs.txt')
        
        # 设置日志
        setup_logging(config['log_file'])
        
        logging.info("=== Report QA Pipeline Start ===")
        logging.info(f"备份目录: {backup_dir}")
        
        # 显示API失败阈值配置
        logging.info(f"API失败阈值设置为: {api_failure_threshold} 次连续失败后自动终止")
        
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
        max_workers = config.get('max_workers', 20)
        zero_qa_threshold = config.get('api_failure_threshold', 5)
        zero_qa_count = 0
        
        logging.info(f"使用 {max_workers} 个线程并发处理")
        
        # 准备输出文件
        output_path = Path(config['output_file'])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 准备进度文件
        progress_file = args.resume if args.resume else os.path.join(backup_dir, 'progress.txt')
        
        # 清空输出文件（仅在非resume模式）
        if not args.resume:
            with open(output_path, 'w', encoding='utf-8') as f:
                pass
        
        # 使用线程池处理每个文件
        total_qa_count = 0
        
        print(f"开始处理 {len(remaining_files)} 个文件...")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            
            # 提交所有文件处理任务
            for file_path in remaining_files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        text = f.read().strip()
                    if text:
                        future = executor.submit(
                            process_single_file,
                            text,
                            file_path.name,
                            config,
                            str(output_path),
                            progress_file
                        )
                        futures[future] = file_path.name
                except Exception as e:
                    logging.error(f"读取文件失败 {file_path}: {e}")
            
            # 用tqdm进度条包裹文件处理
            with tqdm(total=len(futures), desc="处理进度", ncols=80) as pbar:
                for future in as_completed(futures):
                    file_name = futures[future]
                    try:
                        # 检查API失败状态
                        if should_terminate():
                            failure_count = get_api_failure_count()
                            logging.critical(f"检测到API连续失败{failure_count}次，达到阈值{api_failure_threshold}，程序终止！")
                            print(f"\n❌ API连续失败{failure_count}次，程序自动终止！")
                            sys.exit(1)
                        
                        qa_count = future.result()
                        # 成功处理，重置API失败计数
                        reset_api_failure_count()
                        total_qa_count += qa_count
                        pbar.update(1)
                        
                        # QA写入检查
                        if qa_count == 0:
                            zero_qa_count += 1
                            logging.warning(f"文件 {file_name} 未生成QA，连续未生成次数: {zero_qa_count}")
                            if zero_qa_count >= zero_qa_threshold:
                                logging.critical(f"连续{zero_qa_threshold}个文件未生成QA，程序主动中断！")
                                print(f"\n❌ 连续{zero_qa_threshold}个文件未生成QA，程序终止！")
                                raise RuntimeError(f"连续{zero_qa_threshold}个文件未生成QA，程序主动中断！")
                        else:
                            zero_qa_count = 0
                            
                    except RuntimeError as e:
                        if "API调用失败" in str(e):
                            # API失败，增加失败计数
                            increment_api_failure_count()
                            logging.error(f"文件 {file_name} API调用失败: {e}")
                            
                            if should_terminate():
                                failure_count = get_api_failure_count()
                                logging.critical(f"API连续失败{failure_count}次，达到阈值{api_failure_threshold}，程序终止！")
                                print(f"\n❌ API连续失败{failure_count}次，程序自动终止！")
                                sys.exit(1)
                        else:
                            logging.error(f"文件 {file_name} 处理失败: {e}")
                    except Exception as e:
                        logging.error(f"文件 {file_name} 处理失败: {e}")
        
        logging.info(f"Pipeline finished, total items: {total_qa_count}")
        if not args.resume and os.path.exists(progress_file):
            try:
                os.remove(progress_file)
                logging.info(f"已删除进度文件: {progress_file}")
            except Exception as e:
                logging.warning(f"删除进度文件失败: {e}")
        else:
            logging.info(f"进度已保存到: {progress_file}")
        logging.info("=== End ===")
        
    except KeyboardInterrupt:
        logging.info("用户中断程序")
        print("\n⚠️ 程序被用户中断")
        sys.exit(0)
    except Exception as e:
        logging.error(f"程序执行失败: {e}")
        print(f"\n❌ 程序执行失败: {e}")
        raise

if __name__ == "__main__":
    main()