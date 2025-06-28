#!/usr/bin/env python3
"""
Alpaca格式过滤器 - 过滤和验证JSONL文件中的Alpaca格式数据
"""

import json
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple


def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def extract_alpaca_fields(data: Any) -> Optional[Dict[str, str]]:
    """
    从数据中递归提取Alpaca格式的字段
    
    Args:
        data: 可能是字典、列表或其他格式的数据
        
    Returns:
        包含instruction, input, output的字典，如果提取失败返回None
    """
    try:
        # 如果data是列表，取第一个元素
        if isinstance(data, list) and len(data) > 0:
            data = data[0]
        
        # 确保data是字典
        if not isinstance(data, dict):
            return None
        
        # 递归搜索instruction, input, output字段
        instruction = None
        input_data = None
        output_data = None
        
        def search_fields(obj, path=""):
            nonlocal instruction, input_data, output_data
            
            if isinstance(obj, dict):
                for key, value in obj.items():
                    current_path = f"{path}.{key}" if path else key
                    
                    # 检查是否是目标字段
                    if key == 'instruction' and instruction is None:
                        instruction = value
                    elif key == 'input' and input_data is None:
                        input_data = value
                    elif key == 'output' and output_data is None:
                        output_data = value
                    
                    # 递归搜索嵌套对象
                    if isinstance(value, (dict, list)):
                        search_fields(value, current_path)
                        
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    current_path = f"{path}[{i}]"
                    if isinstance(item, (dict, list)):
                        search_fields(item, current_path)
        
        # 开始搜索
        search_fields(data)
        
        # 检查是否找到了所有必需字段
        if instruction is None or output_data is None:
            return None
        
        # 处理instruction字段
        if isinstance(instruction, str):
            instruction = instruction.strip()
        else:
            instruction = str(instruction).strip()
        
        # 处理input字段 - 如果是对象或列表，转换为字符串
        if input_data is None:
            input_text = ""
        elif isinstance(input_data, (dict, list)):
            input_text = json.dumps(input_data, ensure_ascii=False)
        else:
            input_text = str(input_data).strip()
        
        # 处理output字段 - 如果是对象或列表，转换为字符串
        if isinstance(output_data, (dict, list)):
            output = json.dumps(output_data, ensure_ascii=False)
        else:
            output = str(output_data).strip()
        
        # 检查必需字段是否存在且不为空
        if not instruction or not output:
            return None
        
        # 检查output长度不能小于instruction
        if len(output) < len(instruction):
            return None
        
        return {
            'instruction': instruction,
            'input': input_text,
            'output': output
        }
        
    except Exception as e:
        logging.debug(f"提取字段失败: {e}")
        return None


def validate_alpaca_format(alpaca_data: Dict[str, str]) -> bool:
    """
    验证Alpaca格式数据是否有效
    
    Args:
        alpaca_data: 包含instruction, input, output的字典
        
    Returns:
        是否有效
    """
    try:
        instruction = alpaca_data.get('instruction', '').strip()
        output = alpaca_data.get('output', '').strip()
        
        # 检查instruction和output不能为空
        if not instruction or not output:
            return False
        
        # 检查output长度不能小于instruction
        if len(output) < len(instruction):
            return False
        
        return True
        
    except Exception:
        return False


def filter_jsonl_file(input_path: str) -> Tuple[int, int]:
    """
    过滤JSONL文件
    
    Args:
        input_path: 输入文件路径
        
    Returns:
        (保留的行数, 删除的行数)
    """
    input_file = Path(input_path)
    output_file = input_file.parent / f"{input_file.stem}_filtered{input_file.suffix}"
    deleted_file = input_file.parent / f"{input_file.stem}_deleted{input_file.suffix}"
    
    kept_count = 0
    deleted_count = 0
    
    logging.info(f"开始过滤文件: {input_path}")
    logging.info(f"输出文件: {output_file}")
    logging.info(f"删除文件: {deleted_file}")
    
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile, \
         open(deleted_file, 'w', encoding='utf-8') as deleted_outfile:
        
        for line_num, line in enumerate(infile, 1):
            original_line = line.strip()
            if not original_line:
                deleted_count += 1
                deleted_outfile.write(f"# 第{line_num}行: 空行\n")
                logging.debug(f"第{line_num}行: 空行，删除")
                continue
            
            try:
                # 解析JSON
                data = json.loads(original_line)
                
                # 提取Alpaca字段
                alpaca_data = extract_alpaca_fields(data)
                
                if alpaca_data is None:
                    deleted_count += 1
                    deleted_outfile.write(f"# 第{line_num}行: 无法提取Alpaca字段\n")
                    deleted_outfile.write(original_line + '\n')
                    logging.debug(f"第{line_num}行: 无法提取Alpaca字段，删除")
                    continue
                
                # 验证格式
                if not validate_alpaca_format(alpaca_data):
                    deleted_count += 1
                    deleted_outfile.write(f"# 第{line_num}行: 格式验证失败\n")
                    deleted_outfile.write(original_line + '\n')
                    logging.debug(f"第{line_num}行: 格式验证失败，删除")
                    continue
                
                # 写入有效数据
                outfile.write(json.dumps(alpaca_data, ensure_ascii=False) + '\n')
                kept_count += 1
                
            except json.JSONDecodeError as e:
                deleted_count += 1
                deleted_outfile.write(f"# 第{line_num}行: JSON解析失败 ({e})\n")
                deleted_outfile.write(original_line + '\n')
                logging.debug(f"第{line_num}行: JSON解析失败 ({e})，删除")
                continue
            except Exception as e:
                deleted_count += 1
                deleted_outfile.write(f"# 第{line_num}行: 处理失败 ({e})\n")
                deleted_outfile.write(original_line + '\n')
                logging.debug(f"第{line_num}行: 处理失败 ({e})，删除")
                continue
    
    return kept_count, deleted_count


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Alpaca格式过滤器')
    parser.add_argument('-i', '--input', required=True, help='输入JSONL文件路径')
    parser.add_argument('-v', '--verbose', action='store_true', help='显示详细日志')
    args = parser.parse_args()
    
    # 设置日志级别
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        setup_logging()
    
    # 检查输入文件是否存在
    if not Path(args.input).exists():
        logging.error(f"输入文件不存在: {args.input}")
        return 1
    
    try:
        # 过滤文件
        kept_count, deleted_count = filter_jsonl_file(args.input)
        
        # 打印结果
        print(f"\n过滤完成!")
        print(f"保留行数: {kept_count}")
        print(f"删除行数: {deleted_count}")
        print(f"总计行数: {kept_count + deleted_count}")
        
        if kept_count + deleted_count > 0:
            retention_rate = (kept_count / (kept_count + deleted_count)) * 100
            print(f"保留率: {retention_rate:.4f}%")
        
        return 0
        
    except Exception as e:
        logging.error(f"过滤过程中发生错误: {e}")
        return 1


if __name__ == "__main__":
    exit(main()) 