"""
单条文本处理器 - 处理单个文件的摘要和QA生成
"""

import json
import logging
import fcntl
import random
from typing import Dict, Any, List
from utils.llm_api import OpenAILLMApi, DeepSeekLLMApi


class SingleProcessor:
    """单条文本处理器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    def _get_llm_api(self, config):
        llm_provider = config.get('llm_provider', 'openai')
        if llm_provider == 'deepseek':
            return DeepSeekLLMApi(
                api_key=config['deepseek_api_key'],
                api_base=config.get('deepseek_api_base', 'https://api.deepseek.com')
            )
        else:
            return OpenAILLMApi(api_key=config['openai_api_key'])
    
    def generate_summary(self, text: str) -> str:
        """生成摘要"""
        try:
            llm_api = self._get_llm_api(self.config)
            summary_model = self.config.get('summary_model', 'gpt-4o-mini')
            summary_tokens = self.config.get('summary_max_tokens', 2048)
            summary_temp = self.config.get('summary_temperature', 0.3)
            
            summary_prompt = f"""Please provide a professional summary of the following financial report text with the following requirements:
1. Preserve all important financial data, numbers, and specific details
2. Retain key business information, risk factors, and opportunities
3. Preserve important timelines and events
4. Use concise but information-rich language
5. Organize information in logical order
6. Do not omit any important financial metrics or business data

Financial Report Text:
{text}

Please return the summary content directly without adding any additional formatting or explanations."""
            
            messages = [
                {"role": "system", "content": "You are a professional financial analyst specializing in extracting and summarizing key information from financial reports."},
                {"role": "user", "content": summary_prompt}
            ]
            
            summary = llm_api.chat_completion(
                messages=messages,
                model=summary_model,
                max_tokens=summary_tokens,
                temperature=summary_temp
            )
            
            return summary
            
        except Exception as e:
            logging.error(f"摘要生成失败: {e}")
            # 抛出特定异常，让main.py处理API失败计数
            raise RuntimeError(f"API调用失败: {e}")
    
    def generate_qa(self, summary: str) -> List[Dict[str, Any]]:
        """生成QA对"""
        try:
            llm_api = self._get_llm_api(self.config)
            model = self.config.get('model', 'gpt-4o')
            temperature = self.config.get('temperature', 0.7)
            max_tokens = self.config.get('max_tokens', 2048)
            max_q = self.config.get('max_questions', 1)
            
            templates_file = self.config.get('prompt_templates_file', 'prompts/templates.txt')
            templates = self._load_prompt_templates(templates_file)
            selected_template = random.choice(templates) if templates else self._get_default_template()
            
            user_prompt = selected_template.format(
                max_q=max_q,
                report_text=summary
            )
            
            messages = [
                {"role": "system", "content": "You are a financial report analysis expert, skilled at outputting information in Alpaca format."},
                {"role": "user", "content": user_prompt}
            ]
            
            raw = llm_api.chat_completion(
                messages=messages,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            data = self._parse_json_response(raw)
            if isinstance(data, dict):
                return [data]
            return data
            
        except Exception as e:
            logging.error(f"QA生成失败: {e}")
            # 抛出特定异常，让main.py处理API失败计数
            raise RuntimeError(f"API调用失败: {e}")
    
    def _load_prompt_templates(self, templates_file: str) -> List[str]:
        """加载prompt模板"""
        try:
            with open(templates_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            templates = [t.strip() for t in content.split('---') if t.strip()]
            return templates
        except Exception as e:
            logging.error(f"Failed to load prompt templates: {e}")
            return []
    
    def _get_default_template(self) -> str:
        """获取默认模板"""
        return """Please read the following complete financial report text, then generate up to {max_q} high-quality questions focusing on content depth and business insights. Return as a JSON list, each element containing:
- "instruction": the question
- "input": brief context in English
- "output": detailed answer to the question

Full Report:
{report_text}"""
    
    def _parse_json_response(self, raw: str) -> Any:
        """解析JSON响应"""
        try:
            # 简化JSON解析逻辑
            if raw.startswith('```'):
                # 提取代码块内容
                lines = raw.split('\n')
                start_idx = 1 if lines[0].startswith('```') else 0
                end_idx = len(lines) - 1 if lines[-1].startswith('```') else len(lines)
                payload = '\n'.join(lines[start_idx:end_idx]).strip()
            else:
                # 直接查找JSON数组或对象
                start = raw.find('[')
                if start != -1:
                    end = raw.rfind(']') + 1
                    payload = raw[start:end]
                else:
                    start = raw.find('{')
                    end = raw.rfind('}') + 1
                    payload = raw[start:end] if start != -1 else raw
            
            return json.loads(payload)
            
        except Exception as e:
            logging.error(f"JSON解析失败: {e}")
            return []


def write_jsonl(qa_result: Dict[str, Any], output_path: str):
    """写入单个QA结果到jsonl文件（带文件锁，线程安全）"""
    with open(output_path, 'a', encoding='utf-8') as f:
        # 获取文件锁
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            f.write(json.dumps(qa_result, ensure_ascii=False) + '\n')
            f.flush()  # 确保立即写入磁盘
        finally:
            # 释放文件锁
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def process_single_file(text: str, file_name: str, config: Dict[str, Any], output_path: str, progress_path: str) -> int:
    """处理单个文件：摘要 + QA生成"""
    logging.info(f"处理文件: {file_name}")
    
    processor = SingleProcessor(config)
    qa_count = 0
    
    # 第一步：生成摘要
    summary = processor.generate_summary(text)
    
    # 第二步：生成QA
    qa_results = processor.generate_qa(summary)
    
    # 写入QA结果
    for qa_result in qa_results:
        write_jsonl(qa_result, output_path)
        qa_count += 1
    
    # 写入进度文件
    with open(progress_path, 'a', encoding='utf-8') as f:
        # 获取文件锁
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            f.write(file_name + '\n')
            f.flush()  # 确保立即写入磁盘
        finally:
            # 释放文件锁
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    
    logging.info(f"文件 {file_name} 处理完成，生成 {qa_count} 个QA")
    return qa_count 