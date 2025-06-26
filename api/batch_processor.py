"""
批量API处理器 - 支持并行处理和batch API调用
"""

import os
import time
import json
import logging
from typing import List, Dict, Any, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import openai
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from utils.logger import setup_logging


@dataclass
class BatchTask:
    """批量任务数据类"""
    task_id: str
    text: str
    task_type: str  # 'summarize' 或 'qa_generate'
    config: Dict[str, Any]


class BatchProcessor:
    """批量API处理器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.openai_api_key = config['openai_api_key']
        self.max_workers = config.get('max_workers', 4)
        self.batch_size = config.get('batch_size', 10)
        self.max_tokens_per_worker = config.get('max_tokens_per_worker', 50000)
        
        # 设置OpenAI API key
        openai.api_key = self.openai_api_key
        
        # 在子进程中重新配置日志
        try:
            log_file = config.get('log_file', 'output/logs.txt')
            setup_logging(log_file)
        except Exception as e:
            # 如果日志配置失败，继续执行
            pass
        
    def estimate_tokens(self, text: str) -> int:
        """估算文本的token数量（简单估算：1个token约等于4个字符）"""
        return len(text) // 4
    
    def split_text_by_tokens(self, text: str, max_tokens: int) -> List[str]:
        """按token数量分割文本"""
        if self.estimate_tokens(text) <= max_tokens:
            return [text]
        
        # 使用句子分割，然后组合
        from utils.text_processor import split_text_into_sentences
        sentences = split_text_into_sentences(text)
        
        chunks = []
        current_chunk = ""
        current_tokens = 0
        
        for sentence in sentences:
            sentence_tokens = self.estimate_tokens(sentence)
            
            if current_tokens + sentence_tokens <= max_tokens:
                current_chunk += (" " + sentence if current_chunk else sentence)
                current_tokens += sentence_tokens
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
                current_tokens = sentence_tokens
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def create_batch_tasks(self, texts: List[str], task_type: str) -> List[BatchTask]:
        """创建批量任务"""
        tasks = []
        
        for i, text in enumerate(texts):
            # 如果文本太长，分割成多个任务
            text_chunks = self.split_text_by_tokens(text, self.max_tokens_per_worker)
            
            for j, chunk in enumerate(text_chunks):
                task_id = f"{task_type}_{i}_{j}"
                task = BatchTask(
                    task_id=task_id,
                    text=chunk,
                    task_type=task_type,
                    config=self.config
                )
                tasks.append(task)
        
        return tasks
    
    def batch_summarize(self, texts: List[str]) -> List[str]:
        """批量摘要处理"""
        # logging.info(f"开始批量摘要处理，共 {len(texts)} 个文本")
        
        # 创建批量任务
        tasks = self.create_batch_tasks(texts, 'summarize')
        
        # 使用进程池并行处理
        summaries = []
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交任务
            future_to_task = {
                executor.submit(self._process_summarize_task, task): task 
                for task in tasks
            }
            
            # 收集结果
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    result = future.result()
                    summaries.append(result)
                except Exception as e:
                    logging.error(f"摘要任务 {task.task_id} 失败: {e}")
                    # 失败时使用TF-IDF作为fallback
                    fallback = self._fallback_summarize(task.text)
                    summaries.append(fallback)
        
        # 保持每个文本对应一个摘要，不合并
        # logging.info(f"批量摘要完成，共生成 {len(summaries)} 个摘要")
        
        return summaries
    
    def batch_generate_qa(self, summaries: List[str]) -> List[Dict[str, Any]]:
        """批量QA生成处理"""
        # logging.info(f"开始批量QA生成，共 {len(summaries)} 个摘要")
        
        # 创建批量任务
        tasks = self.create_batch_tasks(summaries, 'qa_generate')
        
        # 使用进程池并行处理
        all_qa_results = []
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交任务
            future_to_task = {
                executor.submit(self._process_qa_task, task): task 
                for task in tasks
            }
            
            # 收集结果
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    result = future.result()
                    all_qa_results.extend(result)
                except Exception as e:
                    logging.error(f"QA任务 {task.task_id} 失败: {e}")
        
        # logging.info(f"批量QA生成完成，总共生成 {len(all_qa_results)} 个QA对")
        return all_qa_results
    
    def _process_summarize_task(self, task: BatchTask) -> str:
        """处理单个摘要任务（在子进程中运行）"""
        try:
            # 重新设置API key（子进程需要）
            openai.api_key = task.config['openai_api_key']
            
            summary_model = task.config.get('summary_model', 'gpt-4o-mini')
            summary_tokens = task.config.get('summary_max_tokens', 2048)
            summary_temp = task.config.get('summary_temperature', 0.3)
            
            # 构造摘要prompt
            summary_prompt = f"""Please provide a professional summary of the following financial report text with the following requirements:
1. Preserve all important financial data, numbers, and specific details
2. Retain key business information, risk factors, and opportunities
3. Preserve important timelines and events
4. Use concise but information-rich language
5. Organize information in logical order
6. Do not omit any important financial metrics or business data

Financial Report Text:
{task.text}

Please return the summary content directly without adding any additional formatting or explanations."""

            response = openai.chat.completions.create(
                model=summary_model,
                messages=[
                    {"role": "system", "content": "You are a professional financial analyst specializing in extracting and summarizing key information from financial reports."},
                    {"role": "user", "content": summary_prompt}
                ],
                max_tokens=summary_tokens,
                temperature=summary_temp
            )
            
            summary = response.choices[0].message.content.strip() if response.choices[0].message.content else ""
            return summary
            
        except Exception as e:
            logging.error(f"摘要任务失败: {e}")
            # 返回原始文本作为fallback
            return task.text
    
    def _process_qa_task(self, task: BatchTask) -> List[Dict[str, Any]]:
        """处理单个QA生成任务（在子进程中运行）"""
        try:
            # 重新设置API key（子进程需要）
            openai.api_key = task.config['openai_api_key']
            
            model = task.config.get('model', 'gpt-4o')
            temperature = task.config.get('temperature', 0.7)
            max_tokens = task.config.get('max_tokens', 2048)
            max_q = task.config.get('max_questions', 1)
            
            # 加载prompt模板
            templates_file = task.config.get('prompt_templates_file', 'prompts/templates.txt')
            templates = self._load_prompt_templates(templates_file)
            selected_template = templates[0] if templates else self._get_default_template()
            
            # 格式化模板
            user_prompt = selected_template.format(
                max_q=max_q,
                report_text=task.text
            )
            
            response = openai.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a financial report analysis expert, skilled at outputting information in Alpaca format."},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            raw = response.choices[0].message.content.strip() if response.choices[0].message.content else ""
            
            # 解析JSON结果
            data = self._parse_json_response(raw)
            
            # 统一返回列表
            if isinstance(data, dict):
                return [data]
            return data
            
        except Exception as e:
            logging.error(f"QA任务失败: {e}")
            return []
    
    def _fallback_summarize(self, text: str) -> str:
        """摘要失败时的fallback方法（使用TF-IDF）"""
        try:
            from utils.text_processor import extract_key_sentences_tfidf
            sentences = extract_key_sentences_tfidf(text, 10)
            return " ".join(sentences)
        except Exception as e:
            logging.error(f"Fallback摘要也失败: {e}")
            return text[:1000] + "..." if len(text) > 1000 else text
    
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
            # 提取 JSON 列表 - 处理被代码块包裹的情况
            if raw.startswith('```json'):
                json_start = raw.find('\n', 7) + 1
                json_end = raw.rfind('```')
                if json_end > json_start:
                    payload = raw[json_start:json_end].strip()
                else:
                    payload = raw
            elif raw.startswith('```'):
                json_start = raw.find('\n', 3) + 1
                json_end = raw.rfind('```')
                if json_end > json_start:
                    payload = raw[json_start:json_end].strip()
                else:
                    payload = raw
            else:
                # 直接查找JSON数组或对象
                start = raw.find('[')
                end = raw.rfind(']') + 1
                if start != -1 and end != -1:
                    payload = raw[start:end]
                else:
                    start = raw.find('{')
                    end = raw.rfind('}') + 1
                    if start != -1 and end != -1:
                        payload = raw[start:end]
                    else:
                        payload = raw
            
            return json.loads(payload)
            
        except Exception as e:
            logging.error(f"JSON解析失败: {e}")
            return []


def write_jsonl_threadsafe(results: List[Dict[str, Any]], output_path: str):
    """多线程安全写入jsonl文件"""
    import threading
    
    lock = threading.Lock()
    
    def write_one(item):
        with lock:
            with open(output_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    with ThreadPoolExecutor() as executor:
        list(executor.map(write_one, results))


def write_qa_result_threadsafe(qa_result: Dict[str, Any], output_path: str):
    """线程安全写入单个QA结果（用于流式写入）"""
    import threading
    
    lock = threading.Lock()
    
    with lock:
        with open(output_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(qa_result, ensure_ascii=False) + '\n')


def process_batch(texts: List[str], config: Dict[str, Any], batch_id: int, output_path: str) -> int:
    """处理一批文本：摘要 + QA生成（用于多进程，流式写入）"""
    import logging
    from utils.logger import setup_logging
    
    # 在子进程中重新配置日志
    log_file = config.get('log_file', 'output/logs.txt')
    setup_logging(log_file)
    
    logging.info(f"Worker {batch_id}: 开始处理 {len(texts)} 个文本")
    
    processor = BatchProcessor(config)
    total_qa_count = 0
    
    # 第一步：批量摘要
    summaries = processor.batch_summarize(texts)
    
    # 第二步：分别为每个摘要生成QA并立即写入
    for i, summary in enumerate(summaries):
        qa_results = processor.batch_generate_qa([summary])
        
        # 立即写入每个QA结果
        for qa_result in qa_results:
            write_qa_result_threadsafe(qa_result, output_path)
            total_qa_count += 1
    
    logging.info(f"Worker {batch_id}: 完成，生成 {total_qa_count} 个QA")
    return total_qa_count


def process_files_batch(file_paths: List[str], config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """批量处理文件的主函数"""
    processor = BatchProcessor(config)
    
    # 读取所有文件内容
    texts = []
    for file_path in file_paths:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read().strip()
            if text:
                texts.append(text)
        except Exception as e:
            logging.error(f"读取文件失败 {file_path}: {e}")
    
    if not texts:
        logging.warning("没有有效的文本内容")
        return []
    
    # 第一步：批量摘要
    summaries = processor.batch_summarize(texts)
    
    # 第二步：分别为每个摘要生成QA
    all_qa_results = []
    for summary in summaries:
        qa_results = processor.batch_generate_qa([summary])
        all_qa_results.extend(qa_results)
    
    return all_qa_results 