"""
统一的文本处理模块
包含TF-IDF提取和GPT摘要功能
"""

import time
import openai
import json
import logging
import random
import re
import numpy as np
from typing import List, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer


def extract_key_sentences(text: str, config: Dict[str, Any]) -> List[str]:
    """
    主函数：根据配置选择使用不同的摘要方式
    支持三种方式：
    - "tfidf": 仅使用TF-IDF方法
    - "gpt": 仅使用GPT摘要方法
    - "tfidf_gpt": 先使用TF-IDF提取关键句子，再用GPT进行摘要
    """
    # 获取摘要方式，优先使用新的summary_method配置
    summary_method = config.get('summary_method', 'gpt')
    
    # 保持向后兼容性
    if summary_method == 'gpt' and not config.get('use_gpt_summary', True):
        summary_method = 'tfidf'
    
    logging.info(f"使用摘要方式: {summary_method}")
    
    if summary_method == 'tfidf':
        return extract_key_sentences_tfidf(text, config.get('top_k_sentences', 50))
    
    elif summary_method == 'gpt':
        try:
            return extract_key_sentences_with_gpt(text, config)
        except Exception as e:
            logging.warning(f"GPT摘要失败，回退到TF-IDF方法: {e}")
            return extract_key_sentences_tfidf(text, config.get('top_k_sentences', 50))
    
    elif summary_method == 'tfidf_gpt':
        return extract_key_sentences_tfidf_gpt(text, config)
    
    else:
        logging.warning(f"未知的摘要方式: {summary_method}，使用默认的GPT方式")
        try:
            return extract_key_sentences_with_gpt(text, config)
        except Exception as e:
            logging.warning(f"GPT摘要失败，回退到TF-IDF方法: {e}")
            return extract_key_sentences_tfidf(text, config.get('top_k_sentences', 50))


def extract_key_sentences_with_gpt(text: str, config: Dict[str, Any]) -> List[str]:
    """
    使用便宜的GPT模型进行财报摘要，保留关键数据和细节信息
    """
    # 使用便宜的模型进行摘要
    summary_model = config.get('summary_model', 'gpt-4o-mini')
    summary_tokens = config.get('summary_max_tokens', 2048)
    summary_temp = config.get('summary_temperature', 0.3)
    
    # 如果文本太长，先分段处理
    max_chunk_size = config.get('max_chunk_size', 8000)
    chunks = split_text_into_chunks_smart(text, max_chunk_size)
    
    all_summaries = []
    
    for i, chunk in enumerate(chunks):
        try:
            summary = summarize_chunk_with_gpt(
                chunk, 
                summary_model, 
                summary_tokens, 
                summary_temp,
                config
            )
            all_summaries.append(summary)
            logging.info(f"完成第 {i+1}/{len(chunks)} 个文本块的摘要")
            
        except Exception as e:
            logging.warning(f"第 {i+1} 个文本块摘要失败: {e}")
            # 如果GPT摘要失败，回退到TF-IDF方法
            fallback_sentences = extract_key_sentences_tfidf(chunk, 10)
            all_summaries.append(" ".join(fallback_sentences))
    
    # 合并所有摘要
    combined_summary = " ".join(all_summaries)
    
    # 如果合并后的摘要仍然太长，再次压缩
    if len(combined_summary) > max_chunk_size:
        combined_summary = summarize_chunk_with_gpt(
            combined_summary,
            summary_model,
            summary_tokens // 2,  # 使用更少的token进行最终压缩
            summary_temp,
            config
        )
    
    # 统一用 split_text_into_sentences 分句
    sentences = split_text_into_sentences(combined_summary)
    
    # 限制句子数量
    max_sentences = config.get('top_k_sentences', 50)
    return sentences[:max_sentences]


def summarize_chunk_with_gpt(chunk: str, model: str, max_tokens: int, temperature: float, config: Dict[str, Any]) -> str:
    """
    使用GPT模型对文本块进行摘要
    """
    openai.api_key = config['openai_api_key']
    
    # 构造摘要prompt
    summary_prompt = f"""Please provide a professional summary of the following financial report text with the following requirements:
1. Preserve all important financial data, numbers, and specific details
2. Retain key business information, risk factors, and opportunities
3. Preserve important timelines and events
4. Use concise but information-rich language
5. Organize information in logical order
6. Do not omit any important financial metrics or business data

Financial Report Text:
{chunk}

Please return the summary content directly without adding any additional formatting or explanations."""

    try:
        response = openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a professional financial analyst specializing in extracting and summarizing key information from financial reports."},
                {"role": "user", "content": summary_prompt}
            ],
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        summary = response.choices[0].message.content.strip() if response.choices[0].message.content else ""
        return summary
        
    except Exception as e:
        logging.error(f"GPT摘要失败: {e}")
        raise


def split_text_into_chunks_smart(text: str, max_size: int) -> List[str]:
    """
    智能文本分块：先分句，再分块，避免切割数字和重要信息
    """
    if len(text) <= max_size:
        return [text]
    
    # 第一步：先分割成句子
    sentences = split_text_into_sentences(text)
    
    # 第二步：将句子组合成块
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        # 如果当前块加上这个句子不超过限制
        if len(current_chunk) + len(sentence) + 1 <= max_size:
            current_chunk += (" " + sentence if current_chunk else sentence)
        else:
            # 当前块已满，保存并开始新块
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence
    
    # 添加最后一个块
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    # 确保没有空块
    chunks = [chunk for chunk in chunks if chunk.strip()]
    
    return chunks


def split_text_into_sentences(text: str) -> List[str]:
    """
    智能句子分割，避免切割数字和重要信息
    """
    # 预处理：标准化换行符和空格
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r' +', ' ', text)
    
    # 定义需要保护的模式
    protected_patterns = [
        # 数字模式
        r'\$\d+(?:,\d{3})*(?:\.\d+)?',      # $1,234.56, $1.23
        r'\d+(?:,\d{3})*(?:\.\d+)?%',       # 12.34%, 1.000%, 5%
        r'\d+(?:,\d{3})*(?:\.\d+)?',        # 1,234.56, 1.23, 123
        r'\d+\.\d+',                        # 12.34, 1.000
        r'\d+',                             # 123, 1
        
        # 日期模式
        r'\d{1,2}/\d{1,2}/\d{4}',          # MM/DD/YYYY
        r'\d{4}-\d{1,2}-\d{1,2}',          # YYYY-MM-DD
        r'\w+ \d{1,2}, \d{4}',             # Month DD, YYYY
        
        # 财务术语模式
        r'[A-Z]{2,5}\s*[-=]\s*\d+',        # EPS = 1.23
        r'ROE\s*[-=]\s*\d+',               # ROE = 15.2%
        r'P/E\s*[-=]\s*\d+',               # P/E = 25.4
        
        # 公司名称模式
        r'[A-Z][a-z]+ Inc\.',              # Apple Inc.
        r'[A-Z][a-z]+ Corp\.',             # Microsoft Corp.
        r'[A-Z][a-z]+ Ltd\.',              # Company Ltd.
    ]
    
    # 合并所有保护模式
    protected_regex = '|'.join(protected_patterns)
    
    # 找到所有需要保护的内容
    protected_ranges: List[tuple[int, int]] = []
    for match in re.finditer(protected_regex, text, re.IGNORECASE):
        protected_ranges.append((match.start(), match.end()))
    
    # 使用保守的句子分割策略
    sentences = []
    current_pos = 0
    
    # 找到所有可能的句子边界
    sentence_boundaries = []
    
    # 使用正则表达式找到句子边界，但避免在受保护内容附近分割
    for match in re.finditer(r'[.!?]+(?=\s+[A-Z][a-z])', text):
        boundary_pos = match.end()
        
        # 检查这个边界是否在受保护内容附近
        is_safe_boundary = True
        for start, end in protected_ranges:
            # 如果边界在受保护内容的5个字符范围内，认为不安全
            if abs(boundary_pos - start) < 5 or abs(boundary_pos - end) < 5:
                is_safe_boundary = False
                break
        
        if is_safe_boundary:
            sentence_boundaries.append(boundary_pos)
    
    # 添加文本结尾
    sentence_boundaries.append(len(text))
    
    # 根据边界分割句子
    for boundary in sentence_boundaries:
        sentence = text[current_pos:boundary].strip()
        if sentence:
            sentences.append(sentence)
        current_pos = boundary
    
    # 如果没有找到句子边界，使用简单的分割
    if not sentences:
        # 使用更保守的分割方法
        parts = re.split(r'[.!?]+\s+', text)
        sentences = [part.strip() for part in parts if part.strip()]
    
    # 过滤空句子并清理
    sentences = [s.strip() for s in sentences if s.strip()]
    
    return sentences


def split_text_into_chunks(text: str, max_size: int) -> List[str]:
    """
    将长文本分割成适合处理的块（保留原函数以兼容性）
    """
    return split_text_into_chunks_smart(text, max_size)


def extract_key_sentences_tfidf(text: str, top_k: int) -> List[str]:
    """
    使用TF-IDF方法提取关键句子 - 统一用智能句子分割
    """
    sentences = split_text_into_sentences(text)
    if not sentences or top_k <= 0:
        return []

    try:
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(sentences)
        
        # 计算每个句子的TF-IDF分数总和
        scores = np.array(tfidf_matrix.sum(axis=1)).flatten()
        
        # 获取分数最高的句子索引
        top_indices = scores.argsort()[::-1][:top_k]
        top_indices_sorted = sorted(top_indices)
        
        return [sentences[i] for i in top_indices_sorted]
        
    except Exception as e:
        logging.error(f"TF-IDF处理失败: {e}")
        # 如果TF-IDF失败，返回前top_k个句子
        return sentences[:top_k]


def extract_key_sentences_tfidf_gpt(text: str, config: Dict[str, Any]) -> List[str]:
    """
    TF-IDF+GPT混合方法：先用TF-IDF提取关键句子，再用GPT进行摘要
    """
    logging.info("使用TF-IDF+GPT混合方法")
    
    # 第一步：使用TF-IDF提取关键句子
    tfidf_sentences = extract_key_sentences_tfidf(text, config.get('top_k_sentences', 50))
    
    if not tfidf_sentences:
        logging.warning("TF-IDF未提取到句子，直接返回空列表")
        return []
    
    # 第二步：将TF-IDF提取的句子组合成文本
    tfidf_text = " ".join(tfidf_sentences)
    
    # 第三步：使用GPT对TF-IDF结果进行进一步摘要
    try:
        summary_model = config.get('summary_model', 'gpt-4o-mini')
        summary_tokens = config.get('summary_max_tokens', 2048)
        summary_temp = config.get('summary_temperature', 0.3)
        
        # 使用GPT对TF-IDF结果进行摘要
        gpt_summary = summarize_chunk_with_gpt(
            tfidf_text,
            summary_model,
            summary_tokens,
            summary_temp,
            config
        )
        
        # 将GPT摘要结果分割成句子
        final_sentences = split_text_into_sentences(gpt_summary)
        
        # 限制句子数量
        max_sentences = config.get('top_k_sentences', 50)
        return final_sentences[:max_sentences]
        
    except Exception as e:
        logging.warning(f"TF-IDF+GPT混合方法中GPT摘要失败: {e}")
        # 如果GPT摘要失败，返回TF-IDF的结果
        return tfidf_sentences 