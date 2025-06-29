"""
文本处理模块 - 简化版
只保留必要的TF-IDF提取功能
"""

import re
import numpy as np
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer


def extract_key_sentences_tfidf(text: str, top_k: int = 10) -> List[str]:
    """
    使用TF-IDF方法提取关键句子
    作为摘要失败时的fallback方法
    """
    # 分割句子
    sentences = split_text_into_sentences(text)
    
    if len(sentences) <= top_k:
        return sentences
    
    # 创建TF-IDF向量化器
    vectorizer = TfidfVectorizer(
        stop_words='english',
        ngram_range=(1, 2),
        max_features=1000
    )
    
    try:
        # 计算TF-IDF矩阵
        tfidf_matrix = vectorizer.fit_transform(sentences)
        
        # 计算每个句子的TF-IDF总分
        scores = []
        for i in range(tfidf_matrix.shape[0]):
            score = tfidf_matrix[i].sum()
            scores.append(score)
        
        # 获取得分最高的句子索引
        scores_array = np.array(scores)
        top_indices = np.argsort(scores_array)[-top_k:][::-1]
        
        # 返回关键句子（按原文顺序）
        key_sentences = [sentences[i] for i in sorted(top_indices)]
        return key_sentences
        
    except Exception as e:
        # 如果TF-IDF失败，返回前top_k个句子
        return sentences[:top_k]


def split_text_into_sentences(text: str) -> List[str]:
    """
    智能句子分割，避免切割数字和重要信息
    """
    # 预处理：标准化换行符和空格
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r' +', ' ', text)
    
    # 使用更简单但有效的方法来保护数字中的小数点
    # 临时替换数字中的小数点，避免被误切
    text = re.sub(r'(\d+)\.(\d+)', r'\1__DECIMAL__\2', text)
    
    # 句子分割模式 - 更精确的匹配
    sentence_patterns = [
        r'[.!?]+(?=\s+[A-Z])',  # 句号、感叹号、问号后跟空格和大写字母
        r'[.!?]+\s*$',          # 句号、感叹号、问号在行尾
    ]
    
    # 分割句子
    sentences = []
    current_pos = 0
    
    for pattern in sentence_patterns:
        matches = list(re.finditer(pattern, text))
        for match in matches:
            sentence = text[current_pos:match.end()].strip()
            if sentence:
                sentences.append(sentence)
            current_pos = match.end()
    
    # 添加最后一部分
    if current_pos < len(text):
        last_sentence = text[current_pos:].strip()
        if last_sentence:
            sentences.append(last_sentence)
    
    # 恢复小数点
    restored_sentences = []
    for sentence in sentences:
        sentence = sentence.replace('__DECIMAL__', '.')
        restored_sentences.append(sentence)
    
    # 清理和过滤
    cleaned_sentences = []
    for sentence in restored_sentences:
        sentence = sentence.strip()
        if sentence and len(sentence) > 10:  # 过滤太短的句子
            cleaned_sentences.append(sentence)
    
    return cleaned_sentences


def estimate_tokens(text: str) -> int:
    """
    估算文本的token数量
    使用简单的估算方法：英文约4个字符1个token，中文约2个字符1个token
    """
    if not text:
        return 0
    
    # 分别计算中文字符和英文字符
    chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
    english_chars = len(text) - chinese_chars
    
    # 估算token数量 - 更保守的估算
    estimated_tokens = (english_chars // 3) + (chinese_chars // 1)
    
    # 确保至少返回1个token
    return max(1, estimated_tokens)


def split_text_into_chunks(text: str, max_chunk_tokens: int) -> List[str]:
    """
    将文本按句子分割成chunks，确保每个chunk不超过指定的token数量
    
    Args:
        text: 要分割的文本
        max_chunk_tokens: 每个chunk的最大token数量
    
    Returns:
        List[str]: 分割后的chunks列表
    """
    if not text or max_chunk_tokens <= 0:
        return [text] if text else []
    
    # 使用智能的句子分割，能够正确处理数字、日期等特殊情况
    sentences = split_text_into_sentences(text)
    
    if not sentences:
        return [text]
    
    chunks = []
    current_chunk = []
    current_tokens = 0
    
    for sentence in sentences:
        sentence_tokens = estimate_tokens(sentence)
        
        # 如果单个句子就超过了限制，需要特殊处理
        if sentence_tokens > max_chunk_tokens:
            # 如果当前chunk不为空，先保存当前chunk
            if current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_tokens = 0
            
            # 对于超长的句子，直接作为一个chunk（虽然会超出限制，但保持完整性）
            chunks.append(sentence)
            continue
        
        # 检查添加这个句子是否会超出限制
        if current_tokens + sentence_tokens > max_chunk_tokens and current_chunk:
            # 保存当前chunk
            chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
            current_tokens = sentence_tokens
        else:
            # 添加到当前chunk
            current_chunk.append(sentence)
            current_tokens += sentence_tokens
    
    # 添加最后一个chunk
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks


def simple_split_sentences(text: str) -> List[str]:
    """
    简单的句子分割，避免复杂的protected模式
    """
    # 预处理：标准化换行符和空格
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r' +', ' ', text)
    
    # 简单的句子分割模式
    # 使用句号、感叹号、问号作为句子结束标记，但要注意数字中的小数点
    sentences = []
    
    # 使用正则表达式分割句子，但保护数字中的小数点
    # 匹配句号、感叹号、问号，但后面必须跟空格和大写字母，或者是行尾
    pattern = r'[.!?]+(?=\s+[A-Z]|\s*$)'
    
    parts = re.split(pattern, text)
    
    for i, part in enumerate(parts):
        part = part.strip()
        if part:
            # 如果不是最后一部分，需要加上句号
            if i < len(parts) - 1:
                part += '.'
            sentences.append(part)
    
    # 过滤太短的句子
    filtered_sentences = []
    for sentence in sentences:
        if len(sentence.strip()) > 10:  # 过滤太短的句子
            filtered_sentences.append(sentence.strip())
    
    return filtered_sentences


def merge_summaries(summaries: List[str]) -> str:
    """
    合并多个summary为一个完整的summary
    
    Args:
        summaries: summary列表
    
    Returns:
        str: 合并后的summary
    """
    if not summaries:
        return ""
    
    if len(summaries) == 1:
        return summaries[0]
    
    # 简单合并，用换行符分隔
    merged = "\n\n".join(summaries)
    
    # 如果合并后的文本太长，可以进一步处理
    # 这里可以根据需要添加更复杂的合并逻辑
    
    return merged 