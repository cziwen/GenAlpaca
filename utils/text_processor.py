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
    
    # 定义需要保护的模式
    protected_patterns = [
        # 数字模式
        r'\$\d+(?:,\d{3})*(?:\.\d+)?',      # $1,234.56, $1.23
        r'\d+(?:,\d{3})*(?:\.\d+)?%',       # 12.34%, 1.000%, 5%
        r'\d+(?:,\d{3})*(?:\.\d+)?',        # 1,234.56, 1.23, 123
        r'\d+\.\d+',                        # 12.34, 1.000
        r'\d+',                             # 123, 1
        
        # 日期模式
        r'\d{1,2}/\d{1,2}/\d{2,4}',        # 12/31/2023, 1/1/24
        r'\d{4}-\d{1,2}-\d{1,2}',          # 2023-12-31
        r'\w+ \d{1,2},? \d{4}',            # December 31, 2023
        
        # 时间模式
        r'\d{1,2}:\d{2}(?::\d{2})?(?: [AP]M)?',  # 12:30, 12:30:45, 12:30 PM
        
        # 百分比和比率
        r'\d+(?:\.\d+)?x',                  # 2.5x, 10x
        r'\d+(?:\.\d+)?%',                  # 25%, 12.5%
        
        # 货币和金额
        r'\$\d+(?:,\d{3})*(?:\.\d+)?',      # $1,234.56
        r'\d+(?:,\d{3})*(?:\.\d+)? dollars', # 1,234.56 dollars
        
        # 股票代码
        r'[A-Z]{1,5}',                      # AAPL, GOOGL
        
        # 年份
        r'\b(?:19|20)\d{2}\b',              # 1999, 2023
        
        # 季度
        r'Q[1-4]',                          # Q1, Q2, Q3, Q4
        r'\d{4} Q[1-4]',                    # 2023 Q1
        
        # 财务术语
        r'EBITDA', r'ROI', r'ROE', r'P/E', r'P/E ratio',
        r'earnings per share', r'EPS',
        r'revenue', r'profit', r'loss',
        r'assets', r'liabilities', r'equity',
        
        # 公司名称模式
        r'[A-Z][a-z]+ (?:Inc\.|Corp\.|LLC|Ltd\.)',
        r'[A-Z][a-z]+ & [A-Z][a-z]+',
    ]
    
    # 临时替换需要保护的模式
    protected_map = {}
    for i, pattern in enumerate(protected_patterns):
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for j, match in enumerate(matches):
            placeholder = f"__PROTECTED_{i}_{j}__"
            protected_map[placeholder] = match.group()
            text = text[:match.start()] + placeholder + text[match.end():]
    
    # 句子分割模式
    sentence_patterns = [
        r'[.!?]+(?=\s+[A-Z])',  # 句号、感叹号、问号后跟大写字母
        r'[.!?]+\s*\n',         # 句号、感叹号、问号后跟换行
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
    
    # 恢复被保护的模式
    restored_sentences = []
    for sentence in sentences:
        for placeholder, original in protected_map.items():
            sentence = sentence.replace(placeholder, original)
        restored_sentences.append(sentence)
    
    # 清理和过滤
    cleaned_sentences = []
    for sentence in restored_sentences:
        sentence = sentence.strip()
        if sentence and len(sentence) > 10:  # 过滤太短的句子
            cleaned_sentences.append(sentence)
    
    return cleaned_sentences 