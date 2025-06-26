# utils/text_cleaning.py
import re

def clean_text(text: str) -> str:
    """
    对原始文本进行基本清洗：统一换行、去除多余空行、首尾空白
    """
    # 统一换行符
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    # 合并连续空行
    text = re.sub(r"\n{2,}", "\n", text)
    # 去除首尾空白
    return text.strip()


def chunk_text(text: str, chunk_size: int) -> list[str]:
    """
    将清洗后的文本分块，尽量按照句子边界切分，每块长度不超过 chunk_size 字符
    """
    # 简单拆句，以中文标点或英文句号、问号、感叹号为分隔
    sentences = re.split(r'(?<=[。！？!?!.])\s*', text)
    chunks = []
    current = ""
    for sent in sentences:
        if not sent:
            continue
        # 预估长度
        if len(current) + len(sent) <= chunk_size:
            current += sent
        else:
            # 保存当前块
            if current:
                chunks.append(current)
            current = sent
    # 添加最后一块
    if current:
        chunks.append(current)
    return chunks