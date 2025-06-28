# GenAlpaca - 财报分析问答生成系统

基于OpenAI/DeepSeek API的财报分析问答生成系统，自动提取财报关键信息并生成专业问答。

## 安装

```bash
pip install -r requirements.txt
```

## 配置

复制环境变量示例并设置API密钥：
```bash
cp env_example.txt .env
# 编辑.env文件，设置OPENAI_API_KEY和DEEPSEEK_API_KEY
# 或者 直接在 config里设置即可
```

## 使用方法

### 生成问答数据
```bash
# 处理所有文件
python main.py

# 从进度文件恢复
python main.py -r output/20241201_143022/progress.txt
```

### 过滤数据
```bash
# 基本过滤
python filter.py -i output/20241201_143022/result.jsonl

# 显示详细日志
python filter.py -i output/20241201_143022/result.jsonl -v
```

## 输出文件

- `output/时间戳/result.jsonl` - 生成的问答数据
- `output/时间戳/result_filtered.jsonl` - 过滤后的数据
- `output/时间戳/result_deleted.jsonl` - 被删除的数据及原因 