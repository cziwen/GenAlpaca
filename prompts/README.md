# Prompt模板使用说明

## 模板格式

每个prompt模板用 `---` 分隔，模板中可以包含以下占位符：

- `{max_q}`: 最大问题数量
- `{report_text}`: 财报文本内容

## 模板示例

```
请从财务角度分析以下财报，提出 {max_q} 个关键问题：
- "instruction": 财务分析问题
- "input": 分析背景
- "output": 详细分析

财报内容：{report_text}
---
Please analyze this financial report from a business perspective and ask {max_q} strategic questions:
- "instruction": business question
- "input": context
- "output": analysis

Report: {report_text}
```

## 添加新模板

1. 编辑 `prompts/templates.txt` 文件
2. 在文件末尾添加新模板
3. 用 `---` 分隔每个模板
4. 确保包含必要的占位符 `{max_q}` 和 `{report_text}`

## 注意事项

- 每个模板都应该要求返回JSON格式
- 建议包含中英文混合模板以增加多样性
- 模板应该涵盖不同的分析角度（财务、战略、风险等）
- 确保模板语法正确，避免格式化错误 