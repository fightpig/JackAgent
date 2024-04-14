# JackAgent
Learning agent

## 安装环境
1. conda create -n agent python=3.10 -y

## 报错
>   File "D:\ProgramData\miniconda3\envs\agent\lib\site-packages\langchain_core\prompts\loading.py", line 30, in load_prompt_from_config
    return prompt_loader(config)
  File "D:\ProgramData\miniconda3\envs\agent\lib\site-packages\langchain_core\prompts\loading.py", line 112, in _load_prompt
    config = _load_template("template", config)
  File "D:\ProgramData\miniconda3\envs\agent\lib\site-packages\langchain_core\prompts\loading.py", line 47, in _load_template
    template = f.read()
UnicodeDecodeError: 'gbk' codec can't decode byte 0x80 in position 8: illegal multibyte sequence

```python
        if template_path.suffix == ".txt":
            with open(template_path) as f:
                template = f.read()
        
        # 在源码中，将上面代码改成以下代码        
        if template_path.suffix == ".txt":
            with open(template_path, 'r', encoding='utf-8') as f:
                template = f.read()
```