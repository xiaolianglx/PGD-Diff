import os
import re
from pathlib import Path

def extract_imports(file_path):
    """提取单个 Python 文件中的 import 语句"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 匹配 import xxx 和 from xxx import yyy
    patterns = [
        r'^import\s+([a-zA-Z_][a-zA-Z0-9_.]*)',
        r'^from\s+([a-zA-Z_][a-zA-Z0-9_.]*)\s+import',
    ]
    
    imports = set()
    for line in content.split('\n'):
        line = line.strip()
        for pattern in patterns:
            match = re.match(pattern, line)
            if match:
                # 只取包名的第一级（如 import torch.nn 只取 torch）
                pkg = match.group(1).split('.')[0]
                imports.add(pkg)
                break
    
    return imports

# 扫描项目目录
root = Path(".")
all_imports = set()

for py_file in root.rglob("*.py"):
    if "site-packages" in str(py_file) or "dist-packages" in str(py_file):
        continue
    try:
        imports = extract_imports(py_file)
        all_imports.update(imports)
    except:
        pass

print("项目引用的包（按字母排序）:")
for pkg in sorted(all_imports):
    print(f"  {pkg}")

# 保存到文件
with open("imports.txt", "w") as f:
    for pkg in sorted(all_imports):
        f.write(pkg + "\n")

print(f"\n已保存到 imports.txt，共 {len(all_imports)} 个包")