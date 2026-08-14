import json
import subprocess
import os
import pandas as pd
import numpy as np

# ==================== 固定生成数量 ====================
TARGET_NUM = 200   # 每类癌症生成200条

# ==================== 1. 读取真实数据并过滤长度 ====================
real_csv = "D:/Issue/2024_2025/code/data/processed/CancerPPD_merged_filtered.csv"
df_real = pd.read_csv(real_csv)

# 直接按 'Tissue Affected' 分组，其值就是短名称（如 "Breast", "Lung"）
cancer_lengths = {}
filtered_out = 0
for short_name, group in df_real.groupby('Tissue Affected'):
    # 过滤长度 ≤ 50 的序列（若之前已过滤，此步可省略，但保留安全）
    valid_lengths = [len(seq) for seq in group['Sequence'] if len(seq) <= 50]
    n_filtered = len(group) - len(valid_lengths)
    filtered_out += n_filtered
    if not valid_lengths:
        print(f"警告：{short_name} 过滤后无有效序列（原 {len(group)} 条全被剔除），跳过")
        continue
    cancer_lengths[short_name] = valid_lengths
    print(f"{short_name}: 原始 {len(group)} 条, 过滤后 {len(valid_lengths)} 条 (剔除 {n_filtered}), 长度范围 [{min(valid_lengths)}, {max(valid_lengths)}]")

print(f"\n全局过滤统计：共剔除 {filtered_out} 条长度 >50 的序列（若已预过滤则可能为0）")

# ==================== 2. 验证短名称是否在 cancer_to_idx.json 中 ====================
with open('cancer_to_idx.json', 'r', encoding='utf-8') as f:
    cancer_to_idx = json.load(f)
for short_name in list(cancer_lengths.keys()):
    if short_name not in cancer_to_idx:
        print(f"警告：短名 '{short_name}' 不在 cancer_to_idx.json 中，将跳过生成")
        # 若需移除，可取消注释下面一行
        # del cancer_lengths[short_name]

# ==================== 3. 批量生成 ====================
output_dir = "generated_peptides_matched"
os.makedirs(output_dir, exist_ok=True)

for short_name, lengths in cancer_lengths.items():
    if short_name not in cancer_to_idx:
        print(f"跳过 {short_name}（不在映射文件中）")
        continue

    if len(lengths) >= TARGET_NUM:
        sampled_lengths = np.random.choice(lengths, size=TARGET_NUM, replace=False).tolist()
        sample_info = "无放回抽样"
    else:
        sampled_lengths = np.random.choice(lengths, size=TARGET_NUM, replace=True).tolist()
        sample_info = f"有放回抽样 (原始数据仅 {len(lengths)} 条)"

    np.random.shuffle(sampled_lengths)
    lengths_str = ','.join(map(str, sampled_lengths))

    print(f"正在为 {short_name} 生成 {TARGET_NUM} 条序列 ({sample_info})...")
    command = [
        "python", "generate_conditional_peptides.py",
        "--cancer", short_name,
        "--lengths", lengths_str,
        "--ckpt", "models/prefix_tuned_full/epoch=48-total_loss=0.0314-seq_score=0.1667.ckpt"
    ]
    try:
        subprocess.run(command, check=True)
        print(f"✅ {short_name} 生成完成\n")
    except subprocess.CalledProcessError as e:
        print(f"❌ {short_name} 生成失败: {e}\n")