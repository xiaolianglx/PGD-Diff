# generate_conditional_peptides.py 
# python generate_conditional_peptides.py  --cancer Breast  --n_seq 500  --min_len 16  --max_len 30

import torch
import json
import numpy as np
from pathlib import Path
from models.prefix_tuned.PGD_Diff_condition import PGD_Diff


def parse_lengths(lengths_str):
    """解析逗号分隔的长度列表"""
    return [int(x) for x in lengths_str.split(',')]


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, default="models/prefix_tuned/last.ckpt")
    parser.add_argument('--cancer', type=str, required=True)
    parser.add_argument('--n_seq', type=int, default=None,
                        help="生成数量（当 --lengths 未提供时使用）")
    parser.add_argument('--lengths', type=str, default=None,
                        help="逗号分隔的长度列表，例如 '15,20,18,25'，数量即为 n_seq")
    parser.add_argument('--min_len', type=int, default=5,
                        help="随机长度的下限（仅在未提供 --lengths 且提供 --n_seq 时有效）")
    parser.add_argument('--max_len', type=int, default=50,
                        help="随机长度的上限（仅在未提供 --lengths 且提供 --n_seq 时有效）")
    parser.add_argument('--fasta_out', action='store_true')
    args = parser.parse_args()

    # 确定生成数量和长度列表
    if args.lengths:
        lengths = parse_lengths(args.lengths)
        n_seq = len(lengths)
    else:
        if args.n_seq is None:
            raise ValueError("请提供 --n_seq 或 --lengths")
        n_seq = args.n_seq
        rng = np.random.default_rng()
        lengths = rng.integers(args.min_len, args.max_len + 1, size=n_seq).tolist()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ========== 1. 先读取映射文件，获取类别数 ==========
    ckpt_dir = Path(args.ckpt).parent
    mapping_file = ckpt_dir / "cancer_to_idx.json"
    if not mapping_file.exists():
        raise FileNotFoundError(f"未找到映射文件: {mapping_file}")
    with open(mapping_file, 'r') as f:
        cancer_to_idx = json.load(f)
    num_cancer_types = len(cancer_to_idx)
    print(f"模型支持 {num_cancer_types} 种癌症类型")

    # ========== 2. 加载模型，传入正确的 num_cancer_types ==========
    model = PGD_Diff.load_from_checkpoint(
        args.ckpt,
        use_prefix_condition=True,
        num_cancer_types=num_cancer_types,
        condition_mode='cross_attn',
        prefix_len=20,          # 就是这一行！与训练时保持一致
        loss_weight=1.0,
        strict=False,
        map_location=device
    )
    model.eval()
    model.to(device)

    # ========== 3. 验证癌症名称 ==========
    if args.cancer not in cancer_to_idx:
        print(f"未知癌症: {args.cancer}")
        print(f"可用: {list(cancer_to_idx.keys())}")
        return

    cancer_idx = cancer_to_idx[args.cancer]
    print(f"生成针对 {args.cancer} 的 {n_seq} 条序列，长度列表前5个: {lengths[:5]}...")

    # ========== 4. 调用模型生成 ==========
    sequences, _, _ = model.denoise_seq_sample_with_prefix(
        cancer_idx=cancer_idx,
        n_seq=n_seq,
        seq_length=lengths,
        fasta_out_statue=args.fasta_out
    )

    # ========== 5. 保存结果 ==========
    safe_name = args.cancer.lower().replace(' ', '_')
    out_file = ckpt_dir / f"generated_{safe_name}_peptides.txt"
    with open(out_file, 'w') as f:
        for seq in sequences:
            f.write(seq + '\n')
    print(f"序列已保存至 {out_file}")


if __name__ == '__main__':
    main()