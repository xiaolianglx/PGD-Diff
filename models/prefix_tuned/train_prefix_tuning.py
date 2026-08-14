# finetune.py (修改版)
# 修改点：
# 1. 冻结主干参数，只训练条件模块（cancer_embedding, condition_mlp, cross_attn, classifier, uncond_embedding, cond_proj）
# 2. ModelCheckpoint 保存 top 10 个最佳检查点（按 total_loss 升序），以便回溯早期模型
# 3. 同时保留最后一个检查点 (save_last=True)

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from types import SimpleNamespace, MethodType
import json
import os
import numpy as np
from models.prefix_tuned.PGD_Diff_condition import PGD_Diff
from util.embed.sequence import onehot_encoding

# ==================== Dataset（不变） ====================
class PrefixPeptideDataset(Dataset):
    def __init__(self, txt_path, cancer_to_idx, max_seq_len=50):
        self.data = []
        self.cancer_to_idx = cancer_to_idx
        self.max_seq_len = max_seq_len
        with open(txt_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                end_bracket = line.find(']')
                if end_bracket == -1:
                    continue
                cancer_str = line[1:end_bracket].strip()
                seq = line[end_bracket+1:].strip().upper()
                if not seq:
                    continue
                if cancer_str not in cancer_to_idx:
                    continue
                cancer_idx = cancer_to_idx[cancer_str]
                if len(seq) > max_seq_len:
                    seq = seq[:max_seq_len]
                logit = onehot_encoding(seq)
                pos = np.zeros((len(seq), 4, 3), dtype=np.float32)
                self.data.append({
                    'fasta': seq,
                    'logit': torch.tensor(logit, dtype=torch.float32),
                    'pos': torch.tensor(pos, dtype=torch.float32),
                    'cancer_idx': cancer_idx,
                    'length': len(seq),
                })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(batch):
    batch.sort(key=lambda x: x['length'], reverse=True)
    logits, poses, fastas, cancer_idxs, lengths = [], [], [], [], []
    for item in batch:
        logits.append(item['logit'])
        poses.append(item['pos'])
        fastas.append(item['fasta'])
        cancer_idxs.append(item['cancer_idx'])
        lengths.append(item['length'])
    logit_cat = torch.cat(logits, dim=0)
    pos_cat = torch.cat(poses, dim=0)
    batch_index = []
    for i, L in enumerate(lengths):
        batch_index.extend([i] * L)
    batch_index = torch.tensor(batch_index, dtype=torch.long)
    x_dummy = torch.zeros(len(batch_index), 46)
    return SimpleNamespace(
        x=x_dummy,
        pos=pos_cat,
        fasta=fastas,
        logit=logit_cat,
        cancer_idx=torch.tensor(cancer_idxs),
        batch_index=batch_index,
        lengths=lengths,
        nonamp_fasta=[],
        nonamp_logit=torch.empty(0, 20),
        nonamp_pos=torch.empty(0, 4, 3),
    )

def build_cancer_mapping(txt_path):
    cancers = set()
    with open(txt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                cancer = line.split(' ', 1)[0].strip('[]')
                cancers.add(cancer)
    return {c: i for i, c in enumerate(sorted(cancers))}

def main():
    txt_path = "data/processed/prefix_training_data.txt"
    pretrained_ckpt = r"data/output/test/"
    output_dir = "models/prefix_tuned"
    os.makedirs(output_dir, exist_ok=True)

    cancer_to_idx = build_cancer_mapping(txt_path)
    num_cancers = len(cancer_to_idx)
    print(f"癌症类别数: {num_cancers}")

    # ========== 加载模型 ==========
    model = PGD_Diff.load_from_checkpoint(
        pretrained_ckpt,
        use_prefix_condition=True,
        num_cancer_types=num_cancers,
        condition_mode='cross_attn',
        prefix_len=20,
        loss_weight=1.0,
        strict=False,
    )

    # ========== 修改点：冻结主干，只训练条件模块 ==========
    for name, param in model.named_parameters():
        param.requires_grad = False

    # 解冻条件相关参数（含新增的 classifier、uncond_embedding 和 cond_proj）
    trainable_names = ['cancer_embedding', 'condition_mlp', 'cross_attn', 'classifier', 'uncond_embedding', 'cond_proj']
    for name, param in model.named_parameters():
        if any(x in name for x in trainable_names):
            param.requires_grad = True

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"可训练参数: {trainable_params} / 总参数: {total_params} ({100*trainable_params/total_params:.2f}%)")

    # ========== 优化器：只使用一个学习率（因为只有条件模块可训练） ==========
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4
    )
    # 覆盖 configure_optimizers 方法
    def custom_configure_optimizers(self):
        return optimizer
    model.configure_optimizers = MethodType(custom_configure_optimizers, model)

    # ========== 梯度测试 ==========
    dummy_batch = torch.utils.data.Subset(PrefixPeptideDataset(txt_path, cancer_to_idx, max_seq_len=50), range(2))
    dummy_loader = DataLoader(dummy_batch, batch_size=2, collate_fn=collate_fn)
    batch = next(iter(dummy_loader))
    model.train()
    loss, _, _, _, _ = model.get_loss(batch)
    print(f"测试 batch loss: {loss.item()}, requires_grad: {loss.requires_grad}")
    loss.backward()
    print("梯度测试通过！反向传播成功。")
    model.zero_grad()

    # ========== 训练 ==========
    dataset = PrefixPeptideDataset(txt_path, cancer_to_idx, max_seq_len=50)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn, num_workers=0)

    # ========== 修改点：保存 top 10 个最佳检查点 + 最后一个 ==========
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=output_dir,
        monitor='total_loss',
        mode='min',
        filename='{epoch:02d}-{total_loss:.4f}-{seq_score:.4f}',
        save_top_k=10,          # 保存 loss 最低的 10 个（可调大如 20）
        save_last=True,         # 同时保存最后一个 epoch 的检查点
    )

    trainer = pl.Trainer(
        max_epochs=60,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        log_every_n_steps=5,
        callbacks=[checkpoint_callback],
        gradient_clip_val=1.0,
    )
    trainer.fit(model, dataloader)

    with open(os.path.join(output_dir, "cancer_to_idx.json"), "w") as f:
        json.dump(cancer_to_idx, f)
    print(f"全量微调完成，模型保存在 {output_dir}")

if __name__ == '__main__':
    main()