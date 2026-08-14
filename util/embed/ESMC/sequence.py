import numpy as np
import os
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
from util.constant import *
import json
from typing import Optional, List, Union, Dict

# 全局 ESMC 嵌入缓存
ESMC_EMBEDDINGS: Optional[Dict] = None

def fasta_to_index(fasta: str) -> List[int]:
    """将氨基酸序列字符串转换为索引列表"""
    return [AA_dict[aa] for aa in fasta]

def index_to_fasta(index_list: List[int]) -> str:
    """将索引列表转换为氨基酸序列字符串"""
    return "".join(AA_type[index] for index in index_list)

def onehot_encoding(seq):
    encoding_map = np.eye(len(AA_type))

    residues_map = {}
    for i, aa in enumerate(AA_type):
        residues_map[aa] = encoding_map[i]

    tmp_seq = [residues_map[aa] for aa in seq]
    return np.array(tmp_seq)


def position_encoding(seq_length):
    """
    Position encoding features introduced in "Attention is all your need",
    the b is changed to 50 for the short length of peptides.
    """
    d = 20
    b = 50
    N = seq_length
    value = []
    for pos in range(N):
        tmp = []
        for i in range(d // 2):
            tmp.append(pos / (b ** (2 * i / d)))
        value.append(tmp)

    value = np.array(value)
    pos_encoding = np.zeros((N, d))
    pos_encoding[:, 0::2] = np.sin(value[:, :])
    pos_encoding[:, 1::2] = np.cos(value[:, :])

    return pos_encoding

def save_index_to_fasta(index_list: List[List[int]], output_path: str = "data/output") -> None:
    """将索引列表保存为 FASTA 文件"""
    os.makedirs(output_path, exist_ok=True)
    records = []
    for i, indices in enumerate(index_list):
        seq = index_to_fasta(indices)
        record = SeqRecord(Seq(seq), id=f"output_fasta_{i}", description="")
        records.append(record)
    SeqIO.write(records, f"{output_path}/output.fasta", "fasta")

def load_fasta_to_index(fasta_path: str) -> List[List[int]]:
    """从 FASTA 文件加载序列并转换为索引列表"""
    index_list = []
    for record in SeqIO.parse(fasta_path, "fasta"):
        index_list.append(fasta_to_index(str(record.seq)))
    return index_list

# ==================== ESMC 嵌入 ====================
def load_esmc_embeddings(json_path: str = "data/esmc_embeddings_complete2.json") -> Dict:
    """从 JSON 文件加载预计算的 ESMC 嵌入（键为序列字符串，值为每个残基的 1152 维向量列表）"""
    global ESMC_EMBEDDINGS
    if ESMC_EMBEDDINGS is not None:
        return ESMC_EMBEDDINGS
    try:
        with open(json_path, 'r') as f:
            ESMC_EMBEDDINGS = json.load(f)
        print(f"成功加载 {len(ESMC_EMBEDDINGS)} 个序列的 ESMC 嵌入")
    except Exception as e:
        print(f"警告：加载 ESMC 嵌入失败 ({e})，将使用零向量代替")
        ESMC_EMBEDDINGS = {}
    return ESMC_EMBEDDINGS

def get_esmc_embedding_from_peptide(fasta: str) -> np.ndarray:
    """
    获取序列的 ESMC 嵌入（每个残基 1152 维）
    返回形状 (L, 1152) 的数组
    """
    global ESMC_EMBEDDINGS
    if ESMC_EMBEDDINGS is None:
        ESMC_EMBEDDINGS = load_esmc_embeddings()
    
    if fasta in ESMC_EMBEDDINGS:
        emb = ESMC_EMBEDDINGS[fasta]
        # 确保为二维数组 (L, 1152)
        emb = np.array(emb)
        if emb.ndim == 1:
            # 如果意外是一维（序列级向量），扩展为每个 token 相同向量
            emb = np.tile(emb, (len(fasta), 1))
        return emb
    else:
        # 未找到时返回零向量，但给出警告
        print(f"警告：未找到序列 '{fasta[:20]}...' 的 ESMC 嵌入，返回零向量")
        return np.zeros((len(fasta), 1152))

# ==================== BLOSUM 嵌入 ====================
def get_blosum_embedding_from_peptide(fasta: str) -> np.ndarray:
    """
    获取 BLOSUM62 嵌入，每个氨基酸用 20 维向量表示
    返回形状 (L, 20) 的数组
    """
    embedding = np.array([blosum62[aa] for aa in fasta])
    return embedding  # shape (L, 20)

# ==================== PAM 嵌入 ====================
def get_pam_embedding_from_peptide(fasta: str) -> np.ndarray:
    """
    获取 PAM120 嵌入，每个氨基酸用 20 维向量表示
    返回形状 (L, 20) 的数组
    """
    embedding = np.array([PAM120[aa] for aa in fasta])
    return embedding  # shape (L, 20)

# ==================== 疏水性嵌入 ====================
def get_hydrophobicity_embedding_from_peptide(fasta, scale=100):
    embedding = []
    for aa in fasta:
        embedding = embedding + [score / scale for score in hydrophobicity[aa]]

    return embedding

# ==================== PSSM 嵌入（需 PSI-BLAST）====================
def load_pssm_embedding(pssmDir, pssm_name):
    pssm_path = "{}/{}.pssm".format(pssmDir, pssm_name)
    assert os.path.exists(pssm_path), 'pssm file {} does not exist'.format(pssm_name)

    with open(pssm_path) as f:
        records = f.readlines()[3: -6]

    pssmMatrix = []
    for line in records:
        array = line.strip().split()
        pssmMatrix.append([int(num) for num in array[2:22]])

    return pssmMatrix

def get_pssm_embedding_from_peptide(fasta, db_type="nrdb90", tmp_dir="data/tmp"):
    record_id = "tmp_id"
    record_seq = fasta
    record_name = "tmp_seq"

    assert tmp_dir is not None, "please set the tmp path"
    assert db_type is not None, "please select the psiblast database"

    if db_type == "nrdb90":
        db_path = "data/psiblast/nrdb90/nrdb90"
    else:
        db_path = "data/psiblast/nr/nr"

    save_fasta = SeqRecord(Seq(record_seq), id=record_id, description="")
    SeqIO.write(save_fasta, "{}/{}.fasta".format(tmp_dir, record_name), "fasta")

    input_path = "{}/{}.fasta".format(tmp_dir, record_name)
    output_path = "{}/{}.pssm".format(tmp_dir, record_name)
    assert os.path.exists(output_path), 'pssm file output error'

    log_path = "{}/{}.xml".format(tmp_dir, record_name)
    command = "psiblast -query {} -db {} -evalue 0.001 -num_iterations 3 -num_threads 6 -out_ascii_pssm {} -out {}".format(
        input_path, db_path, output_path, log_path)
    os.system(command)

    embedding = load_pssm_embedding(tmp_dir, record_name)

    return embedding

# ==================== 通用特征分发函数 ====================
def get_embedding_from_peptide(fasta: str, encode_type: str) -> np.ndarray:
    """
    根据类型返回对应的嵌入数组，形状均为 (L, dim)
    支持的类型: 'pssm', 'blosum', 'pam', 'hydrophobicity', 'esmc'
    """
    if encode_type == "pssm":
        return get_pssm_embedding_from_peptide(fasta)
    elif encode_type == "blosum":
        return get_blosum_embedding_from_peptide(fasta)
    elif encode_type == "pam":
        return get_pam_embedding_from_peptide(fasta)
    elif encode_type == "hydrophobicity":
        return get_hydrophobicity_embedding_from_peptide(fasta)
    elif encode_type == "esmc":
        return get_esmc_embedding_from_peptide(fasta)
    else:
        raise ValueError(f"不支持的编码类型: {encode_type}")

# 为了向后兼容，保留别名
get_embedding_form_peptide = get_embedding_from_peptide

def get_bio_embedding_for_sequence(fasta, encode_type=None):
    if encode_type is None:
        encode_type = ["blosum", "pam", "hydrophobicity", "esmc"]
    
    embedding_list = []
    L = len(fasta)
    
    for aa_type in encode_type:
        emb = get_embedding_form_peptide(fasta=fasta, encode_type=aa_type)
        
        # 转换为 numpy 数组
        if isinstance(emb, list):
            emb = np.array(emb)
        
        # 确保是二维 (L, d)
        if emb.ndim == 1:
            # 一维数组，假设总元素数能被 L 整除
            emb = emb.reshape(L, -1)
        elif emb.ndim == 2 and emb.shape[0] != L:
            # 如果已经是二维但第一维不是 L，尝试转置（可能形状是 (d, L)）
            if emb.shape[1] == L:
                emb = emb.T
            else:
                raise ValueError(f"特征 {aa_type} 的形状 {emb.shape} 无法对齐到序列长度 {L}")
        
        embedding_list.append(emb)
    
    # 检查所有特征的第一维长度是否一致
    lengths = [e.shape[0] for e in embedding_list]
    if len(set(lengths)) != 1:
        raise ValueError(f"特征长度不一致: {lengths}")
    
    # 合并特征
    embedding = np.concatenate(embedding_list, axis=1)
    return embedding