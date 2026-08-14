import numpy as np
from Bio.Align import substitution_matrices
from modlamp.descriptors import PeptideDescriptor, GlobalDescriptor
from Bio import SeqIO, Align
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
from tqdm import tqdm
import torch
import os
from transformers import AutoModelForMaskedLM, AutoTokenizer

def evaluate_fasta(file_path, check_type):
    assert check_type in ["instability", "ez", "TM_tend"], "evaluate type error"
    if check_type == "instability":
        check_funcation = instability_score
    elif check_type == "ez":
        check_funcation = ez_score
    elif check_type == "TM_tend":
        check_funcation = TM_tend_score  # 修正这里

    fasta_list = list(SeqIO.parse(file_path, "fasta"))
    score_list = []
    for fasta in tqdm(fasta_list):
        fasta_seq = str(fasta.seq)
        score = check_funcation(fasta_seq)
        score_list.append(score)

    return np.array(score_list)


# https://doi.org/10.1093/protein/4.2.155
# https://doi.org/10.1093/bioinformatics/btx285 modlamp
def instability_score(fasta):
    desc = GlobalDescriptor(fasta)
    desc.instability_index()
    score = desc.descriptor

    return score.squeeze()


# https://doi.org/10.1016/j.jmb.2006.09.020
# https://academic.oup.com/bioinformatics/article/33/17/2753/3796392
def ez_score(fasta, window=10):
    AMP = PeptideDescriptor(fasta, 'Ez')
    AMP.calculate_global(window)
    score = AMP.descriptor

    return score.squeeze()


def TM_tend_score(fasta, window=7):
    AMP = PeptideDescriptor(fasta, 'TM_tend')
    AMP.calculate_global(window)
    score = AMP.descriptor

    return score.squeeze()


def match_score(fasta):
    AMP_path = "data/source/fasta/ACP.fasta"
    aligner = Align.PairwiseAligner()
    aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")

    score_list = []

    for record in SeqIO.parse(AMP_path, "fasta"):
        amp_str = record.seq
        alignments = aligner.align(amp_str, fasta)
        score = alignments.score

        score_list.append(score)

    score_list = np.stack(score_list)
    return score_list.mean()


def match_score_batch(fasta_path):
    fasta_list = list(SeqIO.parse(fasta_path, "fasta"))
    record_list = []

    for index in tqdm(range(len(fasta_list))):
        fasta_id = fasta_list[index].id
        fasta_str = fasta_list[index].seq
        score = match_score(fasta_str)

        record = {"id": fasta_id, "score": score}
        record_list.append(record)

    return record_list





import numpy as np
import math
from collections import Counter
from Bio import SeqIO, Align
from Bio.Align import substitution_matrices
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import rbf_kernel
from tqdm import tqdm

import numpy as np
from collections import defaultdict, Counter

import numpy as np
from collections import defaultdict, Counter

#def calculate_ppl(sequences, n=2):
    # 确保所有序列都是列表形式
#    sequences = [list(seq) if isinstance(seq, str) else seq for seq in sequences]
    
    # 1. 构建n-gram模型（使用条件概率）
#    ngram_counts = defaultdict(Counter)
#    context_counts = Counter()
    
    # 添加开始和结束标记
#    start_symbol = "<s>"
#    end_symbol = "</s>"
    
#    for seq in sequences:
        # 确保序列是列表
#        if isinstance(seq, str):
#            seq = list(seq)
            
        # 添加开始和结束标记
#        padded_seq = [start_symbol] * (n-1) + seq + [end_symbol]
        
        # 统计n-gram
#        for i in range(len(padded_seq) - n + 1):
#            context = tuple(padded_seq[i:i+n-1])
#            next_token = padded_seq[i+n-1]
#            
#            ngram_counts[context][next_token] += 1
#            context_counts[context] += 1
    
    # 2. 计算困惑度（使用加一平滑）
#    log_prob_sum = 0.0
#    total_tokens = 0
    
    # 获取词汇表大小（所有出现的氨基酸和标记）
#    vocab = set()
#    for seq in sequences:
#        vocab.update(seq)
#    vocab.update([start_symbol, end_symbol])
#    vocab_size = len(vocab)
    
#    for seq in sequences:
        # 确保序列是列表
#        if isinstance(seq, str):
#            seq = list(seq)
            
        # 添加开始标记（不需要结束标记，因为只计算序列本身）
#        padded_seq = [start_symbol] * (n-1) + seq
        
        # 计算序列的概率
#        for i in range(len(seq)):
#            context = tuple(padded_seq[i:i+n-1])
#            next_token = seq[i]
            
            # 加一平滑 (Laplace smoothing)
#            count = ngram_counts[context].get(next_token, 0)
#            total_context = context_counts.get(context, 0)
#            prob = (count + 1) / (total_context + vocab_size)
            
#            log_prob_sum += np.log(prob)
#            total_tokens += 1
    
    # 避免除以零
#    if total_tokens == 0:
#        return float('inf')
    
    # 计算困惑度
#    avg_log_prob = log_prob_sum / total_tokens
#    perplexity = np.exp(-avg_log_prob)
#    return perplexity

#。
#def calculate_ppl(protein_sequences, model_path=None):
#    """
#    使用您本地的ESM-2模型计算蛋白质序列集的困惑度(PPL)
#    
#    参数:
#        protein_sequences: 蛋白质序列列表，例如 ["ACDEF", "GHIKLM"]
#        model_path: ESM-2模型在您本地的完整路径
#                   默认使用您图片中显示的路径
#    """
#    # 设置设备
#    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#    print(f"使用设备: {device}")
#    
#    # 1. 设置模型路径（使用您图片中的路径）
#    if model_path is None:
#        model_path = r"D:/jupyter code/MPOGAN-main/MPOGAN-main/models/esm2_t6_8M_UR50D"
    
#    # 检查路径是否存在
#    if not os.path.exists(model_path):
#        raise FileNotFoundError(f"模型路径不存在: {model_path}")
    
#    print(f"从本地路径加载模型: {model_path}")
    
    # 2. 从本地加载模型和分词器
#    tokenizer = AutoTokenizer.from_pretrained(model_path)
#    model = AutoModelForMaskedLM.from_pretrained(model_path)
#    model.to(device)
#    model.eval()  # 设置为评估模式
#
#    total_loss = 0.0
#    total_tokens = 0

    # 确保所有序列是字符串形式
#    sequences = [seq if isinstance(seq, str) else ''.join(seq) for seq in protein_sequences]

    # 3. 处理每条序列
#    with torch.no_grad():
#        for i, seq in enumerate(sequences):
#            print(f"处理序列 {i+1}/{len(sequences)}: {seq[:20]}...")
            
            # 将序列转换为模型输入格式
#            try:
#                inputs = tokenizer(seq, return_tensors="pt", truncation=True, max_length=1024)
#                input_ids = inputs["input_ids"].to(device)
#                attention_mask = inputs["attention_mask"].to(device)
#                sequence_length = input_ids.size(1)
                
                # 计算损失
#                outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
#                loss = outputs.loss
                
#                total_loss += loss.item() * sequence_length
#                total_tokens += sequence_length
                
#                print(f"  序列长度: {sequence_length}, 当前损失: {loss.item():.4f}")
                
#            except Exception as e:
#                print(f"  处理序列时出错: {e}")
#                continue

    # 4. 计算平均困惑度
#    if total_tokens == 0:
#        return float('inf'), float('inf')
#    
#    average_loss = total_loss / total_tokens
#    perplexity = np.exp(average_loss)

#    return perplexity


import numpy as np
from collections import Counter
import math

def calculate_bleu(reference_seqs, candidate_seqs, weights=(0.25, 0.25, 0.25, 0.25)):
    def get_ngrams(seq, n):
        """生成n-gram列表"""
        return [tuple(seq[i:i+n]) for i in range(len(seq)-n+1)]
    
    bleu_scores = []
    max_n = len(weights)  # 最大n-gram长度
    
    for candidate in candidate_seqs:
        # 对于每个候选序列，计算与所有参考序列的BLEU评分
        candidate_bleu_scores = []
        
        for reference in reference_seqs:
            # 计算不同n-gram的精度（使用截断计数）
            precisions = []
            
            for n_val in range(1, max_n+1):
                # 获取候选序列的n-gram计数
                cand_ngrams = get_ngrams(candidate, n_val)
                cand_counts = Counter(cand_ngrams)
                
                # 获取参考序列的n-gram计数
                ref_ngrams = get_ngrams(reference, n_val)
                ref_counts = Counter(ref_ngrams)
                
                if not cand_ngrams:
                    precision = 0
                else:
                    # 计算匹配的n-gram数量（考虑截断）
                    match_count = 0
                    for ngram in cand_counts:
                        # 候选中的计数不能超过参考中的最大计数
                        match_count += min(cand_counts[ngram], ref_counts.get(ngram, 0))
                    
                    precision = match_count / len(cand_ngrams)
                
                precisions.append(precision)
            
            # 计算简洁惩罚 (Brevity Penalty) - 更准确
            c = len(candidate)
            r = len(reference)
            if c > r:
                bp = 1.0
            else:
                bp = math.exp(1 - r / c) if c > 0 else 0.0
            
            # 计算BLEU评分 - 避免log(0)错误
            log_sum = 0
            valid_precisions = 0
            for w, p in zip(weights, precisions):
                if p > 0:
                    log_sum += w * math.log(p)
                    valid_precisions += 1
            
            # 如果有有效的精度值
            if valid_precisions > 0:
                bleu_score = bp * math.exp(log_sum)
            else:
                bleu_score = 0
            
            candidate_bleu_scores.append(bleu_score)
        
        # 取最高BLEU评分作为该候选序列的评分
        bleu_scores.append(max(candidate_bleu_scores))
    
    # 返回平均BLEU分数
    return np.mean(bleu_scores) if bleu_scores else 0.0

def calculate_mmd(seqs1, seqs2, kernel='rbf', gamma=None):
    """
    计算最大平均差异(MMD)，评估两个序列分布之间的差异
    """
    # 将序列转换为数值特征（这里使用简单的氨基酸组成）
    def seq_to_features(seq):
        aa_list = 'ACDEFGHIKLMNPQRSTVWY'
        features = np.zeros(len(aa_list))
        for aa in seq:
            if aa in aa_list:
                features[aa_list.index(aa)] += 1
        # 归一化
        if len(seq) > 0:
            features /= len(seq)
        return features
    
    # 转换为特征向量
    features1 = np.array([seq_to_features(seq) for seq in seqs1])
    features2 = np.array([seq_to_features(seq) for seq in seqs2])
    
    # 计算MMD
    if kernel == 'rbf':
        if gamma is None:
            # 使用中位数启发式设置gamma
            XX = cdist(features1, features1)
            YY = cdist(features2, features2)
            XY = cdist(features1, features2)
            median_dist = np.median(np.concatenate([XX.ravel(), YY.ravel(), XY.ravel()]))
            gamma = 1.0 / (2.0 * median_dist**2) if median_dist > 0 else 1.0
        
        K_XX = rbf_kernel(features1, features1, gamma=gamma)
        K_YY = rbf_kernel(features2, features2, gamma=gamma)
        K_XY = rbf_kernel(features1, features2, gamma=gamma)
        
        mmd = np.mean(K_XX) + np.mean(K_YY) - 2 * np.mean(K_XY)
    else:
        # 线性核
        mmd = np.mean(np.dot(features1, features1.T)) + \
              np.mean(np.dot(features2, features2.T)) - \
              2 * np.mean(np.dot(features1, features2.T))
    
    return mmd

def calculate_jaccard_diversity(sequences):
    """计算序列间Jaccard相似度的补集"""
    total_similarity = 0
    count = 0
    
    for i in range(len(sequences)):
        for j in range(i+1, len(sequences)):
            set1 = set(sequences[i])
            set2 = set(sequences[j])
            
            intersection = len(set1 & set2)
            union = len(set1 | set2)
            
            jaccard_sim = intersection / union if union > 0 else 0
            total_similarity += jaccard_sim
            count += 1
    
    avg_similarity = total_similarity / count if count > 0 else 0
    return 1 - avg_similarity  # 多样性 = 1 - 平均相似度

def calculate_shannon_entropy(sequences):
    """
    计算序列集合的香农熵，评估氨基酸分布的多样性
    """
    # 将所有序列连接成一个长字符串
    all_sequences = ''.join(sequences)
    
    # 如果序列为空，返回0
    if not all_sequences:
        return 0
    
    # 统计每个氨基酸的出现次数
    aa_counts = Counter(all_sequences)
    total_chars = len(all_sequences)
    
    # 计算香农熵
    entropy = 0
    for count in aa_counts.values():
        probability = count / total_chars
        entropy -= probability * math.log2(probability)
    
    return entropy

def evaluate_all_metrics(generated_fasta_path, reference_fasta_path=None, k_values=[2, 3, 4]):
    """
    综合评估多肽序列的多个指标
    
    参数:
        generated_fasta_path: 生成的序列FASTA文件路径
        reference_fasta_path: 参考序列FASTA文件路径
        k_values: 要计算的k-mer长度列表
    
    返回:
        包含所有评估指标的字典
    """
    # 读取生成的序列
    generated_seqs = [str(record.seq) for record in SeqIO.parse(generated_fasta_path, "fasta")]
    
    if not generated_seqs:
        print("警告: 没有找到任何生成序列")
        return None
    
    # 读取参考序列
    if reference_fasta_path is None:
        reference_fasta_path = "./data/source/fasta/ACP.fasta"
    
    try:
        reference_seqs = [str(record.seq) for record in SeqIO.parse(reference_fasta_path, "fasta")]
    except:
        print(f"警告: 无法读取参考序列文件 {reference_fasta_path}，使用默认ACP序列")
        reference_fasta_path = "./data/source/fasta/ACP.fasta"
        reference_seqs = [str(record.seq) for record in SeqIO.parse(reference_fasta_path, "fasta")]
    
    print(f"评估 {len(generated_seqs)} 条生成序列，参考 {len(reference_seqs)} 条序列...")
    
    # 计算所有指标
    results = {}
    
    # 计算困惑度(PPL)
    #print("计算困惑度(PPL)...")
    #results["PPL"] = calculate_ppl(generated_seqs)
    
    # 计算BLEU评分
    print("计算BLEU评分...")
    results["BLEU"] = calculate_bleu(reference_seqs, generated_seqs)
    
    # 计算MMD
    print("计算最大平均差异(MMD)...")
    results["MMD"] = calculate_mmd(generated_seqs, reference_seqs)
    
    # 计算Jaccard多样性
    print("计算最大平均差异(MMD)...")
    results["Jaccard_Diversity"] = calculate_jaccard_diversity(generated_seqs)

    # 计算香农熵
    print("计算香农熵...")
    results["Shannon_Entropy"] = calculate_shannon_entropy(generated_seqs)
    
    # 添加基本信息
    results["Basic_Info"] = {
        "generated_sequences": len(generated_seqs),
        "reference_sequences": len(reference_seqs),
        "avg_generated_length": np.mean([len(seq) for seq in generated_seqs]),
        "avg_reference_length": np.mean([len(seq) for seq in reference_seqs])
    }
    
    return results

def print_comprehensive_results(results):
    """
    打印综合评估结果
    """
    print("=" * 60)
    print("多肽序列综合评估结果")
    print("=" * 60)
    
    # 打印基本信息
    basic_info = results["Basic_Info"]
    print(f"生成序列数量: {basic_info['generated_sequences']}")
    print(f"参考序列数量: {basic_info['reference_sequences']}")
    print(f"生成序列平均长度: {basic_info['avg_generated_length']:.2f}")
    print(f"参考序列平均长度: {basic_info['avg_reference_length']:.2f}")
    print()
    
    # 打印质量评估指标
    print("质量评估指标:")
    #print(f"  困惑度(PPL): {results['PPL']:.4f}")
    print(f"  BLEU评分: {results['BLEU']:.4f}")
    print(f"  最大平均差异(MMD): {results['MMD']:.4f}")
    print(f"  Jaccard多样性: {results['Jaccard_Diversity']:.4f}")
    print(f"  香农熵: {results['Shannon_Entropy']:.4f}")
    print()
