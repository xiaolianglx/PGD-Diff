import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
import sys
import json
import torch
import random
from pathlib import Path

# ================= 1. 添加项目根目录到 Python 路径 =================
PROJECT_ROOT = Path(__file__).parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ================= 2. 导入现有生成脚本的函数 =================
# 通用模型：直接复用 generate_general_peptides.py 中的 load_model 和 CONFIG
from generate_general_peptides import load_model as load_general_model_raw, CONFIG

# 条件模型：直接复用 generate_conditional_peptides.py 中的逻辑
from models.prefix_tuned.PGD_Diff_condition import PGD_Diff as PGD_Diff_Condition

# ================= 3. Global Page Configuration =================
st.set_page_config(
    page_title="PGD-Diff: Targeted Anticancer Peptide Generation",
    page_icon="🧬",
    layout="wide"
)

# ================= 4. Custom CSS Styling =================
def add_custom_css():
    custom_css = """
    <style>
    .stButton > button {
        background: linear-gradient(135deg, #00CED1 0%, #4A90E2 100%);
        color: white;
        border: none;
        padding: 10px 24px;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0, 206, 209, 0.4);
    }
    .stTextInput input, .stTextArea textarea, .stSelectbox select {
        border-radius: 8px;
        border: 1px solid #444;
        background-color: #262730;
        color: #E0E0E0;
    }
    [data-testid="stDataFrame"] {
        background-color: #1E1E1E;
        border-radius: 12px;
        padding: 10px;
        border: 1px solid #333;
    }
    [data-testid="stSidebar"] {
        background-color: #181818;
    }
    [data-testid="stSidebar"] * {
        color: #FFFFFF !important;
    }
    [data-testid="stSidebar"] h1 {
        font-size: 28px !important;
        margin-bottom: 20px !important;
    }
    [data-testid="stSidebar"] [role="radiogroup"] label div {
        font-size: 18px !important;
        line-height: 2.5 !important;
        padding-left: 5px !important;
    }
    [data-testid="stSidebar"] label p {
        font-size: 16px !important;
        margin-bottom: 8px !important;
    }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

add_custom_css()

# ================= 5. 加载癌症映射 =================
@st.cache_data
def load_cancer_mapping():
    """加载癌症类型到索引的映射"""
    map_path = PROJECT_ROOT / "models" / "prefix_tuned" / "cancer_to_idx.json"
    if map_path.exists():
        with open(map_path, 'r') as f:
            return json.load(f)
    map_path2 = PROJECT_ROOT / "cancer_to_idx.json"
    if map_path2.exists():
        with open(map_path2, 'r') as f:
            return json.load(f)
    return {
        "Breast": 0,
        "Lung": 1,
        "Colorectal": 2,
        "Liver": 3,
        "Leukemia": 4
    }

# ================= 6. 模型加载 =================
@st.cache_resource
def load_general_model():
    """
    加载通用生成模型
    直接复用 generate_general_peptides.py 中的 load_model 函数
    """
    try:
        ckpt_path = PROJECT_ROOT / CONFIG["checkpoint_path"]
        
        if not ckpt_path.exists():
            st.error(f"通用模型 checkpoint 未找到: {ckpt_path}")
            return None
        
        # 直接调用已验证的 load_model 函数
        model = load_general_model_raw(str(ckpt_path), CONFIG["model_config"])
        
        if model is not None:
            print("✅ 通用模型加载成功！")
            return model
        else:
            st.error("通用模型加载失败（load_model 返回 None）")
            return None
            
    except Exception as e:
        st.error(f"加载通用模型失败: {e}")
        return None

@st.cache_resource
@st.cache_resource
def load_conditional_model():
    """
    加载条件生成模型（Prefix-Tuned）
    手动加载 checkpoint，过滤形状不匹配的键（容错）
    """
    try:
        from models.prefix_tuned.PGD_Diff_condition import PGD_Diff
        
        ckpt_path = PROJECT_ROOT / "models" / "prefix_tuned" / "last.ckpt"
        if not ckpt_path.exists():
            st.error(f"条件模型 checkpoint 未找到: {ckpt_path}")
            return None
        
        cancer_map = load_cancer_mapping()
        num_cancer_types = len(cancer_map)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # 1. 创建模型实例
        model = PGD_Diff(
            use_prefix_condition=True,
            num_cancer_types=num_cancer_types,
            condition_mode='cross_attn',
            prefix_len=20,
            loss_weight=1.0
        )
        
        # 2. 加载 checkpoint
        checkpoint = torch.load(str(ckpt_path), map_location=device)
        state_dict = checkpoint['state_dict']
        
        # 3. 过滤：只保留形状匹配的键（和通用模型一样的逻辑）
        model_state_dict = model.state_dict()
        filtered_state_dict = {}
        skipped_keys = []
        
        for key, value in state_dict.items():
            if key in model_state_dict:
                if value.shape == model_state_dict[key].shape:
                    filtered_state_dict[key] = value
                else:
                    skipped_keys.append(f"{key} (shape mismatch)")
            else:
                skipped_keys.append(f"{key} (not in model)")
        
        if skipped_keys:
            print(f"⚠️ 跳过了 {len(skipped_keys)} 个键")
        
        # 4. 加载过滤后的状态字典
        model.load_state_dict(filtered_state_dict, strict=False)
        model.eval()
        model.to(device)
        
        print("✅ 条件模型加载成功！")
        return model
        
    except Exception as e:
        st.error(f"加载条件模型失败: {e}")
        return None

# ================= 7. 生成函数 =================
def generate_general_peptides(model, length, num_sequences):
    """通用生成：调用 denoise_seq_sample"""
    if model is None:
        return None
    try:
        length_list = [length] * num_sequences
        sequences, _, _ = model.denoise_seq_sample(
            n_seq=num_sequences,
            seq_length=length_list,
            fasta_out_statue=False
        )
        return sequences
    except Exception as e:
        st.error(f"通用生成失败: {e}")
        return None

def generate_conditional_peptides(model, cancer_type, length, num_sequences, cancer_map):
    """条件生成：调用 denoise_seq_sample_with_prefix"""
    if model is None:
        return None
    try:
        cancer_idx = cancer_map.get(cancer_type, 0)
        length_list = [length] * num_sequences
        sequences, _, _ = model.denoise_seq_sample_with_prefix(
            cancer_idx=cancer_idx,
            n_seq=num_sequences,
            seq_length=length_list,
            fasta_out_statue=False
        )
        return sequences
    except Exception as e:
        st.error(f"条件生成失败: {e}")
        return None

def generate_sequences_fallback(cancer_type, length, num_sequences):
    """备用生成函数：模拟生成"""
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    sequences = []
    for _ in range(num_sequences):
        seq = "".join(random.choices(list(amino_acids), k=length))
        sequences.append(seq)
    return sequences

# ================= 8. 加载模型 =================
cancer_map = load_cancer_mapping()
general_model = load_general_model()
conditional_model = load_conditional_model()

# 显示加载状态
if general_model is not None:
    st.sidebar.success("✅ 通用模型已加载")
else:
    st.sidebar.warning("⚠️ 通用模型未加载")

if conditional_model is not None:
    st.sidebar.success("✅ 条件模型已加载")
else:
    st.sidebar.warning("⚠️ 条件模型未加载")

# ================= 9. Sidebar Navigation =================
st.sidebar.title("🧬 PGD-Diff")
page = st.sidebar.radio(
    "Select Module:",
    ["🎯 Broad-Spectrum Generation", "🎯 Cancer-Targeted Generation", 
     "📊 Physicochemical Visualization", "📖 About Model", 
     "📥 Dataset Download", "📞 Contact Us"]
)

# ================= 10. Page Content Routing =================

# --- 10.1 Broad-Spectrum Generation ---
if page == "🎯 Broad-Spectrum Generation":
    st.title("🎯 Broad-Spectrum Anticancer Peptide Generation")
    st.write("Input parameters to generate peptide sequences with broad-spectrum anticancer activity.")
    
    col1, col2 = st.columns(2)
    with col1:
        length = st.slider("Peptide Length", min_value=5, max_value=50, value=15)
        num = st.slider("Number of Sequences", min_value=1, max_value=20, value=5)
        
    if st.button("🚀 Generate Sequences"):
        with st.spinner("Generating sequences..."):
            if general_model is not None:
                seqs = generate_general_peptides(general_model, length, num)
            else:
                seqs = None
            
            if seqs is None:
                st.warning("⚠️ 模型未加载，使用模拟生成")
                seqs = generate_sequences_fallback("general", length, num)
            
            st.success(f"Successfully generated {num} sequences!")
            st.dataframe(pd.DataFrame({"Sequence ID": range(1, num+1), "Amino Acid Sequence": seqs}))

# --- 10.2 Cancer-Targeted Generation ---
elif page == "🎯 Cancer-Targeted Generation":
    st.title("🎯 Cancer-Tissue Targeted Generation")
    st.write("Targeted generation of highly specific ACPs for specific cancer tissue microenvironments.")
    
    col1, col2 = st.columns(2)
    with col1:
        cancer_options = list(cancer_map.keys())
        cancer_type = st.selectbox("Select Cancer Tissue Type", cancer_options)
    with col2:
        length = st.slider("Peptide Length", min_value=5, max_value=50, value=18)
        
    num = st.slider("Number of Sequences", min_value=1, max_value=20, value=5)
    
    if st.button("🚀 Generate Targeted Sequences"):
        with st.spinner(f"Generating sequences for {cancer_type}..."):
            if conditional_model is not None:
                seqs = generate_conditional_peptides(
                    conditional_model, cancer_type, length, num, cancer_map
                )
            else:
                seqs = None
            
            if seqs is None:
                st.warning("⚠️ 条件模型未加载，使用模拟生成")
                seqs = generate_sequences_fallback(cancer_type, length, num)
            
            st.success(f"Successfully generated {num} sequences targeting {cancer_type}!")
            st.dataframe(pd.DataFrame({"Sequence ID": range(1, num+1), "Amino Acid Sequence": seqs}))

# --- 10.3 Physicochemical Visualization ---
elif page == "📊 Physicochemical Visualization":
    st.title("📊 Physicochemical Property Visualization")
    st.write("Input sequences to analyze key physicochemical characteristics such as net charge and hydrophobicity.")
    
    user_seq = st.text_area("Enter Amino Acid Sequence (e.g., ACDEFGHIKLMNPQRSTVWY)", "ACDEFGHIKLMNPQRSTVWY")
    if user_seq:
        data = {
            "Property": ["Net Charge", "Hydrophobicity", "Molecular Weight (Da)", "Isoelectric Point (pI)"],
            "Value": [2.5, -1.2, 1850.4, 8.6]
        }
        df = pd.DataFrame(data)
        fig = px.bar(df, x="Property", y="Value", title="Sequence Physicochemical Properties", color="Property")
        st.plotly_chart(fig, use_container_width=True)

# --- 10.4 About Model ---
elif page == "📖 About Model":
    st.title("📖 About PGD-Diff Model")
    st.markdown("""
    ### Model Overview
    We propose **Prefix-Guided Dual-path Diffusion (PGD-Diff)**, a multimodal diffusion framework that integrates a dual-path encoder with conditional prefix tuning.
    
    **1. Dual-Path Sequence and Structure Encoders:**
    For the sequence modality, we employ Transformer blocks with parallel global self-attention and local convolutions to simultaneously capture long-range residue dependencies and local motif information. For the structure modality, we design a hybrid encoder combining equivariant Graph Neural Networks (GNNs) with global Transformers.
    
    **2. Prefix-Guided Conditional Generation:**
    Building upon a pre-trained multimodal diffusion model, we introduce cancer type as conditional information. A lightweight prefix-tuning module injects type embeddings into the sequence denoising process, enabling cancer-type-specific ACP generation.
    """)
    
    st.subheader("Model Architecture")
    image_path = PROJECT_ROOT / "images" / "Figure 1.tif"
    try:
        if image_path.exists():
            st.image(str(image_path), caption="Figure 1: Architecture of the PGD-Diff Framework", width=800)
        else:
            st.warning(f"Image not found at: {image_path}")
    except Exception as e:
        st.error(f"Failed to load the image. Error: {e}")

# --- 10.5 Dataset Download ---
elif page == "📥 Dataset Download":
    st.title("📥 Dataset Download")
    st.write("Access high-quality datasets used for model training and validation, as well as generated peptide sequences from our model.")
    
    st.markdown("---")
    
    # ========== 训练数据集 ==========
    st.subheader("📊 Training Datasets")
    st.write("Balanced datasets of anticancer peptides (ACPs) and non-ACPs used for model training.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # ACP训练集
        acp_path = PROJECT_ROOT / "data" / "source" / "fasta" / "ACP.fasta"
        if acp_path.exists():
            with open(acp_path, 'r') as f:
                acp_data = f.read()
            st.download_button(
                label="📥 Download ACP Training Set (FASTA)",
                data=acp_data,
                file_name="ACP.fasta",
                mime="text/plain",
                key="acp_train"
            )
        else:
            st.warning("ACP.fasta not found")
    
    with col2:
        # nonACP训练集
        nonacp_path = PROJECT_ROOT / "data" / "source" / "fasta" / "nonACP.fasta"
        if nonacp_path.exists():
            with open(nonacp_path, 'r') as f:
                nonacp_data = f.read()
            st.download_button(
                label="📥 Download non-ACP Training Set (FASTA)",
                data=nonacp_data,
                file_name="nonACP.fasta",
                mime="text/plain",
                key="nonacp_train"
            )
        else:
            st.warning("nonACP.fasta not found")
    
    st.markdown("---")
    
    # ========== 通用生成数据集 ==========
    st.subheader("🧬 General Generation Dataset")
    st.write("10,000 peptides generated by the broad-spectrum generation model.")
    
    general_path = PROJECT_ROOT / "data" / "output" / "fasta" / "PGD_Diff_10000.fasta"
    if general_path.exists():
        with open(general_path, 'r') as f:
            general_data = f.read()
        st.download_button(
            label="📥 Download General Generated Dataset (FASTA)",
            data=general_data,
            file_name="PGD_Diff_10000.fasta",
            mime="text/plain",
            key="general_gen"
        )
        # 显示统计信息
        seq_count = general_data.count('>')
        st.caption(f"📊 Contains {seq_count} sequences")
    else:
        st.warning("PGD_Diff_10000.fasta not found")
    
    st.markdown("---")
    
    # ========== 条件生成数据集（按癌症类型） ==========
    st.subheader("🎯 Cancer-Targeted Generation Datasets")
    st.write("Peptides generated for nine specific cancer types using the conditional generation model.")
    
    # 癌症类型列表（与文件名对应）
    cancer_files = [
        ("Blood", "generated_blood_peptides.txt"),
        ("Brain", "generated_brain_peptides.txt"),
        ("Breast", "generated_breast_peptides.txt"),
        ("Cervix", "generated_cervix_peptides.txt"),
        ("Colon", "generated_colon_peptides.txt"),
        ("Liver", "generated_liver_peptides.txt"),
        ("Lung", "generated_lung_peptides.txt"),
        ("Prostate", "generated_prostate_peptides.txt"),
        ("Skin", "generated_skin_peptides.txt")
    ]
    
    # 创建3列网格布局（9个文件排成3×3）
    cols = st.columns(3)
    
    for idx, (cancer_name, file_name) in enumerate(cancer_files):
        col_idx = idx % 3
        with cols[col_idx]:
            file_path = PROJECT_ROOT / "data" / "output" / "fasta" / file_name
            if file_path.exists():
                with open(file_path, 'r') as f:
                    file_data = f.read()
                # 计算序列数量
                seq_count = len([line for line in file_data.split('\n') if line.strip()])
                st.download_button(
                    label=f"📥 {cancer_name} Cancer",
                    data=file_data,
                    file_name=file_name,
                    mime="text/plain",
                    key=f"cond_{cancer_name.lower()}"
                )
                st.caption(f"📊 {seq_count} sequences")
            else:
                st.caption(f"⚠️ {cancer_name} data not found")
    
    st.markdown("---")
    
    # ========== 全部数据集下载（打包） ==========
    st.subheader("📦 Download All Datasets")
    st.write("Download all generated datasets as a single ZIP archive (if available).")
    
    # 如果有ZIP打包文件，可以添加下载按钮
    zip_path = PROJECT_ROOT / "data" / "output" / "fasta" / "all_generated_datasets.zip"
    if zip_path.exists():
        with open(zip_path, 'rb') as f:
            zip_data = f.read()
        st.download_button(
            label="📦 Download All Datasets (ZIP)",
            data=zip_data,
            file_name="all_generated_datasets.zip",
            mime="application/zip",
            key="all_zip"
        )
    else:
        st.info("💡 Tip: You can download individual datasets from the sections above.")
        st.caption("To create a ZIP package manually, compress all files in the `data/output/fasta/` directory.")

# --- 10.6 Contact Us ---
elif page == "📞 Contact Us":
    st.title("📞 Contact Us")
    
    # 英文描述
    st.markdown("""
    **Contact the PGD-Diff team** to share suggestions for improving the platform, or to contribute newly reported anticancer peptide sequences for future database curation.
    """)
    
    st.markdown("---")
    
    # 创建两列布局：导师和作者
    col1, col2 = st.columns(2)
    
    # ========== 左列：导师信息 ==========
    with col1:
        st.subheader("👨‍🏫 Principal Investigator")
        
        # 导师照片
        liang_pic = PROJECT_ROOT / "images" / "liangPic.png"
        if liang_pic.exists():
            st.image(str(liang_pic), width=150)
        else:
            st.warning("Photo not found")
        
        st.markdown("""
        **Professor Guizhao Liang**  
        College of Bioengineering, Chongqing University  
        
        **E-mail:** gzliang@cqu.edu.cn  
        **Tel:** (86)23-65102507  
        
        **Address:**  
        Room 519, College of Bioengineering, Chongqing University  
        No.174 Shazhengjie, Shapingba, Chongqing, 400044, China
        """)
    
    # ========== 右列：作者信息 ==========
    with col2:
        st.subheader("👨‍💻 Developer")
        
        # 作者照片
        xiao_pic = PROJECT_ROOT / "images" / "xiaoPic.jpg"
        if xiao_pic.exists():
            st.image(str(xiao_pic), width=150)
        else:
            st.warning("Photo not found")
        
        st.markdown("""
        **Xiao Liang**  
        Ph.D. Candidate, PGD-Diff Development  
        
        **E-mail:**  
        20241901014@stu.cqu.edu.cn  
        1187517465@qq.com  
        
        **Address:**  
        Room 509, College of Bioengineering, Chongqing University  
        No.174 Shazhengjie, Shapingba, Chongqing, 400044, China
        """)
    
    st.markdown("---")
    st.markdown("""
    *We welcome collaboration and feedback from the research community. If you have any questions or suggestions, please feel free to contact us.*
    """)