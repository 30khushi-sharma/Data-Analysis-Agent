from marshal import load

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from dotenv import load_dotenv
load_dotenv()

# -------------------------------------------------------------
# 1. SETUP & CONFIGURATION
# -------------------------------------------------------------
st.set_page_config(page_title="Data Analysis Agent", layout="wide")
st.title(":rainbow[Data Analysis Agent]")

# File Upload
file = st.file_uploader(
    ":blue[Upload file in CSV or Excel Format]",
    type=["csv", "xlsx"]
)

# -------------------------------------------------------------
# 2. MAIN APP CONTENT (Runs only if file is uploaded)
# -------------------------------------------------------------
if file is not None:
    # Read file safely
    if file.name.endswith(".csv"):
        df = pd.read_csv(file)
    elif file.name.endswith(".xlsx"):
        df = pd.read_excel(file)
    else:
        st.error("Unsupported file format. Please upload a CSV or Excel file.")
        st.stop()

    # Preview
    st.subheader(":blue[Dataset Preview]")
    st.dataframe(df.head())

    # Dataset Info
    st.subheader(":blue[Dataset Information]")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Rows", df.shape[0])
    with col2:
        st.metric("Total Columns", df.shape[1])

    # Missing Values
    st.subheader(":blue[Missing Values]")
    missing = pd.DataFrame({
        "Column": df.columns,
        "Missing Values": df.isnull().sum().values
    })
    st.dataframe(missing)

    # Summary Statistics
    st.subheader(":blue[Summary Statistics]")
    st.dataframe(df.describe())

    # -------------------------------------------------------------
    # 3. EXPERT SEABORN VISUALIZATION GALLERY (AUTOMATED)
    # -------------------------------------------------------------
    st.subheader(":blue[📊 Automated Statistical Insights (Seaborn Gallery)]")
    
    # Set expert global Seaborn styling
    sns.set_theme(style="whitegrid")
    
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    
    # Beautiful palette lists to rotate through
    sns_palettes = ["crest", "flare", "magma", "viridis", "rocket", "mako"]
    
    grid_cols = st.columns(2)
    plot_idx = 0

    # A. Render Distribution Insights for Numeric Data
    for col in numeric_cols:
        with grid_cols[plot_idx % 2]:
            with st.container(border=True):
                fig, ax = plt.subplots(figsize=(7, 4.5))
                
                # Expert choice: Distplot with a Kernel Density Estimate curve
                sns.histplot(data=df, x=col, kde=True, ax=ax, 
                             color=sns.color_palette(sns_palettes[plot_idx % len(sns_palettes)])[3], 
                             edgecolor="white", alpha=0.85)
                
                ax.set_title(f"Statistical Distribution of {col}", fontsize=12, fontweight='bold', pad=15)
                sns.despine(left=True, bottom=True) # Remove box borders
                st.pyplot(fig)
                plt.close(fig)
        plot_idx += 1

    # B. Render Trend Insights if there are continuous row logs
    if len(numeric_cols) >= 1:
        with grid_cols[plot_idx % 2]:
            with st.container(border=True):
                fig, ax = plt.subplots(figsize=(7, 4.5))
                # Expert choice: Continuous Line Plot for structural shifts
                target_col = numeric_cols[0]
                sns.lineplot(data=df, x=df.index, y=target_col, ax=ax, color="#E76F51", linewidth=2)
                ax.set_title(f"Sequential Trend Line: {target_col}", fontsize=12, fontweight='bold', pad=15)
                sns.despine(left=True, bottom=True)
                st.pyplot(fig)
                plt.close(fig)
        plot_idx += 1

    # C. Render Frequency/Proportion Insights for Categorical Data
    for col in categorical_cols:
        # Avoid plotting if unique categories are completely messy/infinite (like unique IDs)
        if df[col].nunique() <= 15:
            with grid_cols[plot_idx % 2]:
                with st.container(border=True):
                    fig, ax = plt.subplots(figsize=(7, 4.5))
                    
                    # Expert choice: Ordered Categorical Countplot
                    order = df[col].value_counts().index
                    sns.countplot(data=df, y=col, ax=ax, order=order, 
                                  palette=sns_palettes[plot_idx % len(sns_palettes)])
                    
                    ax.set_title(f"Frequency Count Ranking: {col}", fontsize=12, fontweight='bold', pad=15)
                    sns.despine(left=True, bottom=True)
                    st.pyplot(fig)
                    plt.close(fig)
            plot_idx += 1

    # D. Full Correlation Matrix Heatmap (Expert Standard)
    if len(numeric_cols) > 1:
        st.markdown("#### **Inter-Column Relationship Analysis**")
        with st.container(border=True):
            fig, ax = plt.subplots(figsize=(10, 5))
            corr_matrix = df[numeric_cols].corr()
            
            # Expert choice: Diverging correlation plot with values annotated inside
            sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", 
                        vmin=-1, vmax=1, center=0, ax=ax, linewidths=0.5, square=False)
            
            ax.set_title("Features Correlation Matrix Heatmap", fontsize=13, fontweight='bold', pad=15)
            st.pyplot(fig)
            plt.close(fig)

    st.success("Expert Seaborn Dashboard Rendered!")
    
    
# -------------------------------------------------------------
# CHATBOT
# -------------------------------------------------------------

st.subheader(":blue[🤖 Ask Your Data Questions]")

user_query = st.text_input(
    "Type your question about the dataset here and press Enter:"
)

if user_query and file is not None:

    with st.spinner("Analyzing Dataset..."):

        try:

            # Load Model
            llm = HuggingFacePipeline.from_model_id(
                model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                task="text-generation",
                pipeline_kwargs={
                    "temperature": 0.2,
                    "max_new_tokens": 200,
                    "return_full_text": False
                }
            )

            model = ChatHuggingFace(llm=llm)

            # Dataset Context
            data_context = f"""
Dataset Shape:
{df.shape}

Columns:
{list(df.columns)}

Data Types:
{df.dtypes.to_string()}

Summary Statistics:
{df.describe(include='all').to_string()}

First 5 Rows:
{df.head().to_string()}
"""

            # Prompt Template
            prompt = ChatPromptTemplate.from_template(
                """
You are an expert Data Analyst.

Use ONLY the dataset information below.

Dataset:
{data}

Question:
{question}

Provide:
1. Direct Answer
2. Explanation
3. Insight

Keep response concise.
"""
            )

            # Parser
            parser = StrOutputParser()

            # Chain
            chain = prompt | model | parser

            result = chain.invoke(
                {
                    "data": data_context,
                    "question": user_query
                }
            )

            st.subheader("📊 Answer")
            st.write(result)

        except Exception as e:
            st.error(f"Error: {e}")

else:
    st.info(
        "Upload a dataset and ask questions about it."
    )