import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# --- 1. 웹 페이지 기본 설정 ---
st.set_page_config(page_title="유튜브 영상 추천", page_icon="🎥", layout="centered")

# --- 2. 데이터 및 모델 로드 ---
@st.cache_resource
def load_system():
    video_df = pd.read_csv('video_documents.csv', encoding='utf-8-sig')

    with open('sbert_matrix.pkl', 'rb') as f:
        sbert_matrix = pickle.load(f)

    sbert_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

    kmeans = KMeans(n_clusters=8, random_state=42, n_init=10)
    kmeans.fit(sbert_matrix)

    return video_df, sbert_matrix, sbert_model, kmeans

video_df, sbert_matrix, sbert_model, kmeans = load_system()

# --- 3. 메인 UI 구성 ---
st.title("감정 기반 유튜브 영상 추천")
st.write("키워드를 입력하면 Sentence-BERT와 K-Means 기반으로 가장 알맞은 영상을 추천합니다.")

user_input = st.text_input("검색어를 입력하세요 (예: 전쟁 평화 국제 / 뉴진스 노래 좋아 등)")
top_n = st.slider("추천받을 영상 개수", min_value=1, max_value=10, value=3)

# --- 4. 추천 실행 로직 ---
if st.button("추천 영상 찾기"):
    if user_input:
        with st.spinner("AI가 영상을 분석 중입니다..."):

            query_vec = sbert_model.encode(
                [user_input],
                convert_to_numpy=True,
                normalize_embeddings=True
            )

            cluster_id = kmeans.predict(query_vec)[0]
            cluster_indices = np.where(kmeans.labels_ == cluster_id)[0]

            cluster_matrix = sbert_matrix[cluster_indices]
            similarities = cosine_similarity(query_vec, cluster_matrix).flatten()

            sorted_order = similarities.argsort()[::-1][:top_n]

            st.success(f"분석 완료! (할당된 클러스터: {cluster_id} 그룹)")
            st.markdown("---")

            for rank, order_idx in enumerate(sorted_order, 1):
                video_idx = cluster_indices[order_idx]
                row = video_df.iloc[video_idx]
                sim_score = similarities[order_idx]

                with st.container():
                    st.subheader(f"{rank}. [{row['category']}] {row['Video Title']}")
                    st.write(
                        f"📊 **유사도:** {sim_score:.4f} | "
                        f"❤️ **좋아요:** {row['Likes']:,} | "
                        f"💬 **댓글:** {row['comment_count']}건"
                    )
                    st.write(f"🔗 [영상 보러가기]({row['Video URL']})")
                    st.markdown("<br>", unsafe_allow_html=True)
    else:
        st.warning("검색어를 입력해 주세요.")