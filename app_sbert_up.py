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

            # 1) 검색어를 Sentence-BERT embedding으로 변환
            query_vec = sbert_model.encode(
                [user_input],
                convert_to_numpy=True,
                normalize_embeddings=True
            )

            # 2) 전체 영상과 cosine similarity 계산
            all_similarities = cosine_similarity(query_vec, sbert_matrix).flatten()

            # 3) 검색어와 유사한 상위 후보 영상 추출
            candidate_n = min(30, len(video_df))
            candidate_indices = all_similarities.argsort()[::-1][:candidate_n]

            # 4) 상위 후보들이 속한 cluster 확인
            candidate_clusters = kmeans.labels_[candidate_indices]

            # 5) 상위 cluster 여러 개 선택
            top_cluster_n = 3
            cluster_counts = np.bincount(candidate_clusters, minlength=kmeans.n_clusters)
            top_clusters = cluster_counts.argsort()[::-1][:top_cluster_n]

            # 6) 선택된 여러 cluster 내부 영상 추출
            cluster_indices = np.where(np.isin(kmeans.labels_, top_clusters))[0]

            # 7) 선택된 cluster들 내부에서 similarity 기준 ranking
            cluster_similarities = all_similarities[cluster_indices]
            sorted_order = cluster_similarities.argsort()[::-1][:top_n]

            # cluster별 대표 category 생성
            cluster_labels = {}

            for cluster_id in range(kmeans.n_clusters):

                cluster_data = video_df[kmeans.labels_ == cluster_id]

                if len(cluster_data) > 0:
                    대표카테고리 = cluster_data['category'].mode()[0]
                    cluster_labels[cluster_id] = 대표카테고리
                else:
                    cluster_labels[cluster_id] = f'Cluster {cluster_id}'

            selected_cluster_names = list(set([
                cluster_labels[c]
                for c in top_clusters
            ]))

            st.success(
                f"분석 완료! "
                f"(선택된 클러스터: {selected_cluster_names} / "
                f"후보 영상 수: {len(cluster_indices)}개)"
            )
            st.markdown("---")

            for rank, order_idx in enumerate(sorted_order, 1):
                video_idx = cluster_indices[order_idx]
                row = video_df.iloc[video_idx]
                sim_score = all_similarities[video_idx]

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