import os
os.environ['OMP_NUM_THREADS'] = '1' # K-Means 메모리 누수 경고 방지용 (최상단 유지)

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib
import warnings
warnings.filterwarnings('ignore')

from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

# matplotlib 한글 폰트 설정
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False

# --- 1. 웹 페이지 기본 설정 ---
st.set_page_config(page_title="유튜브 영상 추천", page_icon="🎥", layout="centered")

# --- 2. 헬퍼 함수 (캐싱 밖에서 정의) ---
def get_weighted_vector(tokens_str, w2v_model, tfidf_vocab, tfidf_vec=None):
    """Word2Vec 벡터에 TF-IDF 가중치를 곱하여 가중 평균 벡터 반환"""
    words = str(tokens_str).split()
    valid_words = [word for word in words if word in w2v_model.wv]
    
    if not valid_words:
        return np.zeros(w2v_model.vector_size)
        
    vectors = []
    weights = []
    
    for word in valid_words:
        vectors.append(w2v_model.wv[word])
        weight = 1.0
        if tfidf_vec is not None and word in tfidf_vocab:
            word_idx = tfidf_vocab[word]
            weight = tfidf_vec[word_idx]
            if weight == 0: weight = 0.01 
        weights.append(weight)
        
    vectors = np.array(vectors)
    weights = np.array(weights).reshape(-1, 1)
    weighted_avg = np.sum(vectors * weights, axis=0) / np.sum(weights)
    return weighted_avg

# --- 3. 데이터 및 모델 로드 (캐싱 적용) ---
@st.cache_resource
def load_system():
    # 1. 데이터 및 기존 TF-IDF 로드
    comments_df = pd.read_csv('preprocessed_comments.csv', encoding='utf-8-sig')
    video_df = pd.read_csv('video_documents.csv', encoding='utf-8-sig')
    with open('tfidf_matrix.pkl', 'rb') as f: tfidf_matrix = pickle.load(f)
    with open('tfidf_vectorizer.pkl', 'rb') as f: vectorizer = pickle.load(f)
    
    tfidf_vocab = vectorizer.vocabulary_
    tfidf_dense = tfidf_matrix.toarray()
    
    # 2. Word2Vec 학습
    comments_df['tokens_list'] = comments_df['tokens_clean'].fillna('').apply(lambda x: x.split())
    w2v_model = Word2Vec(sentences=comments_df['tokens_list'], vector_size=100, window=5, min_count=2, workers=4)
    
    # 3. 197개 영상 벡터화 및 정규화
    video_vectors = []
    for i, row in video_df.iterrows():
        tokens = row['all_tokens']
        vec = get_weighted_vector(tokens, w2v_model, tfidf_vocab, tfidf_dense[i])
        video_vectors.append(vec)
    video_vectors = np.vstack(video_vectors)
    
    # 영점 조절 및 L2 정규화
    global_mean = np.mean(video_vectors, axis=0)
    video_vectors_centered = video_vectors - global_mean
    video_vectors_normalized = normalize(video_vectors_centered, norm='l2')
    
    # 4. K-Means 모델 학습 (정규화된 벡터 사용)
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    kmeans.fit(video_vectors_normalized)

    # 5. PCA 2D 좌표 사전 계산
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(video_vectors_normalized)

    return video_df, vectorizer, w2v_model, tfidf_vocab, global_mean, video_vectors_normalized, kmeans, pca, coords

# 최초 1회 로딩 실행
(video_df, vectorizer, w2v_model, tfidf_vocab, 
 global_mean, video_vectors_normalized, kmeans, pca, coords) = load_system()


# --- 4. 시각화 함수 ---
def plot_pca_with_query(coords, kmeans, pca, query_vec_normalized, cluster_id):
    fig, ax = plt.subplots(figsize=(8, 6))
    labels = kmeans.labels_
    colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B3']
    cluster_names = [f'클러스터 {i}' for i in range(kmeans.n_clusters)]

    # 클러스터별 데이터 포인트
    for i in range(kmeans.n_clusters):
        mask = labels == i
        alpha = 0.7 if i == cluster_id else 0.2
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=colors[i], label=cluster_names[i], alpha=alpha, s=20, edgecolors='none')

    # 클러스터 중심점
    centers_2d = pca.transform(kmeans.cluster_centers_)
    ax.scatter(centers_2d[:, 0], centers_2d[:, 1],
               c='red', marker='X', s=200, edgecolors='black', linewidths=1.5, zorder=5, label='클러스터 중심')

    # 사용자 쿼리 위치 (정규화된 벡터 사용)
    query_2d = pca.transform(query_vec_normalized)
    ax.scatter(query_2d[0, 0], query_2d[0, 1],
               c='gold', marker='*', s=400, edgecolors='black', linewidths=1.5, zorder=6, label='입력 키워드')

    ax.set_title('PCA 2D 클러스터 시각화 (사용자 쿼리 포함)', fontsize=13)
    ax.set_xlabel('PCA 1')
    ax.set_ylabel('PCA 2')
    ax.legend(loc='best', fontsize=9)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.caption(f"입력하신 키워드(★)가 **클러스터 {cluster_id}**에 할당되었습니다.")

def plot_category_distribution(video_df, kmeans, cluster_id):
    fig, ax = plt.subplots(figsize=(8, 5))
    df_with_cluster = video_df.copy()
    df_with_cluster['cluster'] = kmeans.labels_
    cross = pd.crosstab(df_with_cluster['category'], df_with_cluster['cluster'])

    colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B3']
    cross.plot(kind='bar', stacked=True, ax=ax, color=colors, edgecolor='none')

    ax.set_title('카테고리별 클러스터 분포 (Weighted W2V)', fontsize=13)
    ax.set_xlabel('카테고리')
    ax.set_ylabel('영상 수')
    ax.legend([f'클러스터 {i}' + (' ◀ 할당됨' if i == cluster_id else '') for i in range(kmeans.n_clusters)], loc='best', fontsize=9)
    ax.tick_params(axis='x', rotation=45)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# --- 5. 메인 UI 구성 ---
st.title("감정 기반 유튜브 영상 추천 (Advanced)")
st.write("사용자의 검색어에 담긴 '문맥과 감정'을 파악하여 가장 알맞은 영상을 찾아줍니다.")

user_input = st.text_input("검색어를 입력하세요 (예: 감동 응원 열정 / 전쟁 평화 국제)")
top_n = st.slider("추천받을 영상 개수", min_value=1, max_value=10, value=5)

# --- 6. 추천 실행 로직 ---
if st.button("추천 영상 찾기"):
    if user_input:
        with st.spinner("AI가 감정선과 문맥을 분석 중입니다..."):
            
            # 사용자 키워드 벡터화
            query_tfidf = vectorizer.transform([user_input]).toarray()[0]
            query_vec = get_weighted_vector(user_input, w2v_model, tfidf_vocab, query_tfidf).reshape(1, -1)
            
            if np.all(query_vec == 0):
                st.error(f"'{user_input}'에 포함된 단어들이 AI 학습 데이터에 존재하지 않습니다. 다른 유사한 단어로 검색해 보세요!")
            else:
                # 쿼리 벡터 정규화
                query_vec_centered = query_vec - global_mean
                query_vec_normalized = normalize(query_vec_centered, norm='l2')

                # 클러스터 예측 및 유사도 계산
                cluster_id = kmeans.predict(query_vec_normalized)[0]
                cluster_indices = np.where(kmeans.labels_ == cluster_id)[0]
                
                # 할당된 클러스터 내부의 정규화된 벡터들과 유사도 비교
                cluster_matrix = video_vectors_normalized[cluster_indices]
                similarities = cosine_similarity(query_vec_normalized, cluster_matrix).flatten()
                
                # 상위 N개 정렬
                sorted_order = similarities.argsort()[::-1][:top_n]

                st.success(f"분석 완료! (할당된 의미망: 클러스터 {cluster_id})")
                st.markdown("---")

                # === AI 분석 근거 섹션 ===
                with st.expander("AI 추천 근거 분석도 보기", expanded=True):
                    tab1, tab2 = st.tabs(["클러스터 시각화 (PCA)", "카테고리 혼합 분포"])
                    with tab1:
                        plot_pca_with_query(coords, kmeans, pca, query_vec_normalized, cluster_id)
                        st.caption("새로운 모델은 데이터가 완벽한 원형으로 펴지면서 다양성이 보존됩니다.")
                    with tab2:
                        plot_category_distribution(video_df, kmeans, cluster_id)
                        st.caption("단순 카테고리가 아닌, 감정(맥락)이 유사한 타 카테고리 영상들도 함께 묶여있는 것을 볼 수 있습니다.")

                st.markdown("---")

                # === 추천 영상 목록 ===
                for rank, order_idx in enumerate(sorted_order, 1):
                    video_idx = cluster_indices[order_idx]
                    row = video_df.iloc[video_idx]
                    sim_score = similarities[order_idx]

                    with st.container():
                        st.subheader(f"{rank}. [{row['category']}] {row['Video Title']}")
                        st.write(f"📊 **매칭 점수:** {sim_score:.4f} | ❤️ **좋아요:** {row['Likes']:,} | 💬 **댓글:** {row['comment_count']}건")
                        st.write(f"🔗 [영상 바로가기]({row['Video URL']})")
                        st.markdown("<br>", unsafe_allow_html=True)
    else:
        st.warning("검색어를 입력해 주세요.")