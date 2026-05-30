import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA

# matplotlib 한글 폰트 설정
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False

# --- 1. 웹 페이지 기본 설정 ---
st.set_page_config(page_title="유튜브 영상 추천", page_icon="🎥", layout="centered")

# --- 2. 데이터 및 모델 로드 (캐싱 적용) ---
# @st.cache_resource를 쓰면 버튼을 누를 때마다 데이터를 다시 읽는 걸 방지해서 속도가 엄청 빨라져!
@st.cache_resource
def load_system():
    # 데이터 로드
    video_df = pd.read_csv('video_documents.csv', encoding='utf-8-sig')

    with open('tfidf_matrix.pkl', 'rb') as f:
        tfidf_matrix = pickle.load(f)

    with open('tfidf_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)

    # K-Means 모델 학습 (최적 k=4)
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    kmeans.fit(tfidf_matrix)

    # PCA 2D 좌표 사전 계산 (매 검색마다 재계산 방지)
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(tfidf_matrix.toarray())

    return video_df, tfidf_matrix, vectorizer, kmeans, pca, coords

# 로딩 함수 실행 (처음 한 번만 실행됨)
video_df, tfidf_matrix, vectorizer, kmeans, pca, coords = load_system()


# --- 시각화 함수 ---
def plot_pca_with_query(coords, kmeans, pca, query_vec, cluster_id):
    """PCA 2D 클러스터 산점도에 사용자 쿼리 위치를 표시"""
    fig, ax = plt.subplots(figsize=(8, 6))

    labels = kmeans.labels_
    colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B3']
    cluster_names = [f'클러스터 {i}' for i in range(kmeans.n_clusters)]

    # 클러스터별 데이터 포인트 산점도
    for i in range(kmeans.n_clusters):
        mask = labels == i
        alpha = 0.7 if i == cluster_id else 0.2
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=colors[i], label=cluster_names[i],
                   alpha=alpha, s=20, edgecolors='none')

    # 클러스터 중심점 표시
    centers_2d = pca.transform(kmeans.cluster_centers_)
    ax.scatter(centers_2d[:, 0], centers_2d[:, 1],
               c='red', marker='X', s=200, edgecolors='black',
               linewidths=1.5, zorder=5, label='클러스터 중심')

    # 사용자 쿼리 위치 표시
    query_2d = pca.transform(query_vec.toarray())
    ax.scatter(query_2d[0, 0], query_2d[0, 1],
               c='gold', marker='*', s=400, edgecolors='black',
               linewidths=1.5, zorder=6, label='입력 키워드')

    ax.set_title('PCA 2D 클러스터 시각화 (사용자 쿼리 포함)', fontsize=13)
    ax.set_xlabel('PCA 1')
    ax.set_ylabel('PCA 2')
    ax.legend(loc='best', fontsize=9)
    fig.tight_layout()

    st.pyplot(fig)
    plt.close(fig)
    st.caption(f"입력하신 키워드(★)가 **클러스터 {cluster_id}**에 할당되었습니다. "
               f"해당 클러스터의 데이터 포인트가 진하게 표시됩니다.")


def plot_category_distribution(video_df, kmeans, cluster_id):
    """카테고리별 클러스터 분포 스택 바 차트"""
    fig, ax = plt.subplots(figsize=(8, 5))

    # 카테고리-클러스터 교차 테이블 생성
    df_with_cluster = video_df.copy()
    df_with_cluster['cluster'] = kmeans.labels_
    cross = pd.crosstab(df_with_cluster['category'], df_with_cluster['cluster'])

    # 스택 바 차트 그리기
    colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B3']
    cross.plot(kind='bar', stacked=True, ax=ax, color=colors, edgecolor='none')

    # 할당된 클러스터 하이라이트 표시
    ax.set_title('카테고리별 클러스터 분포', fontsize=13)
    ax.set_xlabel('카테고리')
    ax.set_ylabel('영상 수')
    ax.legend([f'클러스터 {i}' + (' ◀ 할당됨' if i == cluster_id else '')
               for i in range(kmeans.n_clusters)],
              loc='best', fontsize=9)
    ax.tick_params(axis='x', rotation=45)
    fig.tight_layout()

    st.pyplot(fig)
    plt.close(fig)

    # 할당 클러스터의 주요 카테고리 안내
    cluster_categories = df_with_cluster[df_with_cluster['cluster'] == cluster_id]['category']
    top_category = cluster_categories.value_counts().index[0] if len(cluster_categories) > 0 else '없음'
    st.caption(f"클러스터 {cluster_id}는 주로 **{top_category}** 관련 영상으로 구성되어 있습니다.")


# --- 3. 메인 UI 구성 ---
st.title("감정 기반 유튜브 영상 추천")
st.write("키워드를 입력하면 K-Means 모델이 가장 알맞은 영상을 찾아줍니다.")

# 텍스트 입력창
user_input = st.text_input("검색어를 입력하세요 (예: 전쟁 평화 국제 / 뉴진스 노래 좋아 등)")
top_n = st.slider("추천받을 영상 개수", min_value=1, max_value=10, value=3)

# --- 4. 추천 실행 로직 ---
if st.button("추천 영상 찾기"):
    if user_input:
        with st.spinner("AI가 영상을 분석 중입니다..."):
            # 키워드 벡터화 및 클러스터 예측
            query_vec = vectorizer.transform([user_input])
            cluster_id = kmeans.predict(query_vec)[0]
            cluster_indices = np.where(kmeans.labels_ == cluster_id)[0]

            # 코사인 유사도 계산
            cluster_matrix = tfidf_matrix[cluster_indices]
            similarities = cosine_similarity(query_vec, cluster_matrix).flatten()

            # 상위 N개 정렬
            sorted_order = similarities.argsort()[::-1][:top_n]

            # 결과 화면 출력
            st.success(f"분석 완료! (할당된 클러스터: {cluster_id} 그룹)")
            st.markdown("---")

            # === AI 분석 근거 섹션 ===
            with st.expander("AI 추천 근거 보기", expanded=True):
                tab1, tab2, tab3 = st.tabs(["클러스터 시각화", "카테고리 분포", "모델 선택 근거"])
                with tab1:
                    plot_pca_with_query(coords, kmeans, pca, query_vec, cluster_id)
                with tab2:
                    plot_category_distribution(video_df, kmeans, cluster_id)
                with tab3:
                    st.image("pictures/kmeans_optimal_k.png",
                             use_container_width=True)
                    st.caption("Elbow Method와 Silhouette Score로 최적 클러스터 수(K=4)를 결정한 근거입니다.")

            st.markdown("---")

            # === 추천 영상 목록 ===
            for rank, order_idx in enumerate(sorted_order, 1):
                video_idx = cluster_indices[order_idx]
                row = video_df.iloc[video_idx]
                sim_score = similarities[order_idx]

                # 영상별 결과를 예쁜 박스 형태로 출력
                with st.container():
                    st.subheader(f"{rank}. [{row['category']}] {row['Video Title']}")
                    st.write(f"📊 **유사도:** {sim_score:.4f} | ❤️ **좋아요:** {row['Likes']:,} | 💬 **댓글:** {row['comment_count']}건")
                    st.write(f"🔗 [영상 보러가기]({row['Video URL']})")
                    st.markdown("<br>", unsafe_allow_html=True)
    else:
        st.warning("검색어를 입력해 주세요.")
