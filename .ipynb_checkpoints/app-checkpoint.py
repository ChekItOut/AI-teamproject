import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

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
    
    return video_df, tfidf_matrix, vectorizer, kmeans

# 로딩 함수 실행 (처음 한 번만 실행됨)
video_df, tfidf_matrix, vectorizer, kmeans = load_system()

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