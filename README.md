# 감정 키워드 기반 유튜브 영상 추천 시스템 - 모델링 가이드

> 이 문서는 **모델 개발 담당자**가 전처리 결과 파일만으로 바로 학습/추천 시스템 개발을 시작할 수 있도록 작성된 상세 가이드입니다.
> 전처리 과정(`preprocessing.ipynb`)을 직접 실행할 필요 없이, 생성된 4개 결과 파일을 활용하면 됩니다.

---

## 1. 프로젝트 개요

### 프로젝트 목표

사용자가 **감정 키워드**(예: "감동", "응원", "열정")를 입력하면, 유튜브 댓글 데이터를 기반으로 해당 감정과 가장 관련 높은 영상을 추천하는 시스템을 개발한다.

### 전체 시스템 흐름

```
데이터 수집 (유튜브 댓글 크롤링)
    ↓
전처리 (클리닝 → 토큰화 → 불용어 제거 → 필터링 → 영상 단위 집계)
    ↓
TF-IDF 벡터화 (영상별 댓글 토큰 → 수치 벡터)
    ↓
K-Means 클러스터링 (영상을 k개 그룹으로 분류)
    ↓
사용자 키워드 입력 → TF-IDF 벡터 변환 → 클러스터 예측
    ↓
해당 클러스터 내 코사인 유사도 상위 N개 추천
```

### 학습 담당자의 역할 범위

- **시작점**: 전처리 완료된 4개 결과 파일 (`preprocessed_comments.csv`, `video_documents.csv`, `tfidf_matrix.pkl`, `tfidf_vectorizer.pkl`)
- **담당 범위**: 결과 파일 로드 → 추천 로직 구현 → 모델 확장/개선
- **전처리 재실행 불필요**: 원본 엑셀 파일이나 `preprocessing.ipynb`를 다룰 필요 없음

---

## 2. 전처리 과정 요약

> 이 섹션은 전처리 파이프라인이 어떤 과정을 거쳤는지 이해하기 위한 요약이다. 직접 실행할 필요는 없다.

### 원본 데이터

4개 카테고리의 유튜브 댓글 엑셀 파일, 총 **9,710건**:

| 카테고리 | 댓글 수 | 영상 수 |
|---------|---------|---------|
| 뉴진스 | 2,500건 | 50개 |
| 월드컵 | 2,430건 | 49개 |
| 이스라엘 | 2,410건 | 50개 |
| 인공지능 | 2,370건 | 48개 |

### 전처리 단계

| 단계 | 설명 |
|------|------|
| 1. 텍스트 클리닝 | HTML 태그, URL, 이모지, 특수문자 제거 후 소문자 통일 |
| 2. 언어 감지 | 한글 문자 비율 기반으로 한국어(ko) / 영어(en) 분류 |
| 3. 토큰화 | 한국어: Okt 형태소 분석 + 어간 추출 / 영어: NLTK word_tokenize + Lemmatization |
| 4. 불용어 제거 | 한국어: 조사·접속사 등 커스텀 불용어 / 영어: NLTK 기본 불용어 |
| 5. 짧은 댓글 필터링 | 클린 토큰 2개 미만인 댓글 제거 |
| 6. 영상 단위 집계 | 같은 영상의 모든 댓글 토큰을 하나의 문서로 합침 |
| 7. TF-IDF 벡터화 | 영상 단위 문서를 `max_features=5000`으로 TF-IDF 변환 |

### 데이터 흐름 요약

```
원본 9,710건
    ↓ 중복 제거 (0건 제거)
    ↓ 짧은 댓글 필터링 (244건 제거)
전처리 후 9,466건
    ↓ 영상 단위 집계
197개 영상 문서
    ↓ TF-IDF 벡터화
(197, 5000) TF-IDF 행렬
```

### 한/영 이중 처리 방식

댓글의 한글 문자 비율이 50%를 초과하면 한국어, 이하면 영어로 분류한 후 각각 다른 토큰화 방식을 적용한다:

| 언어 | 토큰화 도구 | 어간 처리 | 불용어 |
|------|-----------|----------|--------|
| 한국어 (ko) | `konlpy.Okt.morphs(stem=True)` | Okt 내장 어간 추출 | 커스텀 리스트 (조사, 접속사 등) |
| 영어 (en) | `nltk.word_tokenize` | `WordNetLemmatizer` | `nltk.stopwords('english')` |

### 카테고리별 TF-IDF 상위 키워드

| 카테고리 | 상위 10개 키워드 |
|---------|-----------------|
| 뉴진스 | 뉴진스, newjeans, 노래, miss, new, song, never, jean, danielle, like |
| 월드컵 | 월드컵, 보다, 선수, 경기, 축구, 손흥민, 이강인, 감독, 좋다, 웃기다 |
| 이스라엘 | 전쟁, 이스라엘, 이란, 미국, 네타냐후, 트럼프, 보다, 나라, 국민, 악마 |
| 인공지능 | ai, 인간, 교수, 보다, 시대, 사람, 가다, 판사, 생각, 감사하다 |

---

## 3. 결과 파일 상세 설명

### 3-1. `preprocessed_comments.csv`

댓글 단위 전처리 결과. **9,466행 × 8컬럼**.

| 컬럼명 | 타입 | 설명 | 샘플값 |
|--------|------|------|--------|
| `Video Title` | str | 영상 제목 | `NewJeans (뉴진스) 'OMG' Official MV ...` |
| `Video URL` | str | 영상 URL | `https://www.youtube.com/watch?v=sVTy_wmn5SU...` |
| `Likes` | int64 | 영상 좋아요 수 (정수 변환 완료) | `2490682` |
| `Comment` | str | 댓글 원문 (클리닝 전) | `민희진님은 절대 ㅈㅅ 안한다고 했습니다...` |
| `category` | str | 카테고리명 (4종) | `뉴진스`, `월드컵`, `이스라엘`, `인공지능` |
| `Comment_clean` | str | 클리닝 완료 텍스트 | `민희진님은 절대 안하다고 했습니다...` |
| `lang` | str | 감지된 언어 | `ko` 또는 `en` |
| `tokens_clean` | str | 불용어 제거 완료 토큰 (**공백 구분 문자열**) | `민희진 절대 댓글 삭제 하이브 삭제 좋다 누르다` |

**주요 특이사항:**

- `tokens_clean`은 리스트가 아닌 **공백 구분 문자열**로 저장되어 있다. 리스트로 복원하려면:
  ```python
  # 문자열 → 리스트 복원
  df['tokens_list'] = df['tokens_clean'].str.split()
  # 예: "민희진 절대 댓글" → ['민희진', '절대', '댓글']
  ```

- `lang` 컬럼 분포: **ko 8,282건** / **en 1,184건**

- `Likes`는 영상 단위 값이므로 같은 영상의 댓글은 동일한 Likes 값을 가진다.

### 3-2. `video_documents.csv`

영상 단위 집계 결과. **197행 × 6컬럼**.

| 컬럼명 | 타입 | 설명 | 샘플값 |
|--------|------|------|--------|
| `Video URL` | str | 영상 URL (고유 식별자) | `https://www.youtube.com/watch?v=-72Dmj3oRd8...` |
| `Video Title` | str | 영상 제목 | `석학들이 머리 맞대고 만든 음식 월드컵` |
| `category` | str | 카테고리명 | `월드컵` |
| `Likes` | int64 | 영상 좋아요 수 | `9242` |
| `all_tokens` | str | 영상의 **모든 댓글 토큰을 합친** 문자열 | `오프닝 편집 까지 잡다 전문 시청 결국...` |
| `comment_count` | int64 | 해당 영상의 댓글 수 | `50` |

**주요 특이사항:**

- `all_tokens`는 해당 영상에 달린 모든 댓글의 클린 토큰을 공백으로 연결한 **하나의 문서**이다. TF-IDF 벡터화의 입력으로 사용되었다.
- `comment_count` 통계: 평균 48.1건, 최소 20건, 최대 50건
- 카테고리별 영상 수: 뉴진스 50, 이스라엘 50, 월드컵 49, 인공지능 48

### 3-3. `tfidf_matrix.pkl`

TF-IDF 벡터화 결과 행렬.

| 항목 | 값 |
|------|-----|
| 차원 | **(197, 5000)** — 197개 영상 × 5,000개 피처 |
| 형식 | `scipy.sparse.csr_matrix` (희소 행렬) |
| 파일 포맷 | Python pickle |

```python
import pickle

# 로드
with open('tfidf_matrix.pkl', 'rb') as f:
    tfidf_matrix = pickle.load(f)

print(type(tfidf_matrix))   # <class 'scipy.sparse._csr.csr_matrix'>
print(tfidf_matrix.shape)   # (197, 5000)

# 밀집 행렬로 변환 (필요 시)
dense_matrix = tfidf_matrix.toarray()  # numpy.ndarray (197, 5000)
```

### 3-4. `tfidf_vectorizer.pkl`

학습 완료된 TF-IDF Vectorizer 객체. **새로운 텍스트를 동일한 벡터 공간으로 변환**할 때 사용한다.

| 항목 | 값 |
|------|-----|
| 형식 | `sklearn.feature_extraction.text.TfidfVectorizer` |
| max_features | 5,000 |
| 학습 데이터 | 197개 영상의 `all_tokens` 문서 |

```python
import pickle

# 로드
with open('tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# 주요 메서드
vectorizer.transform(['키워드1 키워드2'])      # 새 텍스트 → TF-IDF 벡터 변환
vectorizer.get_feature_names_out()             # 5,000개 피처(단어) 목록 반환
```

---

## 4. 퀵 스타트 - 데이터 로드 코드

아래 코드를 복사하여 실행하면 4개 파일을 모두 로드하고 기본 정보를 확인할 수 있다.

```python
import pandas as pd
import pickle

# --- 1) 댓글 단위 전처리 결과 ---
comments_df = pd.read_csv('preprocessed_comments.csv', encoding='utf-8-sig')

# --- 2) 영상 단위 집계 결과 ---
video_df = pd.read_csv('video_documents.csv', encoding='utf-8-sig')

# --- 3) TF-IDF 행렬 ---
with open('tfidf_matrix.pkl', 'rb') as f:
    tfidf_matrix = pickle.load(f)

# --- 4) TF-IDF Vectorizer ---
with open('tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# === 로드 확인 ===
print('=== comments_df ===')
print(f'  Shape: {comments_df.shape}')
print(f'  Columns: {list(comments_df.columns)}')
print(comments_df.head(2))

print('\n=== video_df ===')
print(f'  Shape: {video_df.shape}')
print(f'  Columns: {list(video_df.columns)}')
print(video_df.head(2))

print(f'\n=== tfidf_matrix ===')
print(f'  Type: {type(tfidf_matrix)}')
print(f'  Shape: {tfidf_matrix.shape}')

print(f'\n=== vectorizer ===')
print(f'  Type: {type(vectorizer)}')
print(f'  Feature 수: {len(vectorizer.get_feature_names_out())}')
print(f'  Feature 샘플: {list(vectorizer.get_feature_names_out()[:10])}')
```

---

## 5. 활용 예시

### 예시 1: K-Means 모델 학습 (최적 k 결정)

TF-IDF 행렬을 입력으로 K-Means 클러스터링을 수행한다. **Elbow Method**와 **Silhouette Score**를 사용하여 최적의 클러스터 수(k)를 결정한다.

```python
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# 데이터 로드
with open('tfidf_matrix.pkl', 'rb') as f:
    tfidf_matrix = pickle.load(f)
video_df = pd.read_csv('video_documents.csv', encoding='utf-8-sig')

# --- Elbow Method + Silhouette Score ---
K_range = range(2, 11)
inertias = []
sil_scores = []

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(tfidf_matrix)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(tfidf_matrix, km.labels_))

# 시각화
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(K_range, inertias, 'bo-')
axes[0].set_xlabel('k (클러스터 수)')
axes[0].set_ylabel('Inertia')
axes[0].set_title('Elbow Method')

axes[1].plot(K_range, sil_scores, 'ro-')
axes[1].set_xlabel('k (클러스터 수)')
axes[1].set_ylabel('Silhouette Score')
axes[1].set_title('Silhouette Score')

plt.tight_layout()
plt.savefig('kmeans_optimal_k.png', dpi=150)
plt.show()

# 최적 k로 최종 모델 학습 (예: k=4)
best_k = 4
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
kmeans.fit(tfidf_matrix)

print(f'최적 k: {best_k}')
print(f'Silhouette Score: {silhouette_score(tfidf_matrix, kmeans.labels_):.4f}')
print(f'클러스터별 영상 수: {np.bincount(kmeans.labels_)}')
```

### 예시 2: PCA 2D 시각화로 클러스터 확인

고차원 TF-IDF 벡터를 PCA로 2차원 축소한 뒤, 클러스터 분포를 시각화한다.

```python
from sklearn.decomposition import PCA

# PCA로 2차원 축소
pca = PCA(n_components=2, random_state=42)
coords = pca.fit_transform(tfidf_matrix.toarray())

# 시각화
plt.figure(figsize=(10, 7))
scatter = plt.scatter(
    coords[:, 0], coords[:, 1],
    c=kmeans.labels_, cmap='tab10', alpha=0.7, edgecolors='k', linewidths=0.5
)
plt.colorbar(scatter, label='Cluster')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
plt.title('K-Means 클러스터 시각화 (PCA 2D)')

# 클러스터 중심점 표시
centers_2d = pca.transform(kmeans.cluster_centers_)
plt.scatter(
    centers_2d[:, 0], centers_2d[:, 1],
    c='red', marker='X', s=200, edgecolors='k', linewidths=1.5, label='중심점'
)
plt.legend()
plt.tight_layout()
plt.savefig('kmeans_pca_visualization.png', dpi=150)
plt.show()

# 각 클러스터별 상위 키워드 확인
with open('tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

feature_names = vectorizer.get_feature_names_out()
for i in range(best_k):
    center = kmeans.cluster_centers_[i]
    top_indices = center.argsort()[::-1][:10]
    top_words = [feature_names[idx] for idx in top_indices]
    print(f'클러스터 {i}: {", ".join(top_words)}')
```

### 예시 3: 키워드 기반 클러스터 추천

사용자 키워드를 TF-IDF 벡터로 변환한 뒤, `predict`로 클러스터를 할당하고 해당 클러스터 내에서 코사인 유사도 상위 N개를 추천한다.

```python
from sklearn.metrics.pairwise import cosine_similarity

def recommend_by_cluster(keywords, top_n=5):
    """K-Means 클러스터 기반 키워드 추천

    Args:
        keywords: 검색 키워드 문자열 (공백 구분)
        top_n: 추천 영상 수
    """
    # 키워드 → TF-IDF 벡터 변환
    query_vec = vectorizer.transform([keywords])

    # 클러스터 예측
    cluster_id = kmeans.predict(query_vec)[0]

    # 해당 클러스터에 속한 영상 인덱스 추출
    cluster_indices = np.where(kmeans.labels_ == cluster_id)[0]

    # 클러스터 내 코사인 유사도 계산
    cluster_matrix = tfidf_matrix[cluster_indices]
    similarities = cosine_similarity(query_vec, cluster_matrix).flatten()

    # 유사도 기준 정렬
    sorted_order = similarities.argsort()[::-1][:top_n]

    print(f'검색 키워드: "{keywords}"')
    print(f'할당된 클러스터: {cluster_id} (클러스터 내 영상 수: {len(cluster_indices)}개)\n')

    for rank, order_idx in enumerate(sorted_order, 1):
        video_idx = cluster_indices[order_idx]
        row = video_df.iloc[video_idx]
        print(f'{rank}. [{row["category"]}] {row["Video Title"]}')
        print(f'   유사도: {similarities[order_idx]:.4f} | 좋아요: {row["Likes"]:,} | 댓글: {row["comment_count"]}건')
        print(f'   URL: {row["Video URL"]}')
        print()

# 사용 예시
recommend_by_cluster('감동 응원 열정', top_n=5)
recommend_by_cluster('전쟁 평화 국제', top_n=3)
```

### 예시 4: 카테고리별 클러스터 분포 분석

클러스터와 카테고리 간의 관계를 **교차표(crosstab)**로 분석한다. 클러스터가 카테고리와 얼마나 일치하는지 확인할 수 있다.

```python
# 클러스터 레이블을 video_df에 추가
video_df['cluster'] = kmeans.labels_

# 교차표 생성
cross = pd.crosstab(video_df['category'], video_df['cluster'], margins=True)
print('=== 카테고리 × 클러스터 교차표 ===')
print(cross)

# 비율 교차표 (행 기준)
cross_pct = pd.crosstab(video_df['category'], video_df['cluster'], normalize='index')
print('\n=== 카테고리별 클러스터 비율 ===')
print(cross_pct.round(3))

# 시각화: 스택 바 차트
cross_pct.plot(kind='bar', stacked=True, figsize=(10, 6), colormap='tab10')
plt.title('카테고리별 클러스터 분포')
plt.xlabel('카테고리')
plt.ylabel('비율')
plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('category_cluster_distribution.png', dpi=150)
plt.show()
```

> **참고**: 클러스터 수(k)를 카테고리 수(4)와 동일하게 설정하면, 클러스터가 카테고리와 얼마나 자연스럽게 대응하는지 비교할 수 있다. k를 더 크게 설정하면 각 카테고리 내 세부 주제 그룹을 발견할 수도 있다.

---

## 6. 데이터 특성 및 주의사항

### 한/영 혼합 데이터

- 뉴진스 카테고리는 **영어 댓글 비율이 약 47%** (1,185건 중 영어 1,185건)로 매우 높다.
- 나머지 카테고리는 영어 비율이 1% 미만이다.
- 한영 혼용 댓글은 한글 비율 50% 기준으로 한 언어로만 처리되었으므로, 일부 정보 손실이 있을 수 있다.

### 카테고리별 데이터량

전처리 후 카테고리별 댓글 수는 소폭 차이가 있다:

| 카테고리 | 전처리 후 댓글 수 | 영상 수 |
|---------|-----------------|---------|
| 뉴진스 | 2,402건 | 50개 |
| 월드컵 | 2,391건 | 49개 |
| 이스라엘 | 2,340건 | 50개 |
| 인공지능 | 2,333건 | 48개 |

### Likes 0값 (30건)

좋아요 수가 표시되지 않은 영상 30건은 `Likes=0`으로 처리되었다. Likes 기반 가중치를 사용할 경우 이를 고려해야 한다.

### 토큰 저장 형식

`preprocessed_comments.csv`의 `tokens_clean`과 `video_documents.csv`의 `all_tokens`는 모두 **공백 구분 문자열**이다. 리스트가 아니므로 주의:

```python
# O — 올바른 사용
tokens = row['tokens_clean'].split()

# X — 잘못된 사용 (문자열 그대로 순회하면 글자 단위가 됨)
for char in row['tokens_clean']:  # 이렇게 하면 안 됨
    ...
```

### CSV 인코딩

모든 CSV 파일은 `utf-8-sig` 인코딩으로 저장되어 있다. 로드 시 반드시 인코딩을 지정해야 한다:

```python
pd.read_csv('파일명.csv', encoding='utf-8-sig')
```

### OOV(Out-of-Vocabulary) 문제

`tfidf_vectorizer.pkl`의 Vectorizer는 197개 영상 문서에서 학습되었다. **학습 시 없었던 새로운 단어는 `.transform()` 호출 시 자동으로 무시**된다. 따라서 너무 특수한 키워드를 입력하면 유사도가 0에 가까울 수 있다.

```python
# OOV 단어 확인 방법
vocab = set(vectorizer.get_feature_names_out())
user_words = '테스트 키워드 단어'.split()
oov = [w for w in user_words if w not in vocab]
print(f'OOV 단어: {oov}')
```

---

## 7. 필요 라이브러리 및 환경

### 설치 명령어

```bash
pip install pandas scikit-learn matplotlib konlpy nltk
```

### 패키지 목록

| 패키지 | 용도 | 비고 |
|--------|------|------|
| `pandas` | CSV 데이터 로드 및 처리 | 필수 |
| `scikit-learn` | TF-IDF 벡터 로드, K-Means 클러스터링, 코사인 유사도, PCA | 필수 |
| `matplotlib` | Elbow Method·Silhouette·PCA 시각화 | 필수 |
| `numpy` | 수치 연산 | scikit-learn 의존성으로 자동 설치 |
| `scipy` | 희소 행렬 (tfidf_matrix.pkl 로드) | scikit-learn 의존성으로 자동 설치 |
| `konlpy` | 한국어 형태소 분석 (Okt) | 전처리 재실행 시에만 필요 |
| `nltk` | 영어 토큰화/불용어 | 전처리 재실행 시에만 필요 |

> **참고**: 추천 시스템 개발만 수행하는 경우 `pandas`, `scikit-learn`, `matplotlib`만 있으면 된다. `konlpy`, `nltk`는 전처리 재실행 시에만 필요하다.

---

## 8. FAQ

### Q: `preprocessing.ipynb`를 다시 실행해야 하나요?

**아니요.** 결과 파일 4개(`preprocessed_comments.csv`, `video_documents.csv`, `tfidf_matrix.pkl`, `tfidf_vectorizer.pkl`)만 있으면 추천 시스템 개발이 가능합니다. 전처리 노트북은 참고용입니다.

### Q: 새 카테고리를 추가하려면?

`preprocessing.ipynb`의 셀 2에서 새 엑셀 파일을 `files` 딕셔너리에 추가한 후 전체를 재실행하면 됩니다. 4개 결과 파일이 새로 생성됩니다.

### Q: TF-IDF 대신 Word2Vec이나 BERT를 쓰고 싶다면?

`preprocessed_comments.csv`의 `tokens_clean` 컬럼을 활용하면 됩니다. 토큰화·불용어 제거가 완료된 상태이므로, 이 토큰들을 Word2Vec이나 임베딩 모델의 입력으로 바로 사용할 수 있습니다.

```python
# Word2Vec 예시
from gensim.models import Word2Vec

comments_df = pd.read_csv('preprocessed_comments.csv', encoding='utf-8-sig')
sentences = comments_df['tokens_clean'].str.split().tolist()
model = Word2Vec(sentences, vector_size=100, window=5, min_count=2, workers=4)
```

### Q: k값은 어떻게 정하나요?

**Elbow Method**와 **Silhouette Score** 두 가지를 함께 사용합니다.

- **Elbow Method**: k를 2부터 늘려가며 inertia(클러스터 내 거리 합)를 그래프로 그립니다. 기울기가 완만해지는 "팔꿈치" 지점이 적절한 k입니다.
- **Silhouette Score**: 각 데이터 포인트가 자기 클러스터에 얼마나 잘 속하는지 측정합니다(-1~1). 값이 클수록 클러스터 분리가 잘 된 것입니다.

구체적인 코드는 위 **예시 1**을 참고하세요. 본 프로젝트의 경우 카테고리가 4개이므로 k=4를 시작점으로 실험해 보는 것을 권장합니다.

### Q: 클러스터 결과를 어떻게 해석하나요?

각 클러스터 중심(centroid)에서 **TF-IDF 값이 높은 상위 키워드**를 추출하면 해당 클러스터의 주제를 파악할 수 있습니다.

```python
feature_names = vectorizer.get_feature_names_out()
for i in range(best_k):
    center = kmeans.cluster_centers_[i]
    top_indices = center.argsort()[::-1][:10]
    top_words = [feature_names[idx] for idx in top_indices]
    print(f'클러스터 {i}: {", ".join(top_words)}')
```

또한 위 **예시 4**의 교차표(crosstab)를 통해 클러스터와 실제 카테고리 간의 대응 관계를 확인할 수 있습니다.

### Q: `max_features=5000`을 바꾸려면?

`preprocessing.ipynb`의 셀 11에서 `TfidfVectorizer(max_features=5000)`의 값을 수정한 후 셀 11~12를 재실행하면 됩니다. `tfidf_matrix.pkl`과 `tfidf_vectorizer.pkl`이 새로 생성됩니다.

### Q: 유사도가 모두 0으로 나옵니다.

입력 키워드가 모두 OOV(학습 시 없었던 단어)일 가능성이 높습니다. 위 섹션 6의 OOV 확인 코드를 참고하여 vocabulary에 포함된 키워드를 사용해 보세요. `vectorizer.get_feature_names_out()`으로 사용 가능한 5,000개 단어 목록을 확인할 수 있습니다.
