import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 1. 데이터 로드
df = pd.read_csv('uci-secom.csv')

# 2. 피처와 라벨 분리
X = df.drop(['Pass/Fail', 'Time'], axis=1)
y = df['Pass/Fail']

# 3. 학습/테스트 분리
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 4. 파이프라인 정의
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('classifier', RandomForestClassifier(random_state=42))
])

# 5. GridSearchCV용 하이퍼파라미터 설정
param_grid = {
    'classifier__n_estimators': [50, 100, 200],
    'classifier__max_depth': [None, 5, 10],
    'classifier__min_samples_split': [2, 5],
}

# 6. GridSearchCV 객체 생성
grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=3,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

# 7. 모델 학습
grid_search.fit(X_train, y_train)

# 8. 최적 파라미터 및 성능 확인
print("🔍 최적 파라미터:", grid_search.best_params_)
print(f"훈련 정확도: {grid_search.best_score_:.4f}")

# 9. 테스트 데이터에 예측
y_pred = grid_search.predict(X_test)
print("🔍 테스트 분류 리포트:")
print(classification_report(y_test, y_pred))
print(f"테스트 정확도: {accuracy_score(y_test, y_pred):.4f}")

# 10. 혼동 행렬 시각화
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['정상', '불량'], yticklabels=['정상', '불량'])
plt.xlabel("예측값")
plt.ylabel("실제값")
plt.title("혼동 행렬")
plt.show()

# 11. 중요 특성 시각화 (best_estimator에서 classifier 꺼냄)
best_model = grid_search.best_estimator_.named_steps['classifier']
importances = pd.Series(best_model.feature_importances_, index=X.columns)
top_features = importances.sort_values(ascending=False).head(20)
top_features.plot(kind='barh')
plt.title("중요 센서 TOP 20")
plt.gca().invert_yaxis()
plt.show()
