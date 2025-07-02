import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# 한글 폰트 설정 (Windows용)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 1. CSV 파일 불러오기
df = pd.read_csv('uci-secom.csv')  # 파일 경로에 따라 수정

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
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# 5. 학습
pipeline.fit(X_train, y_train)

# 6. 예측
y_pred = pipeline.predict(X_test)

# 7. 평가
print("🔍 Classification Report:")
print(classification_report(y_test, y_pred))
print(f"정확도: {accuracy_score(y_test, y_pred):.4f}")

# 8. 혼동 행렬 시각화
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['정상', '불량'], yticklabels=['정상', '불량'])
plt.xlabel("예측값")
plt.ylabel("실제값")
plt.title("혼동 행렬")
plt.show()

# 9. 중요 센서 시각화 (feature_importances_는 pipeline 내부에서 꺼내야 함)
importances = pipeline.named_steps['classifier'].feature_importances_
feature_names = X.columns

top_features = pd.Series(importances, index=feature_names).sort_values(ascending=False).head(20)
top_features.plot(kind='barh')
plt.title("중요 센서 TOP 20")
plt.gca().invert_yaxis()
plt.show()
