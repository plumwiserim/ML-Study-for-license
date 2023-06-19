# 출력을 원하실 경우 print() 함수 활용
# 예시) print(df.head())

# getcwd(), chdir() 등 작업 폴더 설정 불필요
# 파일 경로 상 내부 드라이브 경로(C: 등) 접근 불가

# 데이터 파일 읽기 예제
import pandas as pd
X_test = pd.read_csv("data/X_test.csv")
X_train = pd.read_csv("data/X_train.csv")
y_train = pd.read_csv("data/y_train.csv")

# 사용자 코딩
X_train.set_index('cust_id', drop=True, inplace=True)
X_test.set_index('cust_id', drop=True, inplace=True)
y_train.set_index('cust_id', drop=True, inplace=True)

print(X_train.info())
print(X_train.head())
# print(X_train.isnull().sum())	# 환불금액 
# print(y_train.isnull().sum())	
print(X_test.isnull().sum())

# 결측치 처리 
X_train['환불금액'] = X_train['환불금액'].fillna(X_train['환불금액'].median())
X_test['환불금액'] = X_test['환불금액'].fillna(X_test['환불금액'].median())	# ??? 

# one-hot encoding 
# X_train = pd.get_dummies(X_train)
# print(X_train['주구매상품'].head())

# label encoding 
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
X_train['주구매상품'] = encoder.fit_transform(X_train['주구매상품'])
X_test['주구매상품'] = encoder.fit_transform(X_test['주구매상품'])
X_train['주구매지점'] = encoder.fit_transform(X_train['주구매지점'])
X_test['주구매지점'] = encoder.fit_transform(X_test['주구매지점'])

from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2)
print(X_train.shape)
print(X_valid.shape)
print(y_train.shape)
print(y_valid.shape)

# 분류 예측 모델 생성 1
import numpy as np 
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier()
rf_model.fit(X_train, np.ravel(y_train))
y_pred = rf_model.predict_proba(X_valid)

# print(y_train)
# print(np.ravel(y_train))

# 분류 예측 모델 생성 2
# from xgboost import XGBClassifier
# xgb_model = XGBClassifier()
# xgb_model.fit(X_train, y_train)

# 정확도 측정 
'''
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

accuracy = accuracy_score(y_test, y_pred)
print(accuracy)
'''

from sklearn.metrics import roc_auc_score
print(roc_auc_score(y_valid, y_pred[:, 1]))

# 하이퍼파라미터 튜닝 
# from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier()
rf_model.fit(X_train, np.ravel(y_train))
y_pred = rf_model.predict_proba(X_test)

# 답안 제출 
print(y_pred.shape)
answer = y_pred[:, 1]

# 답안 제출 참고
# 아래 코드 예측변수와 수험번호를 개인별로 변경하여 활용
# pd.DataFrame({'cust_id': X_test.cust_id, 'gender': pred}).to_csv('003000000.csv', index=False)
result = pd.DataFrame({
	'cust_id': X_test.index, 
	'gender': answer
})#.to_csv('수험번호.csv', index=False)

print(result)