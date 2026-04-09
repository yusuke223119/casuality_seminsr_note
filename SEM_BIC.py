import numpy as np
import pandas as pd
import statsmodels.api as sm
from itertools import combinations

# 1. データ生成 (DGP)
np.random.seed(42)
n = 10000
z2 = np.random.normal(0, 1, n)
z5 = np.random.normal(0, 1, n)
z6 = np.random.normal(0, 1, n)
z1 = 0.6 * z2 + np.random.normal(0, 1, n)
x = 0.6 * z1 + 0.6 * z5 + np.random.normal(0, 1, n)
z3 = 0.5 * x + np.random.normal(0, 1, n)

y = x + 0.6 * z2 + 0.6 * z3 + 0.6 * z6 + np.random.normal(0, 1, n)
z4 = 0.7 * x + 0.7 * y + np.random.normal(0, 1, n) # 合流点

df = pd.DataFrame({'X':x, 'Y':y, 'Z1':z1, 'Z2':z2, 'Z3':z3, 'Z4':z4, 'Z5':z5, 'Z6':z6})

# 2. BICによる全モデル探索
candidates = ['X', 'Z1', 'Z2', 'Z3', 'Z4', 'Z5', 'Z6']
results = []

for k in range(1, len(candidates) + 1):
    for combo in combinations(candidates, k):
        # モデルのフィッティング
        X_data = df[list(combo)]
        m = sm.OLS(df['Y'], X_data).fit()
        
        res = {'vars': combo, 'bic': m.bic}
        
        # 全ての回帰係数を辞書形式で追加
        res.update(m.params.to_dict())
        
        results.append(res)

# 結果をDataFrameに変換
res_df = pd.DataFrame(results)

# BICが最小のモデルを抽出
best_bic = res_df.loc[res_df['bic'].idxmin()]

print(f"BIC最善モデルの変数: {best_bic['vars']}")
print("【BIC最善モデルの全係数】")
coeffs = best_bic.drop(['vars', 'bic']).dropna()
print(coeffs)