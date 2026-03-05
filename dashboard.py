import streamlit as st
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import plotly.graph_objects as go
import pickle

# 1. データベース接続
engine = create_engine('mysql+pymysql://root:my-secret-pw@localhost/flood_disaster_db')

st.set_page_config(page_title="鬼怒川洪水状況モニタリング", layout="wide")

# スタイル調整（警告色などを定義）
st.markdown("""
    <style>
    .reportview-container { background: #f0f2f6; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

st.title("🌊 鬼怒川洪水予測・意思決定支援システム")

# 2. データの取得と予測
@st.cache_data
def get_data_and_predict():
    query = "SELECT * FROM kinugawa_hydromet ORDER BY observation_datetime ASC"
    df = pd.read_sql(query, engine)
    
    # ラグ特徴量作成
    features = ['water_level_m', 'rain_local_mm', 'rain_upstream_mm']
    for i in range(1, 11):
        col = f'rain_upstream_lag_{i}'
        df[col] = df['rain_upstream_mm'].shift(i)
        features.append(col)
    
    with open('water_level_model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    df_pred = df.dropna(subset=features).copy()
    df_pred['predicted_water_level'] = model.predict(df_pred[features])
    df_pred['predicted_datetime'] = df_pred['observation_datetime'] + pd.Timedelta(hours=3)
    
    # 確信度の計算 (RMSE = 0.130m を 1σ と仮定し、95%信頼区間 2σ を設定)
    rmse = 0.130
    df_pred['upper_bound'] = df_pred['predicted_water_level'] + (rmse * 2)
    df_pred['lower_bound'] = df_pred['predicted_water_level'] - (rmse * 2)
    
    return df, df_pred

df, df_pred = get_data_and_predict()

# 3. 警告表示（ここが現場判断用）
st.subheader("🚨 3時間後の警戒ステータス")
latest_p = df_pred.iloc[-1]
p_val = latest_p['predicted_water_level']

# 判定ロジック
if p_val >= 5.0:
    st.error(f"【直ちに避難】3時間後の予測水位 ({p_val:.2f}m) が氾濫危険水位を超過します！")
elif p_val >= 4.0:
    st.warning(f"【避難準備】3時間後の予測水位 ({p_val:.2f}m) が避難判断水位に達する見込みです。")
else:
    st.success(f"【状況推移注意】3時間後の予測水位 ({p_val:.2f}m) は現在、基準値以下です。")

# 4. メトリクス表示
c1, c2, c3 = st.columns(3)
with c1:
    st.metric("現在（実測）の水位", f"{df.iloc[-1]['water_level_m']:.2f} m")
with c2:
    st.metric("3時間後の予測水位", f"{p_val:.2f} m", delta=f"{p_val - df.iloc[-1]['water_level_m']:.2f} m")
with c3:
    st.metric("予測の確信度 (95%)", "± 0.26 m")

# 5. メイングラフ（確信度バンド付き）
st.subheader("📈 予測シミュレーション（信頼区間表示）")

fig = go.Figure()

# 信頼区間（グレーの塗りつぶし）
fig.add_trace(go.Scatter(
    x=pd.concat([df_pred['predicted_datetime'], df_pred['predicted_datetime'][::-1]]),
    y=pd.concat([df_pred['upper_bound'], df_pred['lower_bound'][::-1]]),
    fill='toself',
    fillcolor='rgba(128, 128, 128, 0.2)',
    line=dict(color='rgba(255,255,255,0)'),
    hoverinfo="skip",
    showlegend=True,
    name='予測の不確実性 (±2σ)'
))

# 実測値
fig.add_trace(go.Scatter(x=df['observation_datetime'], y=df['water_level_m'], name='実測水位', line=dict(color='blue', width=2)))

# 予測値
fig.add_trace(go.Scatter(x=df_pred['predicted_datetime'], y=df_pred['predicted_water_level'], name='3時間後予測', line=dict(color='red', width=2, dash='dot')))

# 閾値線
fig.add_hline(y=4.0, line_dash="dash", line_color="orange", annotation_text="避難判断水位")
fig.add_hline(y=5.0, line_dash="dash", line_color="red", annotation_text="氾濫危険水位")

fig.update_layout(xaxis_title="日時", yaxis_title="水位 (m)", hovermode="x unified")
st.plotly_chart(fig, use_container_width=True)
