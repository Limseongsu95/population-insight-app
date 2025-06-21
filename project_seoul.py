import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pymongo import MongoClient
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
from shapely.geometry import Point, shape
import folium
from folium.plugins import HeatMap
import json
from streamlit_folium import st_folium
from scipy.interpolate import make_interp_spline
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast
import torch
import streamlit.components.v1 as components
import io

# 한글 폰트 설정
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
st.set_page_config(page_title="서울 인구 인사이트 대시보드", layout="wide", initial_sidebar_state="expanded")

# Tailwind CSS for styling
components.html("""
<link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
<style>
    body { background-color: #f9fafb; }
    .stButton>button { background-color: #3b82f6; color: white; padding: 0.5rem 1rem; border-radius: 0.375rem; border: none; transition: background-color 0.3s; }
    .stButton>button:hover { background-color: #2563eb; }
    .card { background: white; padding: 1rem; border-radius: 0.375rem; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1); margin-bottom: 0.5rem; }
    .header { font-size: 2rem; font-weight: bold; color: #1f2937; text-align: center; margin-bottom: 0.5rem; }
    .subheader { font-size: 1.25rem; color: #374151; margin: 0.5rem 0; }
    .content { padding: 0.5rem; }
    .info-panel { background: #e0f2fe; padding: 1rem; border-radius: 0.375rem; margin-top: 1rem; }
</style>
""")

# MongoDB 연결
client = MongoClient("mongodb://localhost:27017/")
db = client["seoul_population_db"]

@st.cache_data
def load_geojson():
    with open("TL_SCCO_SIG.json", encoding="utf-8") as f:
        return json.load(f)

def get_region_name_from_coordinates(lat, lon, geojson):
    point = Point(lon, lat)
    for feature in geojson["features"]:
        polygon = shape(feature["geometry"])
        if polygon.contains(point):
            return feature["properties"]["SIG_KOR_NM"]
    return None

@st.cache_data
def load_population_data(region):
    male_data = list(db.population_male.find({"region": region}, {"_id": 0}))
    female_data = list(db.population_female.find({"region": region}, {"_id": 0}))
    
    if not male_data:
        male_df = pd.DataFrame(columns=["year", "population"])
    else:
        male_df = pd.DataFrame(male_data).sort_values("year")
        male_df['population'] = pd.to_numeric(male_df['population'], errors='coerce').fillna(0).astype(int)
        male_df = male_df.dropna(subset=['year'])

    if not female_data:
        female_df = pd.DataFrame(columns=["year", "population"])
    else:
        female_df = pd.DataFrame(female_data).sort_values("year")
        female_df['population'] = pd.to_numeric(female_df['population'], errors='coerce').fillna(0).astype(int)
        female_df = female_df.dropna(subset=['year'])
        
    return male_df, female_df

def predict_population(df, end_year=2040):
    if df.empty or len(df) < 5:
        return pd.DataFrame({"year": [], "population": []}), None, 0

    x = df["year"].values.reshape(-1, 1)
    y = df["population"].values
    
    poly = PolynomialFeatures(degree=2)
    x_poly = poly.fit_transform(x)
    model = LinearRegression().fit(x_poly, y)
    r2 = r2_score(y, model.predict(x_poly))
    
    future_years = np.arange(df["year"].max() + 1, end_year + 1).reshape(-1, 1)
    future_years_poly = poly.transform(future_years)
    preds = model.predict(future_years_poly)
    
    return pd.DataFrame({"year": future_years.flatten(), "population": preds.astype(int)}), model, r2

def smooth_curve(x, y):
    if len(x) < 2:
        return x, y
    x_new = np.linspace(x.min(), x.max(), 300)
    spl = make_interp_spline(x, y, k=3)
    return x_new, spl(x_new)

def draw_population_chart(male_df, female_df, male_pred, female_pred, region, male_r2, female_r2, start_year, end_year, chart_type):
    fig, ax = plt.subplots(figsize=(14, 6), facecolor="#f9fafb")
    fig.patch.set_facecolor("#f9fafb")

    male_df_filtered = male_df[(male_df["year"] >= start_year) & (male_df["year"] <= end_year)]
    female_df_filtered = female_df[(female_df["year"] >= start_year) & (female_df["year"] <= end_year)]
    male_pred_filtered = male_pred[(male_pred["year"] >= start_year) & (male_pred["year"] <= end_year)]
    female_pred_filtered = female_pred[(female_pred["year"] >= start_year) & (female_pred["year"] <= end_year)]

    if chart_type == "long_term":
        ax.plot(*smooth_curve(male_df_filtered["year"], male_df_filtered["population"]), label="남성 (실제)", color="#3b82f6", linewidth=2)
        ax.plot(*smooth_curve(female_df_filtered["year"], female_df_filtered["population"]), label="여성 (실제)", color="#ef4444", linewidth=2)
        if not male_pred_filtered.empty:
            ax.plot(*smooth_curve(male_pred_filtered["year"], male_pred_filtered["population"]), label="남성 (예측)", color="#3b82f6", linestyle="--", linewidth=2, alpha=0.7)
        if not female_pred_filtered.empty:
            ax.plot(*smooth_curve(female_pred_filtered["year"], female_pred_filtered["population"]), label="여성 (예측)", color="#ef4444", linestyle="--", linewidth=2, alpha=0.7)
        title = f"{region} 남녀 인구 변화 및 2040년 예측 (1995-2024)"
    else:  # short_term
        male_df_short = male_df[male_df["year"] <= 2025]
        female_df_short = female_df[female_df["year"] <= 2025]
        male_pred_short, _, male_r2_short = predict_population(male_df_short, 2040)
        female_pred_short, _, female_r2_short = predict_population(female_df_short, 2040)
        
        ax.plot(*smooth_curve(male_df_short["year"], male_df_short["population"]), label="남성 (실제 1995-2025)", color="#3b82f6", linewidth=2)
        ax.plot(*smooth_curve(female_df_short["year"], female_df_short["population"]), label="여성 (실제 1995-2025)", color="#ef4444", linewidth=2)
        if not male_pred_short.empty:
            ax.plot(*smooth_curve(male_pred_short["year"], male_pred_short["population"]), label="남성 (예측 2026-2040)", color="#3b82f6", linestyle="--", linewidth=2, alpha=0.7)
        if not female_pred_short.empty:
            ax.plot(*smooth_curve(female_pred_short["year"], female_pred_short["population"]), label="여성 (예측 2026-2040)", color="#ef4444", linestyle="--", linewidth=2, alpha=0.7)
        
        male_2025_actual = male_df[male_df["year"] == 2025]["population"].values[0] if 2025 in male_df["year"].values else None
        female_2025_actual = female_df[female_df["year"] == 2025]["population"].values[0] if 2025 in female_df["year"].values else None
        if male_2025_actual is not None:
            ax.plot(2025, male_2025_actual, "bo", label="남성 (실제 2025)", markersize=10)
        if female_2025_actual is not None:
            ax.plot(2025, female_2025_actual, "ro", label="여성 (실제 2025)", markersize=10)

        title = f"{region} 2026-2040년 인구 예측 (1995-2025 기반)"

    ax.set_title(title, fontsize=16, pad=10, color="#1f2937")
    ax.set_xlabel("연도", fontsize=10)
    ax.set_ylabel("인구 수 (명)", fontsize=10)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=9, loc="upper right", frameon=True, facecolor="white")
    ax.set_facecolor("#f9fafb")
    
    st.pyplot(fig)
    
    st.markdown(f"""
    <div class='card'><h3 class='subheader'>예측 정확도</h3>
    <p class='content'>2차 다항 회귀 모델. R²: 남성 {male_r2_short if chart_type == 'short_term' else male_r2:.2f}, 여성 {female_r2_short if chart_type == 'short_term' else female_r2:.2f} (0.7 이상 신뢰)</p></div>
    """, unsafe_allow_html=True)

def draw_total_population_chart(male_df, female_df, male_pred, female_pred, region):
    total_df = male_df.merge(female_df, on="year", suffixes=("_male", "_female"))
    total_df["total"] = total_df["population_male"] + total_df["population_female"]
    total_pred = pd.DataFrame({"year": male_pred["year"], "total": male_pred["population"] + female_pred["population"]}) if not male_pred.empty and not female_pred.empty else pd.DataFrame()

    fig, ax = plt.subplots(figsize=(14, 6), facecolor="#f9fafb")
    fig.patch.set_facecolor("#f9fafb")

    ax.plot(*smooth_curve(total_df["year"], total_df["total"]), label="총인구 (실제)", color="#10b981", linewidth=2)
    if not total_pred.empty:
        ax.plot(*smooth_curve(total_pred["year"], total_pred["total"]), label="총인구 (예측)", color="#10b981", linestyle="--", linewidth=2, alpha=0.7)

    ax.set_title(f"{region} 총인구 변화 및 예측", fontsize=16, pad=10, color="#1f2937")
    ax.set_xlabel("연도", fontsize=10)
    ax.set_ylabel("총 인구 수 (명)", fontsize=10)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=9, loc="upper right", frameon=True, facecolor="white")
    ax.set_facecolor("#f9fafb")
    
    st.pyplot(fig)

def draw_population_histogram(male_df, female_df, region):
    total_df = male_df.merge(female_df, on="year", suffixes=("_male", "_female"))
    total_df["total"] = total_df["population_male"] + total_df["population_female"]

    fig, ax = plt.subplots(figsize=(14, 6), facecolor="#f9fafb")
    ax.bar(total_df["year"], total_df["total"], color="#93c5fd", alpha=0.7)
    ax.set_title(f"{region} 연도별 총 인구 히스토그램", fontsize=16, pad=10, color="#1f2937")
    ax.set_xlabel("연도", fontsize=10)
    ax.set_ylabel("총 인구 수 (명)", fontsize=10)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_facecolor("#f9fafb")
    
    st.pyplot(fig)

def draw_growth_rate_chart(male_df, female_df, region):
    total_df = male_df.merge(female_df, on="year", suffixes=("_male", "_female"))
    total_df["total"] = total_df["population_male"] + total_df["population_female"]
    growth_rate = total_df["total"].pct_change() * 100
    growth_rate = growth_rate.dropna()

    fig, ax = plt.subplots(figsize=(14, 6), facecolor="#f9fafb")
    ax.plot(growth_rate.index, growth_rate, label="인구 성장률 (%)", color="#f59e0b", linewidth=2)
    ax.set_title(f"{region} 연도별 인구 성장률", fontsize=16, pad=10, color="#1f2937")
    ax.set_xlabel("연도", fontsize=10)
    ax.set_ylabel("성장률 (%)", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=9, loc="upper right", frameon=True, facecolor="white")
    ax.set_facecolor("#f9fafb")
    
    st.pyplot(fig)

def simulate_population(male_df, female_df, growth_rate=0.0):
    male_pred = predict_population(male_df)[0]
    female_pred = predict_population(female_df)[0]
    if not male_pred.empty and not female_pred.empty:
        male_pred["population"] = (male_pred["population"] * (1 + growth_rate)).astype(int)
        female_pred["population"] = (female_pred["population"] * (1 + growth_rate)).astype(int)
    return male_pred, female_pred

@st.cache_resource
def load_korean_model():
    tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2")
    model = GPT2LMHeadModel.from_pretrained("skt/kogpt2-base-v2")
    return tokenizer, model

def generate_korean_comment(region, male_df, female_df):
    if male_df.empty or female_df.empty:
        return "<div class='card'><p class='content'>💬 <strong>AI 코멘트:</strong> 데이터 부족으로 상세 분석 불가</p></div>"

    recent_m = int(male_df.tail(1)["population"].values[0])
    recent_f = int(female_df.tail(1)["population"].values[0])
    recent_year = int(male_df.tail(1)["year"].values[0])

    male_change_rate = ((male_df.tail(1)["population"].values[0] - male_df.iloc[-5]["population"]) / male_df.iloc[-5]["population"] * 100).round(2) if len(male_df) >= 5 else 0
    female_change_rate = ((female_df.tail(1)["population"].values[0] - female_df.iloc[-5]["population"]) / female_df.iloc[-5]["population"] * 100).round(2) if len(female_df) >= 5 else 0
    
    trend_m = "증가" if male_change_rate > 0 else "감소" if male_change_rate < 0 else "유지"
    trend_f = "증가" if female_change_rate > 0 else "감소" if female_change_rate < 0 else "유지"

    region_traits = {
        "강남구": "경제 중심지, 높은 부동산 가격과 기업 유입",
        "종로구": "역사적 중심지, 관광객과 고령 인구 비율 높음",
        "서초구": "법률/교육 중심, 중산층 및 전문직 밀집",
        "마포구": "문화/IT 산업 활성화, 젊은 층과 창작자 거주",
        "성북구": "주거 밀집 지역, 도시재생 뉴딜사업으로 주목"
    }.get(region, "다양한 특성을 가진 지역")

    prompt = (
        f"{recent_year}년 {region} 인구: 남성 {recent_m:,}명, 여성 {recent_f:,}명. "
        f"지난 5년간 남성 {trend_m}({male_change_rate}%), 여성 {trend_f}({female_change_rate}%). "
        f"{region}는 {region_traits}. 300-400자 내로:\n- 최근 인구 변화의 주요 원인(경제, 사회적 요인 등) 분석\n"
        f"- 변화가 지역에 미친 영향 간결히 설명\n- 인구 유입과 삶의 질 향상을 위한 3가지 구체적 정책 제안"
    )

    tokenizer, model = load_korean_model()
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    
    if torch.cuda.is_available():
        model.cuda()
        input_ids = input_ids.cuda()

    output = model.generate(
        input_ids, max_length=500, num_return_sequences=1, num_beams=5, temperature=0.7, top_k=50, top_p=0.95, no_repeat_ngram_size=2, early_stopping=True
    )
    
    if torch.cuda.is_available():
        model.cpu()
        torch.cuda.empty_cache()

    comment = tokenizer.decode(output[0], skip_special_tokens=True)[len(prompt):].strip()[:400]
    if len(comment) < 100:
        return "<div class='card'><p class='content'>💬 <strong>AI 코멘트:</strong> 데이터 부족으로 상세 분석 불가</p></div>"
    
    return f"<div class='card'><p class='content'>💬 <strong>AI 코멘트:</strong> {comment}</p></div>"

def download_data(male_df, female_df, male_pred, female_pred, region):
    combined_df = pd.concat([male_df.assign(gender="남성"), female_df.assign(gender="여성"),
                            male_pred.assign(gender="남성 (예측)"), female_pred.assign(gender="여성 (예측)")])
    combined_df = combined_df.rename(columns={"population": "인구 수"})
    output = io.BytesIO()
    combined_df.to_csv(output, index=False, encoding="utf-8-sig")
    return output

def get_region_stats(male_df, female_df):
    total_df = male_df.merge(female_df, on="year", suffixes=("_male", "_female"))
    total_df["total"] = total_df["population_male"] + total_df["population_female"]
    return {
        "최대 인구": int(total_df["total"].max()) if not total_df.empty else 0,
        "최소 인구": int(total_df["total"].min()) if not total_df.empty else 0,
        "평균 인구": int(total_df["total"].mean()) if not total_df.empty else 0
    }

def compare_regions(selected_region, other_region):
    male_df1, female_df1 = load_population_data(selected_region)
    male_df2, female_df2 = load_population_data(other_region)
    total_df1 = male_df1.merge(female_df1, on="year", suffixes=("_male", "_female"))
    total_df2 = male_df2.merge(female_df2, on="year", suffixes=("_male", "_female"))
    total1 = total_df1["population_male"].sum() + total_df1["population_female"].sum() if not total_df1.empty else 0
    total2 = total_df2["population_male"].sum() + total_df2["population_female"].sum() if not total_df2.empty else 0
    return {selected_region: total1, other_region: total2}

def main():
    st.markdown("<h1 class='header'>🌆 서울 인구 인사이트</h1>", unsafe_allow_html=True)
    st.markdown("<p class='content text-gray-600 text-center'>구 클릭으로 트렌드와 분석 확인!</p>", unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("<h2 class='subheader'>설정</h2>", unsafe_allow_html=True)
        if st.button("새로고침", key="refresh"):
            st.cache_data.clear()
            st.experimental_rerun()
        if st.button("지도 초기화", key="reset_map"):
            st.session_state["map_reset"] = True
        if st.button("데이터 갱신", key="update_data"):
            st.cache_data.clear()
            st.experimental_rerun()
        if st.button("구별 인구 비교", key="compare_regions"):
            st.session_state["compare_mode"] = True
        if st.button("데이터 필터링", key="filter_data"):
            st.session_state["filter_mode"] = True

    geojson = load_geojson()
    m = folium.Map(location=[37.5665, 126.9780], zoom_start=11, tiles="cartodbpositron", zoom_control=True)
    # 인구 밀도 히트맵 데이터 준비 (int로 변환)
    heat_data = []
    for feature in geojson["features"]:
        region = feature["properties"]["SIG_KOR_NM"]
        male_df, female_df = load_population_data(region)
        total_df = male_df.merge(female_df, on="year", suffixes=("_male", "_female"))
        total_pop = int(total_df["population_male"].sum() + total_df["population_female"].sum()) if not total_df.empty else 0
        centroid = shape(feature["geometry"]).centroid
        heat_data.append([centroid.y, centroid.x, total_pop])
    HeatMap(heat_data, radius=15).add_to(m)
    folium.GeoJson(geojson, name="구", tooltip=folium.GeoJsonTooltip(fields=["SIG_KOR_NM"], aliases=["구 이름"]),
                   style_function=lambda x: {'fillColor': '#93c5fd', 'color': '#1f2937', 'weight': 1, 'fillOpacity': 0.4},
                   highlight_function=lambda x: {'fillColor': '#3b82f6', 'color': '#1f2937', 'weight': 2}).add_to(m)
    m.add_child(folium.LatLngPopup())

    col1, col2 = st.columns([2.5, 1.5])

    with col1:
        st.markdown("<div class='card'><h2 class='subheader'>서울시 지도 (인구 밀도 히트맵)</h2></div>", unsafe_allow_html=True)
        if "map_reset" in st.session_state:
            m.location = [37.5665, 126.9780]
            m.zoom_start = 11
            del st.session_state["map_reset"]
        st_map = st_folium(m, width=600, height=600, key="seoul_map")

    with col2:
        st.markdown("<div class='card'><h2 class='subheader'>지역 정보</h2></div>", unsafe_allow_html=True)
        selected_region = st.empty()
        
        if st_map and st_map.get("last_clicked"):
            lat, lon = st_map["last_clicked"]["lat"], st_map["last_clicked"]["lng"]
            region = get_region_name_from_coordinates(lat, lon, geojson)
            
            if region:
                selected_region.markdown(f"<div class='card'><p class='content text-green-600'>✅ {region}</p></div>", unsafe_allow_html=True)
                
                male_df, female_df = load_population_data(region)
                
                if male_df.empty or female_df.empty or len(male_df) < 5 or len(female_df) < 5:
                    st.markdown("<div class='card'><p class='content text-yellow-600'>⚠️ 5년치 데이터 부족</p></div>", unsafe_allow_html=True)
                else:
                    years = list(male_df["year"].dropna().unique())
                    start_year, end_year = st.slider("연도 범위 선택", min(years), max(years), (1995, max(years)))

                    st.markdown("<div class='card'><h3 class='subheader'>2025년 데이터</h3></div>", unsafe_allow_html=True)
                    male_2025 = male_df[male_df["year"] == 2025]
                    female_2025 = female_df[female_df["year"] == 2025]
                    if not male_2025.empty and not female_2025.empty:
                        st.table({"구": [region], "남성": [f"{int(male_2025['population'].values[0]):,}"],
                                "여성": [f"{int(female_2025['population'].values[0]):,}"],
                                "총계": [f"{int(male_2025['population'].values[0] + female_2025['population'].values[0]):,}"]})
                    else:
                        st.markdown("<p class='content text-gray-600'>2025년 데이터 없음</p>", unsafe_allow_html=True)
                    
                    # 1995-2025 데이터로 2040년 예측
                    male_pred_long, male_model_long, male_r2_long = predict_population(male_df[male_df["year"] <= 2025])
                    female_pred_long, female_model_long, female_r2_long = predict_population(female_df[female_df["year"] <= 2025])
                    # 1995-2025 데이터로 2040년 예측 (short_term도 동일 범위로 확장)
                    male_pred_short, _, male_r2_short = predict_population(male_df[male_df["year"] <= 2025], 2040)
                    female_pred_short, _, female_r2_short = predict_population(female_df[female_df["year"] <= 2025], 2040)

                    chart_type = st.selectbox("그래프 선택", ["long_term", "short_term"], index=0)
                    st.markdown(f"<div class='card'><h3 class='subheader'>{region} 인구 분석</h3></div>", unsafe_allow_html=True)
                    draw_population_chart(male_df, female_df, male_pred_long if chart_type == "long_term" else male_pred_short,
                                        female_pred_long if chart_type == "long_term" else female_pred_short,
                                        region, male_r2_long if chart_type == "long_term" else male_r2_short,
                                        female_r2_long if chart_type == "long_term" else female_r2_short, 1995, end_year, chart_type)
                    
                    csv = download_data(male_df, female_df, male_pred_long if chart_type == "long_term" else male_pred_short,
                                      female_pred_long if chart_type == "long_term" else female_pred_short, region)
                    st.download_button(label="데이터 다운로드 (CSV)", data=csv.getvalue(), file_name=f"{region}_population.csv", mime="text/csv")

                    st.markdown(generate_korean_comment(region, male_df, female_df), unsafe_allow_html=True)
                    
                    st.markdown(f"<div class='card'><h3 class='subheader'>{region} 총인구 예측</h3></div>", unsafe_allow_html=True)
                    growth_rate = st.slider("성장률 조정 (%)", -5.0, 5.0, 0.0, 0.1, key="growth_slider")
                    male_pred_sim, female_pred_sim = simulate_population(male_df[male_df["year"] <= 2025], female_df[female_df["year"] <= 2025], growth_rate / 100)
                    draw_total_population_chart(male_df, female_df, male_pred_sim, female_pred_sim, region)

                    # 추가 UI: 지역 통계 요약
                    stats = get_region_stats(male_df, female_df)
                    st.markdown(f"<div class='card'><h3 class='subheader'>{region} 인구 통계</h3></div>", unsafe_allow_html=True)
                    st.table({"지표": ["최대 인구", "최소 인구", "평균 인구"], "값": [f"{stats['최대 인구']:,}", f"{stats['최소 인구']:,}", f"{stats['평균 인구']:,}"]})

                    # 추가 UI: 인구 변화 히스토그램
                    st.markdown(f"<div class='card'><h3 class='subheader'>{region} 인구 변화 히스토그램</h3></div>", unsafe_allow_html=True)
                    draw_population_histogram(male_df, female_df, region)

                    # 추가 UI: 인구 성장률 그래프
                    st.markdown(f"<div class='card'><h3 class='subheader'>{region} 인구 성장률</h3></div>", unsafe_allow_html=True)
                    draw_growth_rate_chart(male_df, female_df, region)

                    # 추가 UI: 정보 패널
                    info = f"{region}은(는) 서울의 대표적인 구로, 다양한 문화와 경제 활동이 활발합니다. 최근 인구 변화는 지역 특성에 따라 달라질 수 있습니다."
                    st.markdown(f"<div class='info-panel'><p class='content'>{info}</p></div>", unsafe_allow_html=True)

                    # 구별 인구 비교 모드
                    if "compare_mode" in st.session_state:
                        other_region = st.selectbox("비교할 구 선택", [f["properties"]["SIG_KOR_NM"] for f in geojson["features"] if f["properties"]["SIG_KOR_NM"] != region], key="compare_select")
                        comparison = compare_regions(region, other_region)
                        st.markdown(f"<div class='card'><h3 class='subheader'>{region} vs {other_region} 인구 비교</h3></div>", unsafe_allow_html=True)
                        st.table({"구": [region, other_region], "총 인구": [f"{comparison[region]:,}", f"{comparison[other_region]:,}"]})
                        del st.session_state["compare_mode"]

                    # 데이터 필터링 모드
                    if "filter_mode" in st.session_state:
                        filter_years = st.slider("필터링 연도 범위", min(years), max(years), (1995, max(years)), key="filter_slider")
                        male_df_filtered = male_df[(male_df["year"] >= filter_years[0]) & (male_df["year"] <= filter_years[1])]
                        female_df_filtered = female_df[(female_df["year"] >= filter_years[0]) & (female_df["year"] <= filter_years[1])]
                        st.markdown(f"<div class='card'><h3 class='subheader'>{region} 필터링된 인구 데이터</h3></div>", unsafe_allow_html=True)
                        st.line_chart(male_df_filtered.set_index("year")["population"])
                        st.line_chart(female_df_filtered.set_index("year")["population"])
                        del st.session_state["filter_mode"]
            else:
                selected_region.markdown("<div class='card'><p class='content text-blue-600'>ℹ️ 구를 클릭</p></div>", unsafe_allow_html=True)
        else:
            selected_region.markdown("<div class='card'><p class='content text-blue-600'>ℹ️ 지도에서 구 선택</p></div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()