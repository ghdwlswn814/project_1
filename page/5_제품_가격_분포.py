
import streamlit as st
import pandas as pd
import plotly.express as px

# --------------------------------------------------
# 페이지 설정
# --------------------------------------------------
st.set_page_config(
    page_title="제품 유형별 가격 분포",
    page_icon="💰",
    layout="wide"
)

st.title("💰 제품 유형별 가격 분포")

st.write(
    "제품 유형별 가격의 분포를 비교하고, "
    "제품 유형에 따라 가격 차이가 어떻게 나타나는지 살펴봅니다."
)

# --------------------------------------------------
# 데이터 불러오기
# --------------------------------------------------
DATA_URL = (
    "https://raw.githubusercontent.com/"
    "LadyFelicia/Sephora-Analysis/main/Product%20Info.csv"
)


@st.cache_data
def load_data():
    return pd.read_csv(DATA_URL, encoding="utf-8")


try:
    df = load_data()

except Exception as e:
    st.error("데이터를 불러오는 중 오류가 발생했습니다.")
    st.code(str(e))
    st.stop()


# --------------------------------------------------
# 필요한 열 확인
# --------------------------------------------------
required_columns = ["price_usd", "category"]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    st.error(
        "데이터에 필요한 열이 없습니다: "
        + ", ".join(missing_columns)
    )
    st.stop()


# --------------------------------------------------
# 분석에 필요한 데이터 정리
# --------------------------------------------------
data = df[["price_usd", "category"]].copy()

# 가격을 숫자형으로 변환
data["price_usd"] = pd.to_numeric(
    data["price_usd"],
    errors="coerce"
)

# 제품 유형 앞뒤 공백 제거
data["category"] = data["category"].astype("string").str.strip()

# 가격 또는 제품 유형이 비어 있는 데이터 제외
data = data.dropna(subset=["price_usd", "category"])

# 빈 제품 유형 제외
data = data[data["category"] != ""].copy()

# 가격이 0 이하인 데이터 제외
data = data[data["price_usd"] > 0].copy()

if data.empty:
    st.warning("분석할 수 있는 가격 데이터가 없습니다.")
    st.stop()


# --------------------------------------------------
# 제품 유형별 제품 수 계산
# --------------------------------------------------
category_counts = (
    data.groupby("category")
    .size()
    .sort_values(ascending=False)
)

if category_counts.empty:
    st.warning("분석할 제품 유형이 없습니다.")
    st.stop()


# --------------------------------------------------
# 제품 유형 선택
# --------------------------------------------------
st.subheader("비교할 제품 유형 선택")

default_categories = category_counts.head(3).index.tolist()

selected_categories = st.multiselect(
    "비교할 제품 유형을 선택하세요.",
    options=category_counts.index.tolist(),
    default=default_categories
)

if not selected_categories:
    st.info("비교할 제품 유형을 하나 이상 선택해 주세요.")
    st.stop()


# --------------------------------------------------
# 선택한 제품 유형의 데이터
# --------------------------------------------------
plot_data = data[
    data["category"].isin(selected_categories)
].copy()


# --------------------------------------------------
# 상자그림(Box Plot)
# --------------------------------------------------
fig = px.box(
    plot_data,
    x="category",
    y="price_usd",
    color="category",
    points="all",
    hover_data={
        "category": False,
        "price_usd": ":.2f"
    },
    labels={
        "category": "제품 유형",
        "price_usd": "제품 가격(USD)",
        "color": "제품 유형"
    },
    title="제품 유형별 가격 분포"
)

fig.update_traces(
    marker=dict(
        size=4,
        opacity=0.55
    ),
    boxmean=False,
    hovertemplate=(
        "제품 유형: %{x}<br>"
        "제품 가격: $%{y:,.2f}"
        "<extra></extra>"
    )
)

fig.update_layout(
    height=650,
    showlegend=False,
    font=dict(size=13),
    xaxis=dict(
        title="제품 유형",
        tickangle=-35,
        automargin=True
    ),
    yaxis=dict(
        title="제품 가격(USD)",
        tickprefix="$",
        tickformat=",.2f",
        rangemode="tozero"
    ),
    margin=dict(
        l=90,
        r=40,
        t=80,
        b=150
    )
)

st.plotly_chart(fig, use_container_width=True)


# --------------------------------------------------
# 제품 유형별 가격 통계
# --------------------------------------------------
summary = (
    plot_data.groupby("category")["price_usd"]
    .agg(
        제품수="count",
        평균="mean",
        중앙값="median",
        최솟값="min",
        최댓값="max"
    )
    .reset_index()
)

summary["평균"] = summary["평균"].round(2)
summary["중앙값"] = summary["중앙값"].round(2)

summary = summary.rename(
    columns={
        "category": "제품 유형",
        "제품수": "분석 제품 수",
        "평균": "평균 가격(USD)",
        "중앙값": "중앙 가격(USD)",
        "최솟값": "최저 가격(USD)",
        "최댓값": "최고 가격(USD)"
    }
)

st.subheader("제품 유형별 가격 통계")

st.dataframe(
    summary,
    use_container_width=True,
    hide_index=True,
    column_config={
        "평균 가격(USD)": st.column_config.NumberColumn(
            format="$%.2f"
        ),
        "중앙 가격(USD)": st.column_config.NumberColumn(
            format="$%.2f"
        ),
        "최저 가격(USD)": st.column_config.NumberColumn(
            format="$%.2f"
        ),
        "최고 가격(USD)": st.column_config.NumberColumn(
            format="$%.2f"
        )
    }
)


# --------------------------------------------------
# 그래프 해석
# --------------------------------------------------
st.subheader("이 그래프로 알 수 있는 것")

st.write(
    "상자그림의 가운데 선은 제품 유형별 가격의 중앙값을 "
    "나타냅니다. 중앙값은 가격을 낮은 순서부터 높은 순서로 "
    "나열했을 때 가운데에 위치하는 값입니다."
)

st.write(
    "상자의 아래쪽과 위쪽은 각각 1사분위수와 3사분위수를 "
    "나타냅니다. 따라서 상자 안에는 가운데 50%의 제품이 "
    "포함됩니다. 상자와 수염의 길이를 비교하면 제품 유형별 "
    "가격의 분포 범위와 차이를 살펴볼 수 있습니다."
)

st.write(
    "그래프에 표시된 점은 개별 제품의 가격입니다. "
    "상자에서 멀리 떨어진 점은 이상치로 나타날 수 있지만, "
    "반드시 잘못된 가격 데이터라는 뜻은 아닙니다. "
    "실제로 다른 제품보다 가격이 높거나 낮은 제품일 수 있습니다."
)


# --------------------------------------------------
# 제품 유형별 중앙값 비교
# --------------------------------------------------
st.markdown("**제품 유형별 가격 중앙값 비교**")

median_summary = (
    plot_data.groupby("category")["price_usd"]
    .median()
    .sort_values(ascending=False)
)

highest_category = median_summary.index[0]
highest_value = median_summary.iloc[0]

lowest_category = median_summary.index[-1]
lowest_value = median_summary.iloc[-1]

st.write(
    f"선택한 제품 유형 중 중앙값이 가장 높은 유형은 "
    f"{highest_category}이며, 중앙 가격은 "
    f"${highest_value:,.2f}입니다."
)

st.write(
    f"중앙값이 가장 낮은 유형은 {lowest_category}이며, "
    f"중앙 가격은 ${lowest_value:,.2f}입니다."
)


# --------------------------------------------------
# 제품 유형별 가격 분포 차이
# --------------------------------------------------
st.markdown("**제품 유형별 가격 차이**")

spread_summary = (
    plot_data.groupby("category")["price_usd"]
    .agg(
        최소="min",
        최대="max",
        사분위범위=lambda x: x.quantile(0.75) - x.quantile(0.25)
    )
)

widest_category = spread_summary["사분위범위"].idxmax()
widest_value = spread_summary.loc[widest_category, "사분위범위"]

narrowest_category = spread_summary["사분위범위"].idxmin()
narrowest_value = spread_summary.loc[narrowest_category, "사분위범위"]

st.write(
    f"가운데 50% 제품의 가격 차이가 가장 큰 유형은 "
    f"{widest_category}입니다. 사분위범위는 "
    f"${widest_value:,.2f}입니다."
)

st.write(
    f"가운데 50% 제품의 가격 차이가 가장 작은 유형은 "
    f"{narrowest_category}입니다. 사분위범위는 "
    f"${narrowest_value:,.2f}입니다."
)


# --------------------------------------------------
# 분석의 의미와 주의점
# --------------------------------------------------
st.markdown("**분석 결과의 의미와 주의점**")

st.write(
    "제품 유형별 가격의 중앙값과 분포를 비교하면 "
    "어떤 유형의 제품이 상대적으로 가격이 높은 편인지, "
    "어떤 유형은 가격이 낮거나 제품마다 가격 차이가 큰지 "
    "파악할 수 있습니다."
)

st.write(
    "하지만 가격이 높다고 해서 제품의 효능이나 품질이 "
    "반드시 더 뛰어나다는 뜻은 아닙니다. 브랜드, 용량, "
    "제품 특성 등 여러 요인이 가격에 영향을 줄 수 있습니다."
)

st.write(
    "이번 분석은 제품 유형별 가격 특징을 이해하기 위한 "
    "탐색 분석입니다. 앞으로 만들 전성분 기반 제품 유형 "
    "분류 모델에는 가격을 입력 변수로 사용하지 않습니다. "
    "제품 유형을 전성분으로 예측한다는 프로젝트의 목적을 "
    "유지하기 위해서입니다."
)
