
import streamlit as st
import pandas as pd
import plotly.express as px

# --------------------------------------------------
# 페이지 설정
# --------------------------------------------------
st.set_page_config(
    page_title="제품 유형별 성분 개수 분포",
    page_icon="🧴",
    layout="wide"
)

st.title("🧴 제품 유형별 성분 개수 분포")

st.write(
    "제품 하나에 포함된 서로 다른 성분의 개수를 계산하고, "
    "제품 유형별로 성분 개수의 분포가 어떻게 다른지 비교합니다."
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
required_columns = ["ingredients", "category"]

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
data = df[["ingredients", "category"]].copy()

# 전성분이나 제품 유형이 비어 있는 행 제외
data = data.dropna(subset=["ingredients", "category"])

# 문자열 앞뒤 공백 제거
data["ingredients"] = data["ingredients"].astype(str).str.strip()
data["category"] = data["category"].astype(str).str.strip()

# 빈 문자열 제외
data = data[
    (data["ingredients"] != "")
    & (data["category"] != "")
].copy()

if data.empty:
    st.warning("분석할 수 있는 데이터가 없습니다.")
    st.stop()


# --------------------------------------------------
# 성분 분리 및 정규화
# --------------------------------------------------
def split_ingredients(ingredient_text):
    """
    쉼표를 기준으로 성분을 분리합니다.
    앞뒤 공백을 제거하고 대소문자를 통일합니다.
    한 제품 안의 중복 성분은 한 번만 계산합니다.
    """
    ingredients = set()

    for ingredient in str(ingredient_text).split(","):
        ingredient = ingredient.strip().casefold()

        if ingredient:
            ingredients.add(ingredient)

    return sorted(ingredients)


data["성분목록"] = data["ingredients"].apply(split_ingredients)

# 성분이 하나도 남지 않은 제품 제외
data = data[data["성분목록"].map(len) > 0].copy()

if data.empty:
    st.warning("분석할 수 있는 전성분 데이터가 없습니다.")
    st.stop()


# --------------------------------------------------
# 제품별 성분 개수 계산
# --------------------------------------------------
data["성분개수"] = data["성분목록"].apply(len)


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
    y="성분개수",
    color="category",
    points="all",
    hover_data={
        "category": False,
        "성분개수": True
    },
    labels={
        "category": "제품 유형",
        "성분개수": "제품 하나의 성분 개수",
        "color": "제품 유형"
    },
    title="제품 유형별 성분 개수 분포"
)

fig.update_traces(
    marker=dict(size=4, opacity=0.55),
    boxmean=False,
    hovertemplate=(
        "제품 유형: %{x}<br>"
        "성분 개수: %{y}개"
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
        title="제품 하나의 성분 개수",
        rangemode="tozero"
    ),
    margin=dict(
        l=80,
        r=40,
        t=80,
        b=150
    )
)

st.plotly_chart(fig, use_container_width=True)


# --------------------------------------------------
# 제품 유형별 성분 개수 통계
# --------------------------------------------------
summary = (
    plot_data.groupby("category")["성분개수"]
    .agg(
        제품수="count",
        평균="mean",
        중앙값="median",
        최솟값="min",
        최댓값="max"
    )
    .reset_index()
)

st.subheader("제품 유형별 성분 개수 통계")

summary["평균"] = summary["평균"].round(1)
summary["중앙값"] = summary["중앙값"].round(1)

summary = summary.rename(
    columns={
        "category": "제품 유형",
        "제품수": "분석 제품 수",
        "평균": "평균 성분 개수",
        "중앙값": "중앙 성분 개수",
        "최솟값": "최소 성분 개수",
        "최댓값": "최대 성분 개수"
    }
)

st.dataframe(
    summary,
    use_container_width=True,
    hide_index=True
)


# --------------------------------------------------
# 그래프 해석
# --------------------------------------------------
st.subheader("이 그래프로 알 수 있는 것")

st.write(
    "상자그림의 가운데 선은 제품 유형별 성분 개수의 중앙값을 "
    "나타냅니다. 중앙값은 제품을 성분 개수 순서대로 나열했을 때 "
    "가운데에 위치하는 값입니다."
)

st.write(
    "상자의 아래쪽과 위쪽은 각각 1사분위수와 3사분위수를 "
    "나타냅니다. 따라서 상자 안에는 가운데 50%의 제품이 포함됩니다. "
    "상자와 수염의 길이를 비교하면 제품 유형별 성분 개수의 "
    "분포 범위와 차이를 살펴볼 수 있습니다."
)

st.write(
    "그래프에 표시된 점은 개별 제품을 의미합니다. "
    "상자에서 멀리 떨어진 점은 이상치로 나타날 수 있지만, "
    "반드시 잘못된 데이터라는 뜻은 아닙니다. "
    "실제로 다른 제품보다 성분이 많거나 적은 제품일 수도 있습니다."
)


# --------------------------------------------------
# 제품 유형별 중앙값 비교 해석
# --------------------------------------------------
st.markdown("**제품 유형별 중앙값 비교**")

median_summary = (
    plot_data.groupby("category")["성분개수"]
    .median()
    .sort_values(ascending=False)
)

highest_category = median_summary.index[0]
highest_value = median_summary.iloc[0]

lowest_category = median_summary.index[-1]
lowest_value = median_summary.iloc[-1]

st.write(
    f"선택한 제품 유형 중 중앙값이 가장 높은 유형은 "
    f"{highest_category}이며, 중앙 성분 개수는 "
    f"{highest_value:.1f}개입니다."
)

st.write(
    f"중앙값이 가장 낮은 유형은 {lowest_category}이며, "
    f"중앙 성분 개수는 {lowest_value:.1f}개입니다."
)


# --------------------------------------------------
# 제품 유형별 분포 차이 해석
# --------------------------------------------------
st.markdown("**제품 유형별 성분 개수의 차이**")

spread_summary = (
    plot_data.groupby("category")["성분개수"]
    .agg(
        최소="min",
        최대="max",
        중앙값="median",
        사분위범위=lambda x: x.quantile(0.75) - x.quantile(0.25)
    )
)

widest_category = spread_summary["사분위범위"].idxmax()
widest_value = spread_summary.loc[widest_category, "사분위범위"]

narrowest_category = spread_summary["사분위범위"].idxmin()
narrowest_value = spread_summary.loc[narrowest_category, "사분위범위"]

st.write(
    f"가운데 50% 제품의 성분 개수 차이가 가장 큰 유형은 "
    f"{widest_category}입니다. 사분위범위는 "
    f"{widest_value:.1f}개입니다."
)

st.write(
    f"가운데 50% 제품의 성분 개수 차이가 가장 작은 유형은 "
    f"{narrowest_category}입니다. 사분위범위는 "
    f"{narrowest_value:.1f}개입니다."
)


# --------------------------------------------------
# 분석의 의미와 주의점
# --------------------------------------------------
st.markdown("**분석 결과의 의미와 주의점**")

st.write(
    "제품 유형별 성분 개수의 중앙값과 분포를 비교하면 "
    "어떤 유형의 제품이 상대적으로 성분이 많은 편인지, "
    "어떤 유형은 성분 개수가 적거나 제품마다 차이가 큰지 "
    "파악할 수 있습니다."
)

st.write(
    "하지만 성분 개수가 많다고 해서 제품의 효능이 더 뛰어나거나 "
    "품질이 더 좋다는 뜻은 아닙니다. 성분 개수는 제품의 구성을 "
    "비교하기 위한 하나의 정보일 뿐입니다."
)

st.write(
    "이 분석은 제품 유형에 따라 전성분의 구성이 어떻게 달라지는지 "
    "이해하는 데 도움이 됩니다. 또한 제품별 성분 개수라는 특징이 "
    "전성분을 활용한 제품 유형 분류 모델에서 어떤 역할을 할 수 "
    "있는지 생각해 보는 기초 자료가 됩니다."
)
