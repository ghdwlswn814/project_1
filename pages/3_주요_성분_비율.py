
import streamlit as st
import pandas as pd
import plotly.graph_objects as go


# --------------------------------------------------
# 페이지 설정
# --------------------------------------------------
st.set_page_config(
    page_title="주요 성분 사용 비율 비교",
    page_icon="🧴",
    layout="wide"
)

st.title("🧴 주요 성분 사용 비율 비교")
st.write(
    "제품 유형별로 주요 성분이 얼마나 자주 사용되는지 "
    "히트맵으로 비교합니다."
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
    df = pd.read_csv(DATA_URL, encoding="utf-8")
    return df


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
# 데이터 전처리
# --------------------------------------------------

# 제품 유형이 비어 있는 제품 제외
# 전성분이 비어 있는 제품 제외
work = df[["ingredients", "category"]].copy()

work["category"] = (
    work["category"]
    .astype("string")
    .str.strip()
)

work["ingredients"] = (
    work["ingredients"]
    .astype("string")
    .str.strip()
)

work = work.dropna(subset=["category", "ingredients"])

work = work[
    (work["category"] != "")
    & (work["ingredients"] != "")
].copy()

if work.empty:
    st.warning("분석할 수 있는 제품 데이터가 없습니다.")
    st.stop()


# 각 제품에 고유 번호 부여
work = work.reset_index(drop=True)
work["_product_id"] = work.index


# --------------------------------------------------
# 전성분 분리 및 정리
# --------------------------------------------------

def clean_ingredients(ingredient_text):
    """
    쉼표를 기준으로 성분을 분리하고
    공백과 대소문자를 통일합니다.
    한 제품 안의 중복 성분은 한 번만 남깁니다.
    """
    ingredients = []

    for ingredient in str(ingredient_text).split(","):
        ingredient = ingredient.strip().casefold()

        if ingredient and ingredient not in ingredients:
            ingredients.append(ingredient)

    return ingredients


work["성분목록"] = work["ingredients"].apply(clean_ingredients)


# 전처리 후 성분이 하나도 남지 않은 제품 제외
work = work[
    work["성분목록"].apply(len) > 0
].copy()

if work.empty:
    st.warning("분석할 수 있는 성분 데이터가 없습니다.")
    st.stop()


# --------------------------------------------------
# 전체 제품 유형별 제품 수
# --------------------------------------------------
category_counts = (
    work.groupby("category")["_product_id"]
    .nunique()
    .sort_values(ascending=False)
)

if category_counts.empty:
    st.warning("분석할 제품 유형이 없습니다.")
    st.stop()


# --------------------------------------------------
# 전체 데이터에서 주요 성분 상위 15개 찾기
# --------------------------------------------------

exploded = work[
    ["_product_id", "category", "성분목록"]
].explode("성분목록")

exploded = exploded.rename(
    columns={"성분목록": "성분"}
)

# 혹시 모를 중복을 한 번 더 제거
exploded = exploded.drop_duplicates(
    subset=["_product_id", "성분"]
)

# 성분을 포함한 제품 수 기준으로 상위 15개 선정
top_ingredients = (
    exploded["성분"]
    .value_counts()
    .head(15)
    .index
    .tolist()
)

if not top_ingredients:
    st.warning("주요 성분을 찾을 수 없습니다.")
    st.stop()


# --------------------------------------------------
# 제품 유형 선택
# --------------------------------------------------
st.subheader("비교할 제품 유형 선택")

default_categories = category_counts.head(3).index.tolist()

selected_categories = st.multiselect(
    "비교하고 싶은 제품 유형을 선택하세요.",
    options=category_counts.index.tolist(),
    default=default_categories
)

if not selected_categories:
    st.info("비교할 제품 유형을 하나 이상 선택해 주세요.")
    st.stop()


# --------------------------------------------------
# 선택한 제품 유형별 성분 사용 비율 계산
# --------------------------------------------------

# 선택한 제품 유형에 해당하는 제품만 사용
selected_work = work[
    work["category"].isin(selected_categories)
].copy()

# 제품 유형별 전체 분석 대상 제품 수
total_products = (
    selected_work.groupby("category")["_product_id"]
    .nunique()
    .to_dict()
)

# 상위 15개 성분만 남김
selected_exploded = exploded[
    (exploded["category"].isin(selected_categories))
    & (exploded["성분"].isin(top_ingredients))
].copy()

# 제품 유형과 성분별 포함 제품 수
ingredient_counts = (
    selected_exploded
    .groupby(["category", "성분"])["_product_id"]
    .nunique()
    .to_dict()
)


# --------------------------------------------------
# 히트맵에 사용할 데이터 만들기
# --------------------------------------------------

z_values = []
text_values = []
custom_data = []

for category in selected_categories:
    total = total_products.get(category, 0)

    category_rates = []
    category_texts = []
    category_custom = []

    for ingredient in top_ingredients:
        count = ingredient_counts.get(
            (category, ingredient), 0
        )

        if total > 0:
            percentage = count / total * 100
        else:
            percentage = 0

        category_rates.append(percentage)
        category_texts.append(f"{percentage:.1f}%")
        category_custom.append(
            [count, total, percentage]
        )

    z_values.append(category_rates)
    text_values.append(category_texts)
    custom_data.append(category_custom)


# --------------------------------------------------
# 히트맵 그리기
# --------------------------------------------------

st.subheader("제품 유형별 주요 성분 사용 비율")

fig = go.Figure(
    data=go.Heatmap(
        z=z_values,
        x=top_ingredients,
        y=selected_categories,
        text=text_values,
        texttemplate="%{text}",
        textfont={"size": 11},
        customdata=custom_data,
        colorscale="Viridis",
        colorbar={
            "title": "사용 비율(%)"
        },
        hovertemplate=(
            "제품 유형: %{y}<br>"
            "성분 이름: %{x}<br>"
            "해당 성분을 포함한 제품 수: %{customdata[0]}개<br>"
            "전체 제품 수: %{customdata[1]}개<br>"
            "사용 비율: %{customdata[2]:.1f}%"
            "<extra></extra>"
        )
    )
)

fig.update_layout(
    height=max(550, 90 * len(selected_categories) + 250),
    xaxis_title="주요 성분",
    yaxis_title="제품 유형",
    xaxis={
        "tickangle": -45,
        "tickfont": {"size": 11},
        "automargin": True
    },
    yaxis={
        "tickfont": {"size": 12},
        "automargin": True
    },
    margin={
        "l": 120,
        "r": 50,
        "t": 40,
        "b": 180
    }
)

st.plotly_chart(
    fig,
    use_container_width=True
)


# --------------------------------------------------
# 그래프 해석
# --------------------------------------------------

st.subheader("이 그래프로 알 수 있는 것")

st.markdown(
    """
    이 히트맵은 제품 유형별로 주요 성분이 사용되는 비율을
    비교한 그래프입니다.

    색이 진한 칸일수록 해당 제품 유형에서 그 성분을 포함하는
    제품의 비율이 높다는 뜻입니다.

    여러 제품 유형에서 공통적으로 높은 사용 비율을 보이는 성분은
    다양한 제품에 널리 사용되는 성분일 가능성이 있습니다.
    반면 특정 제품 유형에서 상대적으로 높은 비율을 보이는 성분은
    그 제품 유형의 성분 구성에서 특징적으로 나타나는 성분일 수 있습니다.

    이 그래프를 통해 제품 유형에 따라 전성분 구성이 어떻게 다른지
    비교할 수 있습니다.

    다만 성분 사용 비율만으로 성분의 효능이나 제품의 품질을
    단정할 수는 없습니다. 성분의 함량이나 배합 목적까지
    이 그래프가 보여 주는 것은 아니기 때문입니다.

    또한 제품 유형에 따라 자주 등장하는 성분을 파악하는 것은
    전성분을 입력값으로 활용하는 제품 유형 분류 모델의
    입력 특징을 이해하는 데 도움이 됩니다.
    """
)


# --------------------------------------------------
# 분석 기준 안내
# --------------------------------------------------

with st.expander("분석 기준 확인하기"):
    st.write(
        "주요 성분은 전체 분석 대상 제품에서 "
        "포함 제품 수가 많은 상위 15개 성분입니다."
    )

    st.write(
        "사용 비율은 해당 성분을 포함한 제품 수를 "
        "그 제품 유형의 전체 분석 대상 제품 수로 나눈 뒤 "
        "100을 곱해 계산합니다."
    )

    st.write(
        "전성분이 비어 있거나 제품 유형이 비어 있는 제품은 "
        "분석에서 제외했습니다. 한 제품 안에서 같은 성분이 "
        "중복되어도 한 번만 계산했습니다."
    )
