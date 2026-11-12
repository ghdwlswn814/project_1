
import streamlit as st
import pandas as pd
import plotly.express as px


# --------------------------------------------------
# 기본 설정
# --------------------------------------------------

st.set_page_config(
    page_title="화장품 성분 예측실 - 성분 사용 빈도 비교",
    page_icon="🧴",
    layout="wide"
)

st.title("🧴 화장품 성분 예측실")
st.subheader("성분 사용 빈도 비교")

st.write(
    "제품 유형에 따라 자주 사용되는 화장품 성분이 "
    "어떻게 다른지 확인합니다."
)


# --------------------------------------------------
# 데이터 주소
# --------------------------------------------------

DATA_URL = (
    "https://raw.githubusercontent.com/"
    "LadyFelicia/Sephora-Analysis/main/Product%20Info.csv"
)


# --------------------------------------------------
# 데이터 불러오기
# --------------------------------------------------

@st.cache_data
def load_data():
    return pd.read_csv(
        DATA_URL,
        encoding="utf-8",
        low_memory=False
    )


try:
    df = load_data()

except Exception as e:
    st.error("데이터를 불러오지 못했습니다.")
    st.write("인터넷 연결이나 데이터 주소를 확인해 주세요.")
    st.code(str(e))
    st.stop()


# --------------------------------------------------
# 필요한 열 확인
# --------------------------------------------------

required_columns = ["ingredients", "category"]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    st.error(
        "데이터에서 필요한 열을 찾을 수 없습니다: "
        + ", ".join(missing_columns)
    )
    st.stop()


# --------------------------------------------------
# 데이터 전처리
# --------------------------------------------------

st.divider()
st.markdown("### 분석할 데이터 준비")

work = df[["category", "ingredients"]].copy()

# 제품을 구분하기 위한 번호
work["제품번호"] = range(len(work))

# 제품 유형의 앞뒤 공백 제거
work["category"] = (
    work["category"]
    .astype("string")
    .str.strip()
)

# 제품 유형이 비어 있는 데이터 확인
category_missing = (
    work["category"].isna()
    | work["category"].eq("")
)

category_missing_count = int(category_missing.sum())

# 제품 유형이 있는 데이터만 사용
work = work[~category_missing].copy()

# 전성분이 비어 있는 데이터 확인
ingredient_text = (
    work["ingredients"]
    .fillna("")
    .astype(str)
    .str.strip()
)

ingredient_missing = ingredient_text.eq("")
ingredient_missing_count = int(ingredient_missing.sum())

work["ingredients"] = ingredient_text

# 전성분이 있는 데이터만 사용
work = work[~ingredient_missing].copy()


# --------------------------------------------------
# 전성분을 개별 성분으로 분리
# --------------------------------------------------

# 쉼표를 기준으로 성분 분리
work["성분"] = work["ingredients"].str.split(",")

# 한 행에 하나의 성분만 남도록 변환
ingredient_data = work.explode("성분")

# 성분 이름 앞뒤 공백 제거 및 대소문자 통일
ingredient_data["성분"] = (
    ingredient_data["성분"]
    .astype("string")
    .str.strip()
    .str.casefold()
)

# 빈 성분 제거
ingredient_data = ingredient_data[
    ingredient_data["성분"].notna()
    & ingredient_data["성분"].ne("")
].copy()

# 같은 제품 안에서 같은 성분이 중복되면 한 번만 계산
ingredient_data = ingredient_data.drop_duplicates(
    subset=["제품번호", "성분"]
)

if ingredient_data.empty:
    st.warning("분석할 수 있는 성분 데이터가 없습니다.")
    st.stop()


# --------------------------------------------------
# 데이터 요약
# --------------------------------------------------

col1, col2, col3 = st.columns(3)

col1.metric(
    "분석 가능한 제품 수",
    f"{work['제품번호'].nunique():,}개"
)

col2.metric(
    "분석 가능한 성분 종류",
    f"{ingredient_data['성분'].nunique():,}개"
)

col3.metric(
    "전성분 결측 제품",
    f"{ingredient_missing_count:,}개"
)

st.caption(
    f"제품 유형이 비어 있는 제품 {category_missing_count:,}개와 "
    f"전성분이 비어 있는 제품 {ingredient_missing_count:,}개는 "
    "성분 사용 빈도 분석에서 제외했습니다."
)


# --------------------------------------------------
# 제품 유형 선택
# --------------------------------------------------

st.divider()
st.markdown("### 분석할 제품 유형 선택")

# 제품 유형별 제품 수 계산
category_counts = (
    work["category"]
    .value_counts()
)

category_options = category_counts.index.tolist()

if not category_options:
    st.warning("선택할 수 있는 제품 유형이 없습니다.")
    st.stop()

# 가장 많은 제품 유형을 기본 선택
default_category = category_options[0]

selected_categories = st.multiselect(
    "비교할 제품 유형을 선택하세요.",
    options=category_options,
    default=[default_category],
    help="여러 제품 유형을 선택하면 각각의 성분 사용 빈도를 비교할 수 있습니다."
)

if not selected_categories:
    st.info("분석할 제품 유형을 하나 이상 선택해 주세요.")
    st.stop()


# --------------------------------------------------
# 성분 사용 빈도 계산
# --------------------------------------------------

st.divider()
st.markdown("### 제품 유형별 성분 사용 빈도")


def get_top_ingredients(category):
    """선택한 제품 유형에서 성분을 포함한 제품 수 계산"""

    selected_data = ingredient_data[
        ingredient_data["category"] == category
    ]

    # 성분별로 해당 성분을 포함한 제품 수 계산
    counts = (
        selected_data
        .groupby("성분")["제품번호"]
        .nunique()
        .reset_index(name="제품 수")
    )

    # 제품 수가 많은 순서대로 정렬
    counts = counts.sort_values(
        by=["제품 수", "성분"],
        ascending=[False, True]
    )

    # 상위 10개 성분만 선택
    return counts.head(10).copy()


# --------------------------------------------------
# 제품 유형별 그래프
# --------------------------------------------------

comparison_results = []

for category in selected_categories:

    category_product_count = int(
        work.loc[
            work["category"] == category,
            "제품번호"
        ].nunique()
    )

    top_ingredients = get_top_ingredients(category)

    st.markdown(f"#### {category}")

    st.caption(
        f"분석 대상 제품 수: {category_product_count:,}개"
    )

    if top_ingredients.empty:
        st.warning(
            f"{category} 유형에는 분석할 수 있는 성분이 없습니다."
        )
        continue

    # 가로 막대그래프에서 높은 값이 위쪽에 오도록 정렬
    plot_data = top_ingredients.sort_values(
        by="제품 수",
        ascending=True
    )

    fig = px.bar(
        plot_data,
        x="제품 수",
        y="성분",
        orientation="h",
        text="제품 수",
        labels={
            "제품 수": "해당 성분을 포함한 제품 수(개)",
            "성분": "성분 이름"
        },
        title=f"{category} 유형에서 자주 사용되는 상위 10개 성분"
    )

    fig.update_traces(
        textposition="outside",
        cliponaxis=False,
        hovertemplate=(
            "성분: %{y}<br>"
            "해당 성분을 포함한 제품 수: %{x:,}개"
            "<extra></extra>"
        )
    )

    fig.update_layout(
        height=470,
        xaxis_title="해당 성분을 포함한 제품 수(개)",
        yaxis_title="성분 이름",
        font=dict(size=13),
        title_font_size=18,
        margin=dict(
            l=220,
            r=70,
            t=70,
            b=50
        ),
        hoverlabel=dict(font_size=13)
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # 가장 자주 사용되는 성분 기록
    max_count = top_ingredients["제품 수"].max()

    most_used = top_ingredients[
        top_ingredients["제품 수"] == max_count
    ]["성분"].tolist()

    comparison_results.append({
        "제품 유형": category,
        "가장 자주 사용되는 성분": ", ".join(most_used),
        "포함 제품 수": int(max_count)
    })


# --------------------------------------------------
# 그래프 해석
# --------------------------------------------------

st.divider()
st.markdown("### 이 그래프로 알 수 있는 것")

if comparison_results:

    for result in comparison_results:
        st.write(
            f"**{result['제품 유형']}** 유형에서 가장 자주 사용되는 "
            f"성분은 **{result['가장 자주 사용되는 성분']}**이며, "
            f"{result['포함 제품 수']:,}개 제품에 포함되어 있습니다."
        )

    if len(comparison_results) >= 2:
        st.write(
            "선택한 제품 유형들의 상위 성분을 비교하면 "
            "제품 유형에 따라 자주 사용되는 성분의 구성이 "
            "어떻게 다른지 확인할 수 있습니다."
        )

    st.write(
        "성분 사용 빈도가 높다는 사실만으로 해당 성분의 효능이나 "
        "제품의 품질이 더 높다고 판단할 수는 없습니다."
    )

    st.write(
        "이 분석은 전성분을 바탕으로 제품 유형을 예측하는 "
        "분류 모델을 만들 때, 제품 유형을 구분하는 데 활용할 수 있는 "
        "성분 특징을 파악하는 기초 자료가 됩니다."
    )


# --------------------------------------------------
# 마지막 안내
# --------------------------------------------------

st.divider()

st.info(
    "이 페이지는 제품 유형별 성분 사용 빈도를 분석하는 기능입니다. "
    "분류 모델과 새로운 제품 유형 예측 기능은 이후 단계에서 추가할 예정입니다."
)
