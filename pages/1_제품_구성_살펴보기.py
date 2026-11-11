
import streamlit as st
import pandas as pd
import plotly.express as px


# --------------------------------------------------
# 페이지 기본 설정
# --------------------------------------------------
st.set_page_config(
    page_title="화장품 성분 예측실 - 탐색",
    page_icon="🧴",
    layout="wide"
)

st.title("🧴 화장품 성분 예측실 - 탐색")
st.subheader("첫 번째 탐색: 제품 유형별 개수")
st.write(
    "화장품 데이터에 어떤 제품 유형이 얼마나 포함되어 있는지 "
    "막대그래프로 확인합니다."
)


# --------------------------------------------------
# 데이터 주소 및 불러오기
# --------------------------------------------------
DATA_URL = (
    "https://raw.githubusercontent.com/"
    "LadyFelicia/Sephora-Analysis/main/Product%20Info.csv"
)


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
    st.error("데이터를 불러오는 중 오류가 발생했습니다.")
    st.write("오류 내용:", str(e))
    st.stop()


# --------------------------------------------------
# 필요한 열 확인
# --------------------------------------------------
if "category" not in df.columns:
    st.error("데이터에 제품 유형을 나타내는 'category' 열이 없습니다.")
    st.stop()


# --------------------------------------------------
# 제품 유형별 개수 계산
# --------------------------------------------------
category_data = df["category"].astype("string")

# 비어 있는 값과 공백만 있는 값 확인
is_missing = (
    category_data.isna()
    | category_data.str.strip().eq("")
)

missing_count = int(is_missing.sum())

# 결측치를 제외하고 원본 제품 유형별로 개수 집계
valid_categories = category_data[~is_missing]

category_counts = (
    valid_categories
    .value_counts()
    .rename_axis("제품 유형")
    .reset_index(name="제품 수")
)

# 제품 수가 많은 순서대로 정렬
category_counts = category_counts.sort_values(
    by=["제품 수", "제품 유형"],
    ascending=[False, True]
).reset_index(drop=True)


# --------------------------------------------------
# 데이터 요약
# --------------------------------------------------
st.divider()
st.markdown("### 데이터 요약")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "전체 제품 수",
        f"{len(df):,}개"
    )

with col2:
    st.metric(
        "제품 유형 수",
        f"{len(category_counts):,}개"
    )

with col3:
    st.metric(
        "제품 유형 결측치",
        f"{missing_count:,}개"
    )

st.caption(
    "제품 유형이 비어 있는 데이터는 그래프의 집계에서 제외했습니다. "
    "제품 유형의 원본 값은 임의로 합치거나 변경하지 않았습니다."
)


# --------------------------------------------------
# 첫 번째 그래프
# --------------------------------------------------
st.divider()
st.markdown("### 1. 제품 유형별 제품 수")

if category_counts.empty:
    st.warning("집계할 수 있는 제품 유형 데이터가 없습니다.")

else:
    # 가로 막대그래프에서 가장 많은 유형이 위쪽에 오도록
    # 그래프용 데이터는 오름차순으로 정렬
    plot_data = category_counts.sort_values(
        by="제품 수",
        ascending=True
    )

    # 제품 유형 수에 따라 그래프 높이 조절
    graph_height = max(500, len(plot_data) * 32)

    fig = px.bar(
        plot_data,
        x="제품 수",
        y="제품 유형",
        orientation="h",
        text="제품 수",
        labels={
            "제품 유형": "제품 유형",
            "제품 수": "제품 수(개)"
        },
        title="화장품 제품 유형별 개수"
    )

    fig.update_traces(
        textposition="outside",
        hovertemplate=(
            "제품 유형: %{y}<br>"
            "제품 수: %{x:,}개"
            "<extra></extra>"
        ),
        cliponaxis=False
    )

    fig.update_layout(
        height=graph_height,
        xaxis_title="제품 수(개)",
        yaxis_title="제품 유형",
        yaxis={
            "categoryorder": "array",
            "categoryarray": plot_data["제품 유형"].tolist()
        },
        font=dict(size=13),
        title_font_size=20,
        margin=dict(l=220, r=80, t=70, b=50),
        hoverlabel=dict(font_size=13)
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


    # --------------------------------------------------
    # 제품 유형별 개수 표
    # --------------------------------------------------
    with st.expander("제품 유형별 개수 자세히 보기"):
        display_data = category_counts.copy()
        display_data.index = range(1, len(display_data) + 1)

        st.dataframe(
            display_data,
            use_container_width=True
        )


    # --------------------------------------------------
    # 그래프 해석
    # --------------------------------------------------
    st.divider()
    st.markdown("### 이 그래프로 알 수 있는 것")

    max_count = category_counts["제품 수"].max()
    min_count = category_counts["제품 수"].min()

    most_categories = category_counts.loc[
        category_counts["제품 수"] == max_count,
        "제품 유형"
    ].tolist()

    least_categories = category_counts.loc[
        category_counts["제품 수"] == min_count,
        "제품 유형"
    ].tolist()

    most_text = ", ".join(most_categories)
    least_text = ", ".join(least_categories)

    st.markdown(
        f"- **가장 많은 제품 유형:** {most_text} "
        f"({max_count:,}개)"
    )

    st.markdown(
        f"- **가장 적은 제품 유형:** {least_text} "
        f"({min_count:,}개)"
    )

    st.write(
        f"전체 {len(category_counts):,}개의 제품 유형을 비교하면 "
        "제품 유형마다 데이터에 포함된 제품 수가 서로 다르다는 것을 "
        "확인할 수 있습니다."
    )

    st.write(
        "이 결과는 이후 분류 모델을 만들 때 예측 대상으로 사용할 "
        "제품 유형의 분포를 파악하는 데 도움이 됩니다."
    )

    # 제품 유형별 데이터 수 차이 확인
    if len(category_counts) >= 2:
        largest = max_count
        smallest = min_count

        if smallest == 0:
            st.write(
                "제품 유형별 데이터 수에 차이가 있습니다. "
                "데이터가 적은 유형은 분류 모델 학습에 충분하지 않을 수 있습니다."
            )
        else:
            ratio = largest / smallest
            st.write(
                f"제품 수가 가장 많은 유형은 가장 적은 유형보다 "
                f"약 {ratio:.1f}배 많습니다. "
                "제품 유형별 데이터 수의 차이가 큰 경우 "
                "분류 모델의 예측 성능에도 영향을 줄 수 있습니다."
            )
