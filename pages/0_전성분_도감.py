
import re
import streamlit as st
import pandas as pd

# --------------------------------------------------
# 페이지 설정
# --------------------------------------------------
st.set_page_config(
    page_title="전성분 한글 사전",
    page_icon="📖",
    layout="wide"
)

DATA_URL = "https://raw.githubusercontent.com/LadyFelicia/Sephora-Analysis/main/Product%20Info.csv"


# --------------------------------------------------
# 데이터 불러오기
# --------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")

    if "ingredients" not in df.columns:
        raise ValueError("데이터에 ingredients 컬럼이 없습니다.")

    return df


# --------------------------------------------------
# 성분 이름 정리
# --------------------------------------------------
def clean_ingredient(name):
    name = str(name).strip()
    name = re.sub(r"\s+", " ", name)
    return name


@st.cache_data
def get_ingredient_list():
    df = load_data()

    ingredients_set = set()

    for value in df["ingredients"].dropna():
        # 쉼표를 기준으로 성분 분리
        parts = str(value).split(",")

        for part in parts:
            ingredient = clean_ingredient(part)

            if ingredient and ingredient.lower() not in ["nan", "none"]:
                ingredients_set.add(ingredient)

    return sorted(ingredients_set, key=str.lower)


# --------------------------------------------------
# 자주 사용되는 성분 설명 사전
# 실제 데이터에 등장하는 성분에 대해서만 표시됨
# --------------------------------------------------
INGREDIENT_INFO = {
    "water": {
        "korean": "정제수",
        "role": "기본 용매",
        "description": "화장품에서 다른 성분을 녹이거나 섞는 데 사용되는 가장 기본적인 성분입니다. 제형의 바탕이 되는 역할을 합니다.",
        "category": "기타 기본 성분"
    },
    "aqua": {
        "korean": "정제수",
        "role": "기본 용매",
        "description": "화장품에서 다른 성분을 녹이거나 섞는 데 사용되는 가장 기본적인 성분입니다. 제형의 바탕이 되는 역할을 합니다.",
        "category": "기타 기본 성분"
    },
    "glycerin": {
        "korean": "글리세린",
        "role": "보습",
        "description": "주변의 수분을 끌어당기는 성질이 있는 보습 성분입니다. 피부가 건조해지는 것을 줄이고 촉촉함을 유지하는 데 도움을 줍니다.",
        "category": "보습 성분"
    },
    "niacinamide": {
        "korean": "나이아신아마이드",
        "role": "피부 컨디셔닝",
        "description": "비타민 B3의 한 종류로, 피부 장벽을 건강하게 유지하고 피부 톤과 피부 상태를 관리하는 데 사용됩니다.",
        "category": "피부 컨디셔닝 성분"
    },
    "butylene glycol": {
        "korean": "부틸렌글라이콜",
        "role": "보습 및 용매",
        "description": "수분을 유지하는 데 도움을 주며, 화장품 속 여러 성분이 잘 섞이도록 돕는 용매로도 사용됩니다.",
        "category": "보습 성분"
    },
    "panthenol": {
        "korean": "판테놀",
        "role": "보습 및 피부 컨디셔닝",
        "description": "비타민 B5의 전구체로 알려진 성분입니다. 피부에 수분을 공급하고 피부를 편안하게 관리하는 데 사용됩니다.",
        "category": "피부 컨디셔닝 성분"
    },
    "hyaluronic acid": {
        "korean": "히알루론산",
        "role": "보습",
        "description": "수분을 끌어당기고 유지하는 성질이 있어 피부에 촉촉함을 주는 데 사용되는 보습 성분입니다.",
        "category": "보습 성분"
    },
    "sodium hyaluronate": {
        "korean": "소듐하이알루로네이트",
        "role": "보습",
        "description": "히알루론산의 염 형태로, 수분을 유지하고 피부에 촉촉함을 주는 데 사용되는 성분입니다.",
        "category": "보습 성분"
    },
    "dimethicone": {
        "korean": "다이메티콘",
        "role": "피부 보호 및 사용감 개선",
        "description": "실리콘 계열 성분으로 피부 표면에 부드러운 막을 형성하고 발림성과 사용감을 개선하는 데 사용됩니다.",
        "category": "피부 컨디셔닝 성분"
    },
    "tocopherol": {
        "korean": "토코페롤",
        "role": "산화 방지",
        "description": "비타민 E의 한 종류입니다. 화장품에서 산화를 방지하는 역할을 하며 피부 컨디셔닝 목적으로도 사용됩니다.",
        "category": "기타 기능성 성분"
    },
    "ascorbic acid": {
        "korean": "아스코빅애씨드",
        "role": "피부 컨디셔닝",
        "description": "비타민 C의 한 종류로, 항산화 및 피부 톤 관리 목적으로 화장품에 사용됩니다.",
        "category": "기타 기능성 성분"
    },
    "salicylic acid": {
        "korean": "살리실릭애씨드",
        "role": "각질 관리",
        "description": "피부 표면의 각질을 관리하는 데 사용되는 성분입니다. 제품의 농도와 사용 방법에 따라 피부에 자극을 줄 수 있습니다.",
        "category": "각질 관리 성분"
    },
    "allantoin": {
        "korean": "알란토인",
        "role": "피부 컨디셔닝",
        "description": "피부를 부드럽고 편안하게 관리하는 목적으로 사용되는 성분입니다.",
        "category": "피부 컨디셔닝 성분"
    },
    "squalane": {
        "korean": "스쿠알란",
        "role": "피부 유연화",
        "description": "피부 표면을 부드럽게 하고 수분 손실을 줄이는 데 도움을 주는 유연화 성분입니다.",
        "category": "피부 컨디셔닝 성분"
    },
    "ceramide np": {
        "korean": "세라마이드엔피",
        "role": "피부 장벽 관리",
        "description": "피부 장벽을 구성하는 지질 성분 중 하나와 유사한 성분으로, 피부 장벽 관리와 보습에 사용됩니다.",
        "category": "피부 컨디셔닝 성분"
    },
    "carbomer": {
        "korean": "카보머",
        "role": "점도 조절",
        "description": "화장품의 점도를 높이고 젤 형태의 제형을 만드는 데 사용되는 성분입니다.",
        "category": "제형 조절 성분"
    },
    "xanthan gum": {
        "korean": "잔탄검",
        "role": "점도 조절",
        "description": "화장품의 점도를 조절하고 제형이 안정적으로 유지되도록 돕는 성분입니다.",
        "category": "제형 조절 성분"
    },
    "cetearyl alcohol": {
        "korean": "세테아릴알코올",
        "role": "유화 및 피부 유연화",
        "description": "지방알코올의 한 종류입니다. 화장품의 질감을 조절하고 유화 안정화와 피부 유연화에 사용됩니다.",
        "category": "제형 조절 성분"
    },
    "phenoxyethanol": {
        "korean": "페녹시에탄올",
        "role": "보존",
        "description": "화장품이 미생물에 의해 변질되는 것을 줄이기 위해 사용하는 보존 성분입니다.",
        "category": "보존 성분"
    },
    "fragrance": {
        "korean": "향료",
        "role": "향 부여",
        "description": "화장품에 향을 더하기 위해 사용하는 성분입니다. 개인에 따라 민감하게 반응할 수 있습니다.",
        "category": "향료 성분"
    },
    "parfum": {
        "korean": "향료",
        "role": "향 부여",
        "description": "화장품에 향을 더하기 위해 사용하는 성분입니다. 개인에 따라 민감하게 반응할 수 있습니다.",
        "category": "향료 성분"
    },
}


# --------------------------------------------------
# 성분 정보 가져오기
# --------------------------------------------------
def get_ingredient_info(ingredient):
    key = ingredient.strip().lower()

    if key in INGREDIENT_INFO:
        return INGREDIENT_INFO[key]

    return {
        "korean": "한국어 이름 정보 없음",
        "role": "정보 없음",
        "description": "성분 설명이 준비되지 않았습니다.",
        "category": "분류 정보 없음"
    }


# --------------------------------------------------
# 페이지 제목 및 소개
# --------------------------------------------------
st.title("📖 전성분 한글 사전")

st.markdown(
    """
    화장품 전성분에는 영어로 표기된 다양한 성분이 포함되어 있습니다.

    이 페이지에서는 실제 화장품 데이터에 등장하는 성분을 검색하고,
    한국어 이름과 주요 역할 및 특징을 확인할 수 있습니다.

    낯선 성분을 찾아보거나 여러 성분을 선택해 한꺼번에 살펴보세요.
    """
)

st.divider()


# --------------------------------------------------
# 데이터 불러오기
# --------------------------------------------------
try:
    ingredient_list = get_ingredient_list()

except Exception as e:
    st.error(f"데이터를 불러오는 중 오류가 발생했습니다: {e}")
    st.stop()


# --------------------------------------------------
# 데이터 요약
# --------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    st.metric("전체 제품에 등장하는 성분 수", f"{len(ingredient_list):,}개")

with col2:
    st.metric("설명이 준비된 성분 수", f"{sum(1 for x in ingredient_list if x.lower() in INGREDIENT_INFO):,}개")

st.divider()


# --------------------------------------------------
# 성분 검색
# --------------------------------------------------
st.subheader("🔎 성분 검색")

st.write("영어 성분명 또는 한국어 성분명을 입력해 원하는 성분을 찾아보세요.")

search_text = st.text_input(
    "성분 검색",
    placeholder="예: glycerin, 글리세린, niacinamide",
    key="ingredient_search"
)

category_options = [
    "전체 성분",
    "보습 성분",
    "피부 컨디셔닝 성분",
    "각질 관리 성분",
    "제형 조절 성분",
    "보존 성분",
    "향료 성분",
    "기타 기능성 성분",
    "기타 기본 성분",
    "분류 정보 없음"
]

selected_category = st.selectbox(
    "성분 역할별로 보기",
    category_options,
    key="ingredient_category"
)


# --------------------------------------------------
# 검색 결과 필터링
# --------------------------------------------------
filtered_ingredients = []

for ingredient in ingredient_list:
    info = get_ingredient_info(ingredient)

    # 역할별 분류 필터
    if selected_category != "전체 성분":
        if info["category"] != selected_category:
            continue

    # 영어 이름과 한국어 이름 모두 검색
    if search_text:
        search_lower = search_text.strip().lower()
        english_match = search_lower in ingredient.lower()
        korean_match = search_lower in info["korean"].lower()

        if not english_match and not korean_match:
            continue

    filtered_ingredients.append(ingredient)


st.caption(f"검색 결과: {len(filtered_ingredients):,}개 성분")

if filtered_ingredients:
    st.dataframe(
        pd.DataFrame({
            "영어 성분명": filtered_ingredients,
            "한국어 성분명": [
                get_ingredient_info(x)["korean"]
                for x in filtered_ingredients
            ],
            "주요 역할": [
                get_ingredient_info(x)["role"]
                for x in filtered_ingredients
            ]
        }),
        use_container_width=True,
        hide_index=True
    )
else:
    st.info("검색 조건에 맞는 성분이 없습니다.")


# --------------------------------------------------
# 성분 선택
# --------------------------------------------------
st.divider()
st.subheader("📝 성분 선택하기")

if "saved_ingredients" not in st.session_state:
    st.session_state.saved_ingredients = []

if filtered_ingredients:
    selected_from_search = st.multiselect(
        "자세히 알아볼 성분을 선택하세요. 여러 개를 선택할 수 있습니다.",
        options=filtered_ingredients,
        format_func=lambda x: (
            f"{x}  |  {get_ingredient_info(x)['korean']}"
        ),
        key="ingredient_search_selection"
    )

    if st.button("선택한 성분 추가", type="primary"):
        for ingredient in selected_from_search:
            if ingredient not in st.session_state.saved_ingredients:
                st.session_state.saved_ingredients.append(ingredient)

        st.success("선택한 성분을 목록에 추가했습니다.")

else:
    st.caption("검색 결과가 있을 때 성분을 선택할 수 있습니다.")


# --------------------------------------------------
# 선택한 성분 전체 확인
# --------------------------------------------------
st.divider()
st.subheader("📚 선택한 성분 모아보기")

saved_ingredients = st.session_state.saved_ingredients

if saved_ingredients:
    st.write(f"현재 선택한 성분: **{len(saved_ingredients)}개**")

    if st.button("선택한 성분 전체 해제"):
        st.session_state.saved_ingredients = []
        st.rerun()

    for index, ingredient in enumerate(saved_ingredients):
        info = get_ingredient_info(ingredient)

        with st.container(border=True):
            col1, col2 = st.columns([5, 1])

            with col1:
                st.markdown(f"### {ingredient}")
                st.write(f"**한국어 이름:** {info['korean']}")
                st.write(f"**주요 역할:** {info['role']}")
                st.write(f"**분류:** {info['category']}")
                st.write(info["description"])

            with col2:
                if st.button(
                    "해제",
                    key=f"remove_ingredient_{index}"
                ):
                    st.session_state.saved_ingredients.remove(ingredient)
                    st.rerun()

else:
    st.info("아직 선택한 성분이 없습니다. 위 검색 결과에서 성분을 선택해 보세요.")


# --------------------------------------------------
# 성분 설명 안내
# --------------------------------------------------
st.divider()

with st.expander("ℹ️ 성분 정보 이용 안내"):
    st.write(
        """
        이 사전은 데이터에 실제로 등장하는 성분을 바탕으로 구성되어 있습니다.

        한국어 이름과 역할 설명은 일부 자주 사용되는 성분에 대해서만 제공됩니다.
        설명이 없는 성분은 정확하지 않은 정보를 임의로 작성하지 않고
        '성분 설명이 준비되지 않았습니다.'라고 표시합니다.

        성분의 역할은 화장품에서 일반적으로 사용되는 목적을 설명한 것이며,
        실제 제품에서의 기능은 배합 비율과 다른 성분, 제품의 제형 등에 따라 달라질 수 있습니다.
        """
    )
