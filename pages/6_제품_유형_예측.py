
import streamlit as st
import pandas as pd
import numpy as np
import re
import math

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# ---------------------------------------------------------
# 1. 페이지 설정
# ---------------------------------------------------------

st.set_page_config(
    page_title="제품 유형 예측",
    page_icon="🧴",
    layout="wide"
)

st.title("🧴 제품 유형 예측")
st.write(
    "화장품 전성분을 선택하거나 직접 입력하면 "
    "인공지능 분류 모델이 어떤 유형의 제품인지 예측합니다."
)

DATA_URL = (
    "https://raw.githubusercontent.com/LadyFelicia/"
    "Sephora-Analysis/main/Product%20Info.csv"
)


# ---------------------------------------------------------
# 2. 자주 사용되는 성분의 한국어 이름과 설명
# ---------------------------------------------------------

INGREDIENT_INFO = {
    "water": {
        "korean": "정제수",
        "description": "화장품의 기본적인 용매로 사용되는 성분"
    },
    "aqua": {
        "korean": "정제수",
        "description": "화장품의 기본적인 용매로 사용되는 성분"
    },
    "glycerin": {
        "korean": "글리세린",
        "description": "피부의 수분 유지에 도움을 주는 보습 성분"
    },
    "niacinamide": {
        "korean": "나이아신아마이드",
        "description": "피부 컨디셔닝에 사용되는 성분"
    },
    "butylene glycol": {
        "korean": "부틸렌글라이콜",
        "description": "보습 및 용매 역할을 하는 성분"
    },
    "panthenol": {
        "korean": "판테놀",
        "description": "피부 컨디셔닝과 보습에 사용되는 성분"
    },
}

# 한국어 성분명과 데이터에 들어 있을 수 있는 영어 성분명 연결
KOREAN_ALIASES = {
    "정제수": ["water", "aqua"],
    "글리세린": ["glycerin"],
    "나이아신아마이드": ["niacinamide"],
    "부틸렌글라이콜": ["butylene glycol"],
    "판테놀": ["panthenol"],
}


# ---------------------------------------------------------
# 3. 데이터 전처리 함수
# ---------------------------------------------------------

def normalize_ingredient(value):
    """성분 이름의 공백과 대소문자를 정리한다."""
    if pd.isna(value):
        return ""

    value = str(value).strip()
    value = re.sub(r"\s+", " ", value)

    return value.casefold()


def parse_ingredients(value):
    """쉼표로 구분된 전성분을 분리하고 중복을 제거한다."""
    if pd.isna(value):
        return []

    ingredients = str(value).split(",")

    result = []
    seen = set()

    for ingredient in ingredients:
        normalized = normalize_ingredient(ingredient)

        if normalized and normalized not in seen:
            result.append(normalized)
            seen.add(normalized)

    return result


def get_ingredient_display(ingredient):
    """성분의 영어 이름과 한국어 이름을 함께 표시한다."""
    info = INGREDIENT_INFO.get(ingredient)

    if info:
        return f"{ingredient} ({info['korean']})"

    return ingredient


def get_ingredient_description(ingredient):
    """성분 설명을 반환한다."""
    info = INGREDIENT_INFO.get(ingredient)

    if info:
        return info["description"]

    return "성분 설명이 준비되지 않았습니다."


# ---------------------------------------------------------
# 4. 데이터 불러오기 및 모델 학습
# ---------------------------------------------------------

@st.cache_data(show_spinner="화장품 데이터를 불러오고 예측 모델을 학습하고 있습니다...")
def load_and_train():

    df = pd.read_csv(
        DATA_URL,
        encoding="utf-8"
    )

    required_columns = ["ingredients", "category"]

    for column in required_columns:
        if column not in df.columns:
            raise ValueError(
                f"데이터에 '{column}' 컬럼이 없습니다."
            )

    # 필요한 컬럼만 사용
    data = df[["ingredients", "category"]].copy()

    # 결측치 제거
    data = data.dropna(
        subset=["ingredients", "category"]
    )

    # 제품 유형 정리
    data["category"] = (
        data["category"]
        .astype(str)
        .str.strip()
    )

    data = data[
        (data["category"] != "") &
        (data["category"].str.lower() != "nan")
    ]

    # 전성분 분리
    data["ingredient_list"] = data["ingredients"].apply(
        parse_ingredients
    )

    # 성분이 없는 제품 제외
    data = data[
        data["ingredient_list"].apply(len) > 0
    ].copy()

    if len(data) < 10:
        raise ValueError(
            "학습에 사용할 수 있는 데이터가 너무 적습니다."
        )

    if data["category"].nunique() < 2:
        raise ValueError(
            "분류할 제품 유형이 2개 미만입니다."
        )

    # 독립변수와 종속변수
    X = data["ingredient_list"]
    y = data["category"]

    # 학습용/테스트용 데이터 분리
    test_count = math.ceil(len(data) * 0.2)
    train_count = len(data) - test_count
    class_count = y.nunique()

    # 층화 분할이 가능한지 확인
    category_counts = y.value_counts()

    can_stratify = (
        category_counts.min() >= 2
        and test_count >= class_count
        and train_count >= class_count
    )

    stratify_value = y if can_stratify else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=stratify_value
    )

    # 학습 데이터에 존재하는 성분만으로 특징 목록 생성
    mlb = MultiLabelBinarizer()

    X_train_encoded = mlb.fit_transform(X_train)
    X_test_encoded = mlb.transform(X_test)

    if len(mlb.classes_) == 0:
        raise ValueError(
            "학습에 사용할 수 있는 성분이 없습니다."
        )

    if y_train.nunique() < 2:
        raise ValueError(
            "학습 데이터에 제품 유형이 2개 이상 필요합니다."
        )

    # 랜덤 포레스트 분류 모델
    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced_subsample",
        n_jobs=-1
    )

    model.fit(
        X_train_encoded,
        y_train
    )

    # 테스트 데이터 예측
    y_pred = model.predict(X_test_encoded)

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    # 모델이 학습한 제품 유형만 성능표에 포함
    report = classification_report(
        y_test,
        y_pred,
        labels=model.classes_,
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(report).transpose()

    # 실제 테스트 데이터에만 있고 학습에는 없었던 제품 유형
    unseen_test_categories = (
        set(y_test.unique()) - set(model.classes_)
    )

    # 데이터 전체에 존재하는 성분 목록
    all_ingredients = sorted(
        set(
            ingredient
            for ingredient_list in data["ingredient_list"]
            for ingredient in ingredient_list
        )
    )

    return {
        "model": model,
        "mlb": mlb,
        "accuracy": accuracy,
        "report": report_df,
        "ingredients": all_ingredients,
        "classes": list(model.classes_),
        "total_rows": len(data),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "category_count": y.nunique(),
        "unseen_categories": unseen_test_categories,
        "stratified": can_stratify
    }


# ---------------------------------------------------------
# 5. 데이터 불러오기
# ---------------------------------------------------------

try:
    result = load_and_train()

except Exception as e:
    st.error("데이터를 불러오거나 모델을 학습하는 중 문제가 발생했습니다.")
    st.write("오류 내용:", str(e))
    st.info(
        "데이터 주소와 컬럼 이름을 확인하거나 "
        "잠시 후 다시 실행해 주세요."
    )
    st.stop()


model = result["model"]
mlb = result["mlb"]
ingredient_list = result["ingredients"]

# 학습에 사용된 성분 목록
trained_ingredients = set(mlb.classes_)

# 전체 데이터에 존재하지만 학습 데이터에는 없을 수 있는 성분은
# 선택 목록에는 표시하되 예측에는 반영되지 않도록 한다.


# ---------------------------------------------------------
# 6. 한국어 성분명 연결 함수
# ---------------------------------------------------------

def find_actual_ingredient(korean_name, available_ingredients):
    """한국어 성분명에 해당하는 실제 데이터 성분을 찾는다."""

    candidates = KOREAN_ALIASES.get(korean_name, [])

    for candidate in candidates:
        normalized = normalize_ingredient(candidate)

        if normalized in available_ingredients:
            return normalized

    return None


# 한국어 별칭 검색용 사전
alias_to_ingredient = {}

for korean_name in KOREAN_ALIASES:
    actual = find_actual_ingredient(
        korean_name,
        set(ingredient_list)
    )

    if actual:
        alias_to_ingredient[korean_name] = actual


# ---------------------------------------------------------
# 7. 화면: 성분 검색 및 선택
# ---------------------------------------------------------

st.divider()
st.subheader("1. 성분 검색 및 선택")

st.write(
    "화장품 성분 이름을 검색해 선택해 보세요. "
    "여러 성분을 한 번에 선택할 수 있습니다."
)

search_word = st.text_input(
    "성분 검색",
    placeholder="예: water, glycerin, 정제수, 글리세린"
)

# 검색어 정리
search_normalized = normalize_ingredient(search_word)

# 한국어 별칭 검색어 처리
search_target = alias_to_ingredient.get(
    search_word.strip(),
    search_normalized
)

filtered_ingredients = []

for ingredient in ingredient_list:

    korean_name = INGREDIENT_INFO.get(
        ingredient, {}
    ).get("korean", "")

    if (
        search_target in ingredient
        or search_normalized in ingredient
        or (
            korean_name
            and search_normalized in normalize_ingredient(korean_name)
        )
    ):
        filtered_ingredients.append(ingredient)

if search_word.strip() and not filtered_ingredients:
    st.info(
        "검색 결과가 없습니다. 다른 성분 이름을 입력해 주세요."
    )

# ---------------------------------------------------------
# 8. 예시 성분 조합
# ---------------------------------------------------------

st.divider()
st.subheader("2. 예시 성분 조합")

st.caption(
    "아래 조합은 성분 선택을 연습하기 위한 예시이며, "
    "실제 특정 제품의 전성분을 의미하지 않습니다."
)

example_sets = {
    "보습 성분 예시": [
        "정제수",
        "글리세린",
        "부틸렌글라이콜"
    ],
    "피부 컨디셔닝 성분 예시": [
        "나이아신아마이드",
        "판테놀",
        "글리세린"
    ]
}

col1, col2 = st.columns(2)

with col1:
    if st.button(
        "💧 보습 성분 예시 선택",
        use_container_width=True
    ):
        selected_example = []

        for korean_name in example_sets["보습 성분 예시"]:
            actual = find_actual_ingredient(
                korean_name,
                set(ingredient_list)
            )

            if actual:
                selected_example.append(actual)

        st.session_state["selected_ingredients"] = list(
            dict.fromkeys(selected_example)
        )

        st.rerun()

with col2:
    if st.button(
        "🌿 피부 컨디셔닝 성분 예시 선택",
        use_container_width=True
    ):
        selected_example = []

        for korean_name in example_sets["피부 컨디셔닝 성분 예시"]:
            actual = find_actual_ingredient(
                korean_name,
                set(ingredient_list)
            )

            if actual:
                selected_example.append(actual)

        st.session_state["selected_ingredients"] = list(
            dict.fromkeys(selected_example)
        )

        st.rerun()


# ---------------------------------------------------------
# 9. 검색 결과에서 여러 성분 선택
# ---------------------------------------------------------

st.divider()
st.subheader("3. 검색 결과에서 성분 선택")

if "selected_ingredients" not in st.session_state:
    st.session_state["selected_ingredients"] = []

# 검색 결과와 이미 선택한 성분을 함께 표시
selected_now = st.session_state["selected_ingredients"]

available_options = sorted(
    set(filtered_ingredients) | set(selected_now)
)

selected_ingredients = st.multiselect(
    "사용할 성분을 선택하세요.",
    options=available_options,
    format_func=get_ingredient_display,
    key="selected_ingredients"
)

# 선택된 성분 설명
if selected_ingredients:

    st.markdown("**선택한 성분의 설명**")

    for ingredient in selected_ingredients:
        st.write(
            f"**{get_ingredient_display(ingredient)}**"
            f" — {get_ingredient_description(ingredient)}"
        )


# ---------------------------------------------------------
# 10. 직접 전성분 입력
# ---------------------------------------------------------

st.divider()
st.subheader("4. 직접 전성분 입력")

st.write(
    "성분 이름을 알고 있다면 직접 입력할 수도 있습니다. "
    "성분은 쉼표(,)로 구분해 주세요."
)

direct_input = st.text_area(
    "전성분 입력",
    placeholder=(
        "예: Water, Glycerin, Niacinamide, "
        "Butylene Glycol, Panthenol"
    ),
    height=100
)

st.caption(
    "한국어 성분명도 일부 지원합니다. "
    "예: 정제수, 글리세린, 나이아신아마이드"
)


# ---------------------------------------------------------
# 11. 입력 성분 합치기 및 정리
# ---------------------------------------------------------

direct_ingredients = parse_ingredients(direct_input)

# 한국어 성분명을 데이터의 영어 성분명으로 변환
converted_direct = []
unknown_ingredients = []

for ingredient in direct_ingredients:

    # 한국어 별칭인지 확인
    actual = alias_to_ingredient.get(ingredient)

    if actual:
        converted_direct.append(actual)

    elif ingredient in set(ingredient_list):
        converted_direct.append(ingredient)

    else:
        # 일반적인 영어 성분 이름은 그대로 유지
        converted_direct.append(ingredient)


# 검색 선택 + 직접 입력 성분 합치기
combined_ingredients = list(
    dict.fromkeys(
        selected_ingredients + converted_direct
    )
)

# 학습 데이터에 존재하는 성분과 그렇지 않은 성분 구분
valid_ingredients = [
    ingredient
    for ingredient in combined_ingredients
    if ingredient in trained_ingredients
]

excluded_ingredients = [
    ingredient
    for ingredient in combined_ingredients
    if ingredient not in trained_ingredients
]


# ---------------------------------------------------------
# 12. 선택한 성분 확인
# ---------------------------------------------------------

st.divider()
st.subheader("5. 최종 선택 성분 확인")

if combined_ingredients:

    st.write(
        f"총 {len(combined_ingredients)}개의 성분을 입력했습니다."
    )

    st.markdown("**입력한 성분 목록**")

    st.write(
        ", ".join(
            get_ingredient_display(x)
            for x in combined_ingredients
        )
    )

    st.markdown("**모델에 반영 가능한 성분**")

    if valid_ingredients:
        st.success(
            f"{len(valid_ingredients)}개 성분이 예측에 사용됩니다."
        )

        st.write(
            ", ".join(valid_ingredients)
        )
    else:
        st.warning(
            "학습 데이터에서 확인된 성분이 없습니다."
        )

    if excluded_ingredients:
        st.warning(
            "다음 성분은 학습 데이터에 없어서 예측에 반영되지 않습니다."
        )

        st.write(
            ", ".join(excluded_ingredients)
        )

else:
    st.info(
        "아직 선택하거나 입력한 성분이 없습니다."
    )


# ---------------------------------------------------------
# 13. 제품 유형 예측
# ---------------------------------------------------------

st.divider()
st.subheader("6. 제품 유형 예측")

if st.button(
    "🔍 제품 유형 예측하기",
    type="primary",
    use_container_width=True
):

    if not combined_ingredients:
        st.warning(
            "먼저 성분을 검색하거나 직접 입력해 주세요."
        )

    elif not valid_ingredients:
        st.warning(
            "모델이 학습한 성분이 하나도 없습니다. "
            "데이터에 존재하는 성분을 선택해 주세요."
        )

    else:

        # 학습 때 사용한 MultiLabelBinarizer로 변환
        input_encoded = mlb.transform(
            [valid_ingredients]
        )

        # 예측
        prediction = model.predict(input_encoded)[0]

        # 제품 유형별 예측 확률
        probabilities = model.predict_proba(input_encoded)[0]

        probability_df = pd.DataFrame({
            "제품 유형": model.classes_,
            "예측 확률(%)": probabilities * 100
        })

        probability_df = probability_df.sort_values(
            "예측 확률(%)",
            ascending=False
        ).reset_index(drop=True)

        # 결과 저장
        st.session_state["prediction_result"] = prediction
        st.session_state["probability_result"] = probability_df
        st.session_state["prediction_ingredients"] = valid_ingredients
        st.session_state["excluded_result"] = excluded_ingredients


# ---------------------------------------------------------
# 14. 예측 결과 출력
# ---------------------------------------------------------

if "prediction_result" in st.session_state:

    st.divider()
    st.subheader("예측 결과")

    prediction = st.session_state["prediction_result"]
    probability_df = st.session_state["probability_result"]

    st.success("예측이 완료되었습니다.")

    st.markdown("### 가장 가능성이 높은 제품 유형")

    st.markdown(
        f"<h2 style='text-align:center; color:#2E8B57;'>"
        f"{prediction}</h2>",
        unsafe_allow_html=True
    )

    st.write(
        "입력한 전성분을 바탕으로 랜덤 포레스트 모델이 "
        "가장 높은 확률로 예측한 제품 유형입니다."
    )

    st.markdown("### 제품 유형별 예측 확률")

    chart_df = probability_df.head(10).sort_values(
        "예측 확률(%)",
        ascending=True
    )

    st.bar_chart(
        chart_df.set_index("제품 유형")["예측 확률(%)"],
        horizontal=True
    )

    st.dataframe(
        probability_df.style.format({
            "예측 확률(%)": "{:.2f}%"
        }),
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "예측 확률은 모델이 각 제품 유형에 부여한 상대적인 확률이며, "
        "실제 예측 정확도를 의미하지는 않습니다."
    )

    if st.session_state["excluded_result"]:
        st.warning(
            "학습 데이터에 없는 성분은 예측에서 제외되었습니다: "
            + ", ".join(st.session_state["excluded_result"])
        )


# ---------------------------------------------------------
# 15. 모델 성능 확인
# ---------------------------------------------------------

st.divider()
st.subheader("7. 모델 성능 확인")

st.write(
    "모델이 학습에 사용하지 않은 테스트 데이터를 얼마나 "
    "잘 분류했는지 확인합니다."
)

metric1, metric2, metric3 = st.columns(3)

with metric1:
    st.metric(
        "정확도 (Accuracy)",
        f"{result['accuracy'] * 100:.2f}%"
    )

with metric2:
    st.metric(
        "학습 데이터 수",
        f"{result['train_rows']:,}개"
    )

with metric3:
    st.metric(
        "테스트 데이터 수",
        f"{result['test_rows']:,}개"
    )

st.write(
    f"전체 학습 가능 제품 수: {result['total_rows']:,}개"
)

st.write(
    f"분류한 제품 유형 수: {result['category_count']}개"
)

if result["stratified"]:
    st.caption(
        "제품 유형별 비율을 고려한 층화 분할을 적용했습니다."
    )
else:
    st.caption(
        "일부 제품 유형의 데이터가 적어 일반 무작위 분할을 적용했습니다."
    )

if result["unseen_categories"]:
    st.warning(
        "테스트 데이터에 학습 데이터에서 볼 수 없었던 제품 유형이 있습니다. "
        "해당 유형은 모델이 학습하지 못했으므로 성능표에서 별도로 평가되지 않습니다. "
        "정확도에는 이 유형의 오분류가 반영됩니다."
    )

st.markdown("### 제품 유형별 정밀도와 재현율")

report_df = result["report"].copy()

# 제품 유형 행만 추출
category_rows = [
    category
    for category in result["classes"]
    if str(category) in report_df.index
]

if category_rows:

    display_report = report_df.loc[
        [str(category) for category in category_rows],
        ["precision", "recall", "f1-score", "support"]
    ].copy()

    display_report.columns = [
        "정밀도 (Precision)",
        "재현율 (Recall)",
        "F1 점수",
        "테스트 데이터 수"
    ]

    st.dataframe(
        display_report.style.format({
            "정밀도 (Precision)": "{:.3f}",
            "재현율 (Recall)": "{:.3f}",
            "F1 점수": "{:.3f}",
            "테스트 데이터 수": "{:.0f}"
        }),
        use_container_width=True
    )

else:
    st.info(
        "제품 유형별 성능을 표시할 수 있는 테스트 데이터가 부족합니다."
    )

st.markdown("### 성능 지표 알아보기")

with st.expander("정확도, 정밀도, 재현율이란?"):

    st.write(
        "**정확도 (Accuracy)**\n\n"
        "전체 테스트 데이터 중 모델이 정답을 맞힌 비율입니다."
    )

    st.write(
        "**정밀도 (Precision)**\n\n"
        "모델이 특정 제품 유형이라고 예측한 것 중 실제로 그 유형인 비율입니다."
    )

    st.write(
        "**재현율 (Recall)**\n\n"
        "실제로 특정 제품 유형인 데이터 중 모델이 해당 유형이라고 찾아낸 비율입니다."
    )

    st.write(
        "**F1 점수**\n\n"
        "정밀도와 재현율을 함께 고려한 지표입니다."
    )


# ---------------------------------------------------------
# 16. 마지막 안내
# ---------------------------------------------------------

st.divider()

st.info(
    "이 프로그램은 화장품 전성분의 포함 여부를 바탕으로 "
    "제품 유형을 분류하는 학습용 프로그램입니다. "
    "실제 화장품의 용도나 효과를 확정하는 진단 도구는 아닙니다."
)
