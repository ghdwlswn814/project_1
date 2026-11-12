import math
import re

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer


# ---------------------------------------------------------
# 1. 페이지 설정
# ---------------------------------------------------------
st.set_page_config(
    page_title="제품 유형 예측",
    page_icon="🧴",
    layout="wide",
)

st.title("🧴 제품 유형 예측")
st.write(
    "화장품 전성분을 검색하거나 직접 입력하면 "
    "랜덤 포레스트 분류 모델이 제품 유형을 예측합니다."
)

DATA_URL = (
    "https://raw.githubusercontent.com/LadyFelicia/"
    "Sephora-Analysis/main/Product%20Info.csv"
)


# ---------------------------------------------------------
# 2. 자주 사용되는 성분 이름과 설명
# ---------------------------------------------------------
# 데이터에 실제로 존재하는 영어 성분명과 연결해서 사용합니다.
INGREDIENT_INFO = {
    "water": {
        "korean": "정제수",
        "description": "화장품의 기본적인 용매로 사용되는 성분",
    },
    "aqua": {
        "korean": "정제수",
        "description": "화장품의 기본적인 용매로 사용되는 성분",
    },
    "glycerin": {
        "korean": "글리세린",
        "description": "피부의 수분 유지에 도움을 주는 보습 성분",
    },
    "niacinamide": {
        "korean": "나이아신아마이드",
        "description": "피부 컨디셔닝에 사용되는 성분",
    },
    "butylene glycol": {
        "korean": "부틸렌글라이콜",
        "description": "보습 및 용매 역할을 하는 성분",
    },
    "panthenol": {
        "korean": "판테놀",
        "description": "피부 컨디셔닝과 보습에 사용되는 성분",
    },
}

KOREAN_ALIASES = {
    "정제수": ["water", "aqua"],
    "글리세린": ["glycerin"],
    "나이아신아마이드": ["niacinamide"],
    "부틸렌글라이콜": ["butylene glycol"],
    "판테놀": ["panthenol"],
}

EXAMPLE_SETS = {
    "보습 성분 예시": ["정제수", "글리세린", "부틸렌글라이콜"],
    "피부 컨디셔닝 성분 예시": ["나이아신아마이드", "판테놀", "글리세린"],
}


# ---------------------------------------------------------
# 3. 전성분 전처리 함수
# ---------------------------------------------------------
def normalize_ingredient(value):
    """성분 이름의 앞뒤 공백, 연속 공백, 대소문자를 정리합니다."""
    if pd.isna(value):
        return ""
    value = re.sub(r"\s+", " ", str(value).strip())
    return value.casefold()


def parse_ingredients(value):
    """쉼표로 구분된 전성분을 분리하고 중복을 제거합니다."""
    if pd.isna(value):
        return []

    parsed = []
    seen = set()
    for item in str(value).split(","):
        ingredient = normalize_ingredient(item)
        if ingredient and ingredient not in seen:
            parsed.append(ingredient)
            seen.add(ingredient)
    return parsed


def get_ingredient_display(ingredient):
    info = INGREDIENT_INFO.get(ingredient)
    if info:
        return f"{info['korean']} ({ingredient})"
    return ingredient


def get_ingredient_description(ingredient):
    info = INGREDIENT_INFO.get(ingredient)
    if info:
        return info["description"]
    return "성분 설명이 준비되지 않았습니다."


def find_actual_ingredient(korean_name, available_ingredients):
    """한국어 별칭에 연결된 성분 중 데이터에 실제로 있는 첫 성분을 찾습니다."""
    for candidate in KOREAN_ALIASES.get(korean_name, []):
        normalized = normalize_ingredient(candidate)
        if normalized in available_ingredients:
            return normalized
    return None


# ---------------------------------------------------------
# 4. 데이터 불러오기 및 모델 학습
# ---------------------------------------------------------
@st.cache_resource(show_spinner="화장품 데이터를 불러오고 예측 모델을 학습하고 있습니다...")
def load_and_train():
    df = pd.read_csv(DATA_URL, encoding="utf-8")

    required_columns = {"ingredients", "category"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(
            "데이터에 필요한 컬럼이 없습니다: " + ", ".join(sorted(missing_columns))
        )

    data = df[["ingredients", "category"]].copy()
    data = data.dropna(subset=["ingredients", "category"])
    data["category"] = data["category"].astype(str).str.strip()
    data = data[
        (data["category"] != "")
        & (data["category"].str.casefold() != "nan")
    ].copy()

    data["ingredient_list"] = data["ingredients"].apply(parse_ingredients)
    data = data[data["ingredient_list"].map(len) > 0].copy()

    if len(data) < 10:
        raise ValueError("학습에 사용할 수 있는 제품 데이터가 10개 미만입니다.")
    if data["category"].nunique() < 2:
        raise ValueError("분류할 제품 유형이 2개 미만입니다.")

    X = data["ingredient_list"]
    y = data["category"]

    test_count = math.ceil(len(data) * 0.2)
    train_count = len(data) - test_count
    class_count = y.nunique()
    counts = y.value_counts()

    can_stratify = (
        counts.min() >= 2
        and test_count >= class_count
        and train_count >= class_count
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y if can_stratify else None,
    )

    # 특징 사전은 학습 데이터로만 학습해 데이터 누수를 방지합니다.
    mlb = MultiLabelBinarizer()
    X_train_encoded = mlb.fit_transform(X_train)
    X_test_encoded = mlb.transform(X_test)

    if len(mlb.classes_) == 0:
        raise ValueError("학습 데이터에서 사용할 수 있는 성분을 찾지 못했습니다.")
    if y_train.nunique() < 2:
        raise ValueError("학습 데이터에 제품 유형이 2개 이상 필요합니다.")

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced_subsample",
        n_jobs=-1,
    )
    model.fit(X_train_encoded, y_train)

    y_pred = model.predict(X_test_encoded)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(
        y_test,
        y_pred,
        labels=list(model.classes_),
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report).transpose()

    # 검색 목록은 전체 유효 데이터에서 만들고, 실제 예측 가능 여부는
    # 학습 데이터에서 만들어진 mlb.classes_로 별도 판단합니다.
    all_ingredients = sorted(
        {
            ingredient
            for items in data["ingredient_list"]
            for ingredient in items
        }
    )

    unseen_test_categories = sorted(set(y_test.unique()) - set(model.classes_))

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
        "stratified": can_stratify,
    }


try:
    result = load_and_train()
except Exception as error:
    st.error("데이터를 불러오거나 모델을 학습하는 중 문제가 발생했습니다.")
    st.exception(error)
    st.info("데이터 주소, CSV 파일, 컬럼 이름을 확인한 뒤 다시 실행해 주세요.")
    st.stop()

model = result["model"]
mlb = result["mlb"]
ingredient_list = result["ingredients"]
all_ingredient_set = set(ingredient_list)
trained_ingredient_set = set(mlb.classes_)

# 한국어 별칭은 실제 데이터에 존재하는 성분에 대해서만 연결합니다.
alias_to_ingredient = {}
for korean_name in KOREAN_ALIASES:
    actual = find_actual_ingredient(korean_name, all_ingredient_set)
    if actual:
        alias_to_ingredient[korean_name] = actual


# ---------------------------------------------------------
# 5. 예시 버튼 콜백
# ---------------------------------------------------------
def set_example_selection(example_name):
    """multiselect 위젯이 다시 그려지기 전에 선택값을 바꿉니다."""
    selected = []
    for korean_name in EXAMPLE_SETS[example_name]:
        actual = find_actual_ingredient(korean_name, all_ingredient_set)
        if actual:
            selected.append(actual)

    # 위젯 key를 직접 수정하는 작업은 콜백 안에서만 수행합니다.
    st.session_state["ingredient_selector"] = list(dict.fromkeys(selected))


def choose_moisture_example():
    set_example_selection("보습 성분 예시")


def choose_conditioning_example():
    set_example_selection("피부 컨디셔닝 성분 예시")


# 위젯을 만들기 전에 상태를 초기화합니다.
if "ingredient_selector" not in st.session_state:
    st.session_state["ingredient_selector"] = []


# ---------------------------------------------------------
# 6. 성분 검색
# ---------------------------------------------------------
st.divider()
st.subheader("1. 성분 검색")
st.write("성분 이름의 일부를 입력하면 데이터에 실제로 포함된 성분을 검색합니다.")

search_word = st.text_input(
    "성분 검색",
    placeholder="예: water, glycerin, 정제수, 글리세린",
    key="ingredient_search",
)
search_normalized = normalize_ingredient(search_word)
search_target = alias_to_ingredient.get(search_word.strip(), search_normalized)

if search_normalized:
    filtered_ingredients = [
        ingredient
        for ingredient in ingredient_list
        if search_target in ingredient
        or search_normalized in ingredient
        or (
            INGREDIENT_INFO.get(ingredient, {}).get("korean", "")
            and search_normalized
            in normalize_ingredient(INGREDIENT_INFO[ingredient]["korean"])
        )
    ]
else:
    # 검색어가 없을 때 거대한 전체 목록이 펼쳐지지 않도록 검색을 유도합니다.
    filtered_ingredients = []

if search_normalized and not filtered_ingredients:
    st.info("검색 결과가 없습니다. 다른 성분 이름을 입력해 주세요.")
elif search_normalized:
    st.caption(f"검색 결과 {len(filtered_ingredients):,}개")


# ---------------------------------------------------------
# 7. 예시 성분 조합
# ---------------------------------------------------------
st.divider()
st.subheader("2. 예시 성분 조합")
st.caption(
    "아래 조합은 성분 선택을 연습하기 위한 예시이며, "
    "실제 특정 제품의 전성분을 의미하지 않습니다. "
    "데이터에 존재하지 않는 성분은 자동으로 제외됩니다."
)

col1, col2 = st.columns(2)
with col1:
    st.button(
        "💧 보습 성분 예시 선택",
        use_container_width=True,
        on_click=choose_moisture_example,
    )
with col2:
    st.button(
        "🌿 피부 컨디셔닝 성분 예시 선택",
        use_container_width=True,
        on_click=choose_conditioning_example,
    )


# ---------------------------------------------------------
# 8. 성분 여러 개 선택
# ---------------------------------------------------------
st.divider()
st.subheader("3. 검색 결과에서 성분 선택")

# 선택된 성분은 검색어가 바뀌어도 선택창에 계속 남도록 옵션에 합칩니다.
selected_before_widget = st.session_state["ingredient_selector"]
search_options = filtered_ingredients[:200]
available_options = sorted(set(search_options) | set(selected_before_widget))

selected_ingredients = st.multiselect(
    "사용할 성분을 선택하세요. 여러 개를 선택할 수 있습니다.",
    options=available_options,
    format_func=get_ingredient_display,
    key="ingredient_selector",
    help="성분을 검색한 뒤 목록에서 선택하세요. 검색어를 바꾸어도 기존 선택은 유지됩니다.",
)

if search_normalized and len(filtered_ingredients) > 200:
    st.caption("검색 결과가 많아 처음 200개만 선택 목록에 표시했습니다. 검색어를 더 구체적으로 입력해 주세요.")

if selected_ingredients:
    st.markdown("**선택한 성분의 설명**")
    for ingredient in selected_ingredients:
        st.write(
            f"**{get_ingredient_display(ingredient)}** — "
            f"{get_ingredient_description(ingredient)}"
        )


# ---------------------------------------------------------
# 9. 직접 전성분 입력
# ---------------------------------------------------------
st.divider()
st.subheader("4. 직접 전성분 입력")
st.write("성분 이름을 쉼표(,)로 구분해서 입력할 수 있습니다.")

direct_input = st.text_area(
    "전성분 입력",
    placeholder="예: Water, Glycerin, Niacinamide, Butylene Glycol, Panthenol",
    height=100,
    key="direct_ingredient_input",
)
st.caption("한국어 성분명도 일부 지원합니다. 예: 정제수, 글리세린, 나이아신아마이드")


def convert_user_ingredients(items):
    converted = []
    for ingredient in items:
        alias = alias_to_ingredient.get(ingredient)
        if alias:
            converted.append(alias)
        else:
            converted.append(ingredient)
    return list(dict.fromkeys(converted))


direct_ingredients = convert_user_ingredients(parse_ingredients(direct_input))
combined_ingredients = list(dict.fromkeys(selected_ingredients + direct_ingredients))
valid_ingredients = [
    item for item in combined_ingredients if item in trained_ingredient_set
]
excluded_ingredients = [
    item for item in combined_ingredients if item not in trained_ingredient_set
]


# ---------------------------------------------------------
# 10. 최종 선택 성분 확인
# ---------------------------------------------------------
st.divider()
st.subheader("5. 최종 선택 성분 확인")

if combined_ingredients:
    st.write(f"입력한 성분: **{len(combined_ingredients)}개**")
    st.write(", ".join(get_ingredient_display(item) for item in combined_ingredients))

    if valid_ingredients:
        st.success(f"{len(valid_ingredients)}개 성분이 모델 예측에 반영됩니다.")
        st.caption("모델 반영 성분: " + ", ".join(valid_ingredients))
    else:
        st.warning("학습 데이터에서 확인된 성분이 없습니다. 다른 성분을 선택하거나 입력해 주세요.")

    if excluded_ingredients:
        st.warning(
            "다음 성분은 학습 데이터에 없어서 예측에 반영되지 않습니다: "
            + ", ".join(excluded_ingredients)
        )
else:
    st.info("아직 선택하거나 입력한 성분이 없습니다.")


# ---------------------------------------------------------
# 11. 제품 유형 예측
# ---------------------------------------------------------
st.divider()
st.subheader("6. 제품 유형 예측")

if st.button("🔍 제품 유형 예측하기", type="primary", use_container_width=True):
    # 새 예측을 시작하면 이전 결과를 지웁니다.
    st.session_state.pop("prediction_result", None)
    st.session_state.pop("probability_result", None)
    st.session_state.pop("prediction_excluded", None)

    if not combined_ingredients:
        st.warning("먼저 성분을 검색하거나 직접 입력해 주세요.")
    elif not valid_ingredients:
        st.warning(
            "모델이 학습한 성분이 하나도 없습니다. "
            "검색 목록에서 학습에 사용된 성분을 선택해 주세요."
        )
    else:
        try:
            input_encoded = mlb.transform([valid_ingredients])
            prediction = model.predict(input_encoded)[0]
            probabilities = model.predict_proba(input_encoded)[0]

            probability_df = pd.DataFrame(
                {
                    "제품 유형": model.classes_,
                    "예측 확률(%)": probabilities * 100,
                }
            ).sort_values("예측 확률(%)", ascending=False).reset_index(drop=True)

            st.session_state["prediction_result"] = prediction
            st.session_state["probability_result"] = probability_df
            st.session_state["prediction_excluded"] = excluded_ingredients
        except Exception as error:
            st.error("예측 중 문제가 발생했습니다. 성분을 다시 확인해 주세요.")
            st.exception(error)


# ---------------------------------------------------------
# 12. 예측 결과 출력
# ---------------------------------------------------------
if "prediction_result" in st.session_state:
    st.divider()
    st.subheader("예측 결과")

    prediction = st.session_state["prediction_result"]
    probability_df = st.session_state["probability_result"]

    st.success("예측이 완료되었습니다.")
    st.markdown("### 가장 가능성이 높은 제품 유형")
    st.markdown(
        "<div style='text-align:center; padding:18px; border-radius:12px; "
        "background-color:rgba(46,139,87,0.10);'>"
        f"<span style='font-size:28px; font-weight:700;'>{prediction}</span>"
        "</div>",
        unsafe_allow_html=True,
    )

    st.write("입력한 전성분을 바탕으로 모델이 가장 높은 확률을 부여한 제품 유형입니다.")
    st.markdown("### 제품 유형별 예측 확률")

    chart_df = probability_df.head(10).sort_values("예측 확률(%)", ascending=True)
    st.bar_chart(
        chart_df.set_index("제품 유형")["예측 확률(%)"],
        horizontal=True,
    )
    st.dataframe(
        probability_df.style.format({"예측 확률(%)": "{:.2f}%"}),
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        "예측 확률은 모델이 각 유형에 부여한 값이며, "
        "실제 예측 정확도나 제품이 해당 유형일 확률을 보장하지는 않습니다."
    )

    if st.session_state.get("prediction_excluded"):
        st.warning(
            "학습 데이터에 없어 예측에서 제외된 성분: "
            + ", ".join(st.session_state["prediction_excluded"])
        )


# ---------------------------------------------------------
# 13. 모델 성능 확인
# ---------------------------------------------------------
st.divider()
st.subheader("7. 모델 성능 확인")
st.write("모델이 학습에 사용하지 않은 테스트 데이터로 성능을 확인합니다.")

metric1, metric2, metric3 = st.columns(3)
metric1.metric("정확도 (Accuracy)", f"{result['accuracy'] * 100:.2f}%")
metric2.metric("학습 데이터 수", f"{result['train_rows']:,}개")
metric3.metric("테스트 데이터 수", f"{result['test_rows']:,}개")

st.write(f"전체 학습 가능 제품 수: **{result['total_rows']:,}개**")
st.write(f"분류한 제품 유형 수: **{result['category_count']}개**")

if result["stratified"]:
    st.caption("제품 유형별 비율을 고려한 층화 분할을 적용했습니다.")
else:
    st.caption("일부 제품 유형의 데이터가 적어 일반 무작위 분할을 적용했습니다.")

if result["unseen_categories"]:
    st.warning(
        "테스트 데이터에 학습 데이터에서 보지 못한 제품 유형이 포함되어 있습니다. "
        "이 유형은 학습되지 않았으므로 제품 유형별 성능표에는 표시되지 않지만, "
        "정확도 계산에서는 오답으로 반영됩니다."
    )

st.markdown("### 제품 유형별 정밀도와 재현율")
report_df = result["report"]
category_rows = [str(category) for category in result["classes"] if str(category) in report_df.index]

if category_rows:
    display_report = report_df.loc[
        category_rows,
        ["precision", "recall", "f1-score", "support"],
    ].copy()
    display_report.columns = [
        "정밀도 (Precision)",
        "재현율 (Recall)",
        "F1 점수",
        "테스트 데이터 수",
    ]
    st.dataframe(
        display_report.style.format(
            {
                "정밀도 (Precision)": "{:.3f}",
                "재현율 (Recall)": "{:.3f}",
                "F1 점수": "{:.3f}",
                "테스트 데이터 수": "{:.0f}",
            }
        ),
        use_container_width=True,
    )
else:
    st.info("제품 유형별 성능을 표시할 수 있는 테스트 데이터가 부족합니다.")

with st.expander("정확도, 정밀도, 재현율이란?"):
    st.write("**정확도 (Accuracy)**: 전체 테스트 데이터 중 모델이 정답을 맞힌 비율입니다.")
    st.write("**정밀도 (Precision)**: 특정 유형이라고 예측한 제품 중 실제로 그 유형인 비율입니다.")
    st.write("**재현율 (Recall)**: 실제로 특정 유형인 제품 중 모델이 해당 유형이라고 찾아낸 비율입니다.")
    st.write("**F1 점수**: 정밀도와 재현율을 함께 고려한 지표입니다.")


# ---------------------------------------------------------
# 14. 안내
# ---------------------------------------------------------
st.divider()
st.info(
    "이 프로그램은 화장품 전성분에 포함된 성분을 바탕으로 제품 유형을 분류하는 "
    "학습용 프로그램입니다. 실제 제품의 용도나 효과를 확정하는 진단 도구는 아닙니다."
)
