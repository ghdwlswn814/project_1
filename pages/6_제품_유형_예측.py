import math
import re

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.naive_bayes import MultinomialNB
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
    "화장품 전성분을 검색하거나 직접 입력하면 제품 유형을 예측합니다. "
    "여러 분류 모델의 성능을 비교한 뒤, 학습 데이터 검증에서 "
    "가장 좋은 결과를 보인 모델을 사용합니다."
)

DATA_URL = (
    "https://raw.githubusercontent.com/LadyFelicia/"
    "Sephora-Analysis/main/Product%20Info.csv"
)


# ---------------------------------------------------------
# 2. 자주 사용되는 성분 이름과 설명
# ---------------------------------------------------------
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
# 4. 여러 모델을 교차 검증하고 최적 모델 학습
# ---------------------------------------------------------
@st.cache_resource(show_spinner="화장품 데이터를 불러오고 여러 예측 모델의 성능을 비교하고 있습니다...")
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

    # 성분 사전은 학습 데이터에서만 만들고, 테스트 데이터는 변환만 합니다.
    mlb = MultiLabelBinarizer()
    X_train_encoded = mlb.fit_transform(X_train)
    X_test_encoded = mlb.transform(X_test)

    if len(mlb.classes_) == 0:
        raise ValueError("학습 데이터에서 사용할 수 있는 성분을 찾지 못했습니다.")
    if y_train.nunique() < 2:
        raise ValueError("학습 데이터에 제품 유형이 2개 이상 필요합니다.")

    # 클래스별 데이터가 충분하면 층화 교차 검증을 사용합니다.
    # 일부 유형의 데이터가 너무 적으면 일반 K-Fold로 비교하고 안내합니다.
    min_train_class_count = int(y_train.value_counts().min())
    if min_train_class_count >= 2:
        n_splits = min(3, min_train_class_count)
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        cv_splitter_name = f"층화 {n_splits}-겹 교차 검증"
    else:
        n_splits = min(3, len(X_train))
        if n_splits < 2:
            raise ValueError("모델 비교에 필요한 학습 데이터가 부족합니다.")
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        cv_splitter_name = f"일반 {n_splits}-겹 교차 검증"

    model_candidates = {
        "랜덤 포레스트": RandomForestClassifier(
            n_estimators=180,
            random_state=42,
            class_weight="balanced_subsample",
            max_features="sqrt",
            n_jobs=-1,
        ),
        "로지스틱 회귀": LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            C=1.0,
            solver="lbfgs",
        ),
        "나이브 베이즈": MultinomialNB(
            alpha=0.5,
            fit_prior=False,
        ),
    }

    # 테스트 데이터는 모델 선택에 사용하지 않습니다.
    # 학습 데이터 내부의 교차 검증 Macro F1으로 모델을 선택합니다.
    cv_rows = []
    fitted_candidates = {}
    all_train_classes = sorted(y_train.unique().tolist())

    for model_name, estimator in model_candidates.items():
        fold_f1_scores = []
        fold_accuracy_scores = []
        model_failed = False

        for train_indices, valid_indices in cv.split(X_train_encoded, y_train):
            fold_model = clone(estimator)
            X_fold_train = X_train_encoded[train_indices]
            X_fold_valid = X_train_encoded[valid_indices]
            y_fold_train = y_train.iloc[train_indices]
            y_fold_valid = y_train.iloc[valid_indices]

            try:
                fold_model.fit(X_fold_train, y_fold_train)
                fold_pred = fold_model.predict(X_fold_valid)
                fold_f1_scores.append(
                    f1_score(
                        y_fold_valid,
                        fold_pred,
                        labels=all_train_classes,
                        average="macro",
                        zero_division=0,
                    )
                )
                fold_accuracy_scores.append(accuracy_score(y_fold_valid, fold_pred))
            except Exception:
                model_failed = True
                break

        if not model_failed and fold_f1_scores:
            mean_f1 = float(np.mean(fold_f1_scores))
            mean_accuracy = float(np.mean(fold_accuracy_scores))
            fitted_candidates[model_name] = estimator
            cv_rows.append(
                {
                    "모델": model_name,
                    "교차 검증 Macro F1": mean_f1,
                    "교차 검증 정확도": mean_accuracy,
                    "비교 결과": "비교 완료",
                }
            )
        else:
            cv_rows.append(
                {
                    "모델": model_name,
                    "교차 검증 Macro F1": np.nan,
                    "교차 검증 정확도": np.nan,
                    "비교 결과": "데이터 구성상 비교 실패",
                }
            )

    if not fitted_candidates:
        raise ValueError("모든 모델의 교차 검증에 실패했습니다. 제품 유형별 데이터 수를 확인해 주세요.")

    cv_results = pd.DataFrame(cv_rows)
    successful_results = cv_results.dropna(subset=["교차 검증 Macro F1"])
    best_model_name = successful_results.sort_values(
        ["교차 검증 Macro F1", "교차 검증 정확도"],
        ascending=False,
    ).iloc[0]["모델"]

    best_model = clone(fitted_candidates[best_model_name])
    best_model.fit(X_train_encoded, y_train)

    # 최종 모델을 한 번도 학습에 사용하지 않은 테스트 데이터로 평가합니다.
    y_pred = best_model.predict(X_test_encoded)
    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    report = classification_report(
        y_test,
        y_pred,
        labels=list(best_model.classes_),
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report).transpose()

    # 테스트 데이터에만 등장한 제품 유형은 모델이 학습할 수 없으므로 따로 안내합니다.
    unseen_test_categories = sorted(set(y_test.unique()) - set(best_model.classes_))

    # 검색 목록은 전체 유효 데이터에서 만들되, 예측 가능 여부는 학습 어휘로 판단합니다.
    all_ingredients = sorted(
        {
            ingredient
            for items in data["ingredient_list"]
            for ingredient in items
        }
    )

    # 혼동 행렬은 학습된 제품 유형을 기준으로 만듭니다.
    cm = confusion_matrix(y_test, y_pred, labels=list(best_model.classes_))
    cm_df = pd.DataFrame(
        cm,
        index=[f"실제: {name}" for name in best_model.classes_],
        columns=[f"예측: {name}" for name in best_model.classes_],
    )

    category_counts = data["category"].value_counts().rename_axis("제품 유형").reset_index(name="제품 수")

    return {
        "model": best_model,
        "best_model_name": best_model_name,
        "mlb": mlb,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "report": report_df,
        "cv_results": cv_results,
        "cv_splitter_name": cv_splitter_name,
        "ingredients": all_ingredients,
        "classes": list(best_model.classes_),
        "total_rows": len(data),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "category_count": y.nunique(),
        "unseen_categories": unseen_test_categories,
        "stratified": can_stratify,
        "confusion_matrix": cm_df,
        "category_counts": category_counts,
        "min_train_class_count": min_train_class_count,
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
    st.session_state["ingredient_selector"] = list(dict.fromkeys(selected))


def choose_moisture_example():
    set_example_selection("보습 성분 예시")


def choose_conditioning_example():
    set_example_selection("피부 컨디셔닝 성분 예시")


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
valid_ingredients = [item for item in combined_ingredients if item in trained_ingredient_set]
excluded_ingredients = [item for item in combined_ingredients if item not in trained_ingredient_set]


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
            "다음 성분은 학습 데이터에 없어 예측에 반영되지 않습니다: "
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

            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(input_encoded)[0]
                probability_df = pd.DataFrame(
                    {
                        "제품 유형": model.classes_,
                        "모델 점수(%)": probabilities * 100,
                    }
                ).sort_values("모델 점수(%)", ascending=False).reset_index(drop=True)
            else:
                probability_df = pd.DataFrame(
                    {"제품 유형": [prediction], "모델 점수(%)": [np.nan]}
                )

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

    st.write(f"사용한 모델: **{result['best_model_name']}**")
    st.write("입력한 전성분을 바탕으로 모델이 가장 높은 점수를 부여한 제품 유형입니다.")
    st.markdown("### 제품 유형별 모델 점수")

    if probability_df["모델 점수(%)"].notna().any():
        chart_df = probability_df.head(10).sort_values("모델 점수(%)", ascending=True)
        st.bar_chart(chart_df.set_index("제품 유형")["모델 점수(%)"], horizontal=True)
        st.dataframe(
            probability_df.style.format({"모델 점수(%)": "{:.2f}%"}),
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            "모델 점수는 모델이 계산한 분류 점수이며, 실제 제품이 해당 유형일 확률이나 "
            "예측 정확도를 보장하지 않습니다."
        )

    if st.session_state.get("prediction_excluded"):
        st.warning(
            "학습 데이터에 없어 예측에서 제외된 성분: "
            + ", ".join(st.session_state["prediction_excluded"])
        )


# ---------------------------------------------------------
# 13. 모델 성능 비교 및 평가
# ---------------------------------------------------------
st.divider()
st.subheader("7. 모델 성능 비교")
st.write(
    "랜덤 포레스트, 로지스틱 회귀, 나이브 베이즈를 학습 데이터 내부에서 "
    "교차 검증한 뒤 Macro F1 점수를 기준으로 최적 모델을 선택했습니다."
)

st.success(f"선택된 모델: {result['best_model_name']}")
st.caption(
    f"모델 선택 기준: {result['cv_splitter_name']}의 Macro F1 평균 점수. "
    "테스트 데이터는 모델 선택에 사용하지 않았습니다."
)

cv_display = result["cv_results"].copy()
cv_display.columns = ["모델", "교차 검증 Macro F1", "교차 검증 정확도", "비교 결과"]
st.dataframe(
    cv_display.style.format(
        {
            "교차 검증 Macro F1": "{:.3f}",
            "교차 검증 정확도": "{:.3f}",
        },
        na_rep="-",
    ),
    use_container_width=True,
    hide_index=True,
)
st.caption("Macro F1은 데이터가 적은 제품 유형도 큰 제품 유형과 같은 비중으로 평가하는 지표입니다.")

st.markdown("### 선택된 모델의 테스트 데이터 성능")
metric1, metric2, metric3 = st.columns(3)
metric1.metric("정확도 (Accuracy)", f"{result['accuracy'] * 100:.2f}%")
metric2.metric("Macro F1", f"{result['macro_f1']:.3f}")
metric3.metric("Weighted F1", f"{result['weighted_f1']:.3f}")

info1, info2, info3 = st.columns(3)
info1.metric("학습 데이터 수", f"{result['train_rows']:,}개")
info2.metric("테스트 데이터 수", f"{result['test_rows']:,}개")
info3.metric("제품 유형 수", f"{result['category_count']}개")

if result["stratified"]:
    st.caption("학습 데이터와 테스트 데이터는 제품 유형별 비율을 고려해 층화 분할했습니다.")
else:
    st.caption("일부 제품 유형의 데이터가 적어 학습/테스트 분할은 일반 무작위 분할을 적용했습니다.")

if result["min_train_class_count"] < 2:
    st.warning(
        "학습 데이터 중 제품 유형별 제품 수가 1개뿐인 유형이 있습니다. "
        "이 때문에 층화 교차 검증을 적용하기 어려워 일반 K-Fold로 모델을 비교했습니다. "
        "해당 유형의 데이터를 추가하면 평가가 더 안정적일 수 있습니다."
    )

if result["unseen_categories"]:
    st.warning(
        "테스트 데이터에 학습 데이터에서 보지 못한 제품 유형이 포함되어 있습니다. "
        "이 유형은 모델이 학습하지 못했으므로 예측할 수 없습니다: "
        + ", ".join(result["unseen_categories"])
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

with st.expander("혼동 행렬 보기"):
    st.write("행은 실제 제품 유형, 열은 모델이 예측한 제품 유형입니다.")
    st.dataframe(result["confusion_matrix"], use_container_width=True)

with st.expander("제품 유형별 데이터 개수 보기"):
    st.write("제품 유형별 데이터 개수가 크게 다르면 모델이 일부 유형을 더 잘 학습할 수 있습니다.")
    st.dataframe(result["category_counts"], use_container_width=True, hide_index=True)

with st.expander("평가 지표 설명"):
    st.write("**정확도 (Accuracy)**: 전체 테스트 데이터 중 모델이 정답을 맞힌 비율입니다.")
    st.write("**Macro F1**: 제품 유형별 F1 점수를 구한 뒤 동일한 비중으로 평균 낸 점수입니다.")
    st.write("**Weighted F1**: 제품 유형별 F1 점수를 제품 수에 따라 가중 평균한 점수입니다.")
    st.write("**정밀도 (Precision)**: 특정 유형이라고 예측한 제품 중 실제로 그 유형인 비율입니다.")
    st.write("**재현율 (Recall)**: 실제로 특정 유형인 제품 중 모델이 해당 유형이라고 찾아낸 비율입니다.")
    st.write("**F1 점수**: 정밀도와 재현율을 함께 고려한 지표입니다.")


# ---------------------------------------------------------
# 14. 안내
# ---------------------------------------------------------
st.divider()
st.info(
    "이 프로그램은 화장품 전성분에 포함된 성분을 바탕으로 제품 유형을 분류하는 "
    "학습용 프로그램입니다. 데이터 구성과 제품 유형별 표본 수에 따라 성능이 달라질 수 있으며, "
    "실제 제품의 용도나 효과를 확정하는 진단 도구는 아닙니다."
)
