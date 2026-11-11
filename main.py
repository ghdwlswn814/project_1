
import json
from pathlib import Path

import pandas as pd
import streamlit as st


# --------------------------------------------------
# 기본 설정
# --------------------------------------------------

st.set_page_config(
    page_title="화장품 성분 예측실",
    page_icon="🧴",
    layout="wide"
)

st.title("🧴 화장품 성분 예측실")
st.subheader("화장품 전성분 데이터 소개")
st.write(
    "화장품의 전성분 정보를 살펴보고, 성분을 바탕으로 "
    "제품 유형을 예측하기 위한 데이터입니다."
)

DATA_URL = (
    "https://raw.githubusercontent.com/"
    "LadyFelicia/Sephora-Analysis/main/Product%20Info.csv"
)

# 기록 파일 위치
BASE_DIR = Path(__file__).resolve().parent
RECORD_DIR = BASE_DIR / "내_기록"
RECORD_FILE = RECORD_DIR / "화장품_데이터_기록.json"

EXPECTED_COLUMNS = [
    "product_id",
    "product_name",
    "brand_id",
    "brand_name",
    "loves_count",
    "rating",
    "reviews",
    "size",
    "variation_value",
    "ingredients",
    "price_usd",
    "limited_edition",
    "new",
    "online_only",
    "sephora_exclusive",
    "highlights",
    "category",
    "child_count",
    "child_max_price",
    "child_min_price",
    "revenue",
]


# --------------------------------------------------
# 저장된 기록 불러오기
# --------------------------------------------------

def load_records():
    if not RECORD_FILE.exists():
        return {
            "meanings": {},
            "source": ""
        }

    try:
        with open(RECORD_FILE, "r", encoding="utf-8") as file:
            records = json.load(file)

        return {
            "meanings": records.get("meanings", {}),
            "source": records.get("source", "")
        }

    except (json.JSONDecodeError, OSError):
        return {
            "meanings": {},
            "source": ""
        }


# --------------------------------------------------
# 기록 저장하기
# --------------------------------------------------

def save_records(meanings, source):
    RECORD_DIR.mkdir(parents=True, exist_ok=True)

    records = {
        "meanings": meanings,
        "source": source
    }

    with open(RECORD_FILE, "w", encoding="utf-8") as file:
        json.dump(
            records,
            file,
            ensure_ascii=False,
            indent=4
        )


# --------------------------------------------------
# 데이터 불러오기
# --------------------------------------------------

@st.cache_data
def load_data():
    df = pd.read_csv(
        DATA_URL,
        encoding="utf-8",
        low_memory=False
    )

    return df


try:
    df = load_data()

except Exception as e:
    st.error("데이터를 불러오지 못했습니다.")
    st.write("인터넷 연결이나 데이터 주소를 확인해 주세요.")
    st.code(str(e))
    st.stop()


# --------------------------------------------------
# 데이터 열 확인
# --------------------------------------------------

missing_columns = [
    column for column in EXPECTED_COLUMNS
    if column not in df.columns
]

if missing_columns:
    st.warning(
        "CSV 파일에서 예상한 열을 찾지 못했습니다: "
        + ", ".join(missing_columns)
    )


# --------------------------------------------------
# 저장된 기록 읽기
# --------------------------------------------------

records = load_records()
saved_meanings = records["meanings"]
saved_source = records["source"]


# --------------------------------------------------
# 1. 데이터 기본 정보
# --------------------------------------------------

st.divider()
st.header("1. 데이터 기본 정보")

total_products = len(df)
column_count = len(df.columns)

if "category" in df.columns:
    category_count = df["category"].nunique()
else:
    category_count = 0

if "ingredients" in df.columns:
    ingredient_count = (
        df["ingredients"]
        .fillna("")
        .astype(str)
        .str.strip()
        .ne("")
        .sum()
    )
else:
    ingredient_count = 0


col1, col2, col3, col4 = st.columns(4)

col1.metric("전체 제품 수", f"{total_products:,}개")
col2.metric("열 개수", f"{column_count}개")
col3.metric("제품 유형 수", f"{category_count:,}개")
col4.metric("전성분 정보가 있는 제품", f"{ingredient_count:,}개")


# --------------------------------------------------
# 2. 데이터 열 정보
# --------------------------------------------------

st.divider()
st.header("2. 데이터 열 정보")

st.write(
    "각 열의 이름과 값의 종류, 빈 값 개수를 확인할 수 있습니다. "
    "우리말 뜻은 교재를 참고하여 직접 입력해 주세요."
)

metadata = []

for column in df.columns:
    dtype = str(df[column].dtype)
    unique_count = df[column].nunique(dropna=True)
    missing_count = int(df[column].isna().sum())

    if dtype == "object":
        value_kind = f"문자형 / 고유값 {unique_count:,}개"
    elif "int" in dtype:
        value_kind = f"정수형 / 고유값 {unique_count:,}개"
    elif "float" in dtype:
        value_kind = f"실수형 / 고유값 {unique_count:,}개"
    else:
        value_kind = f"{dtype} / 고유값 {unique_count:,}개"

    metadata.append({
        "열 이름": column,
        "우리말 뜻": saved_meanings.get(column, ""),
        "값의 종류": value_kind,
        "빈 값 개수": missing_count
    })

metadata_df = pd.DataFrame(metadata)

with st.form("record_form"):

    edited_metadata = st.data_editor(
        metadata_df,
        use_container_width=True,
        hide_index=True,
        num_rows="fixed",
        disabled=[
            "열 이름",
            "값의 종류",
            "빈 값 개수"
        ],
        column_config={
            "열 이름": st.column_config.TextColumn(
                "열 이름"
            ),
            "우리말 뜻": st.column_config.TextColumn(
                "우리말 뜻",
                help="교재를 참고하여 각 열의 뜻을 입력하세요.",
                width="large"
            ),
            "값의 종류": st.column_config.TextColumn(
                "값의 종류"
            ),
            "빈 값 개수": st.column_config.NumberColumn(
                "빈 값 개수",
                format="%d"
            )
        }
    )

    # --------------------------------------------------
    # 3. 데이터 출처
    # --------------------------------------------------

    st.divider()
    st.header("3. 데이터 출처")

    source_text = st.text_area(
        "데이터 출처를 입력하세요.",
        value=saved_source,
        placeholder="교재를 참고하여 데이터 출처를 입력하세요.",
        height=100
    )

    save_button = st.form_submit_button(
        "우리말 뜻과 데이터 출처 저장하기",
        use_container_width=True
    )

    if save_button:
        meanings = {}

        for _, row in edited_metadata.iterrows():
            meanings[row["열 이름"]] = row["우리말 뜻"]

        try:
            save_records(meanings, source_text)
            st.success("입력한 내용이 저장되었습니다.")
        except OSError as e:
            st.error("기록을 저장하지 못했습니다.")
            st.code(str(e))


# --------------------------------------------------
# 4. 원본 데이터 확인
# --------------------------------------------------

st.divider()
st.header("4. 원본 데이터 미리 보기")

st.write("CSV 파일에 저장된 데이터의 처음 다섯 줄입니다.")

st.dataframe(
    df.head(5),
    use_container_width=True,
    hide_index=True
)

st.caption(
    f"전체 {len(df):,}개 제품 중 처음 5개 제품을 표시하고 있습니다."
)


# --------------------------------------------------
# 마지막 안내
# --------------------------------------------------

st.divider()

st.info(
    "현재 화면은 화장품 데이터 소개 기능만 제공합니다. "
    "그래프와 분류 모델은 이후 단계에서 추가할 예정입니다."
)
