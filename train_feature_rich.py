"""
Feature-Rich Training Script for Product Price Prediction
--------------------------------------------------------
This script builds an extensive feature set from product names, combining
hand-crafted categorical indicators, statistical descriptors, unit parsing,
color/gender cues, and powerful text embeddings (TF-IDF + SVD). It then trains
an ensemble of models (Ridge on sparse text representations and
HistGradientBoosting on dense engineered features) to minimize SMAPE.

Outputs
=======
1. training_log_*.txt              -- full training log with timestamps
2. training_results_feature_rich.json -- validation metrics for each model
3. submission_feature_rich.csv     -- final predictions ready for Kaggle submission
4. category_summary.csv            -- price statistics per detected category
5. feature_price_correlation.csv   -- correlation of numeric features with price
6. feature_importance_gradient_boosting.csv -- feature importances from GBoost
7. feature_sample.csv              -- sample of engineered features (first 200 rows)

The script assumes train.csv, test.csv, sample_submission.csv are present.
"""

import json
import math
import os
import re
import sys
import time
import zlib
from collections import Counter
from datetime import datetime

import jieba
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# ------------------------------
# Configuration & Logging
# ------------------------------
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

LOG_FILE = None

def log_print(message: str) -> None:
    """Print to console and append to log file."""
    print(message)
    if LOG_FILE is not None:
        LOG_FILE.write(message + "\n")
        LOG_FILE.flush()


def format_seconds(seconds: float) -> str:
    """Return human-readable time string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.2f}h"


# ------------------------------
# Utility Functions
# ------------------------------

def smape(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_pred = np.clip(y_pred, 1.0, None)
    numerator = np.abs(y_true - y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    with np.errstate(divide="ignore", invalid="ignore"):
        smape_values = numerator / denominator
    smape_values[denominator == 0] = 0
    return 100 * np.nanmean(smape_values)


def stable_hash(token: str, modulo: int = 2048) -> int:
    return zlib.crc32(token.encode("utf-8")) % modulo


def quick_tokenize(text: str):
    if not text:
        return []
    return re.findall(r"[A-Za-z]+|\d+(?:\.\d+)?|[\u4e00-\u9fff]+", text)


# ------------------------------
# Dictionaries for Feature Extraction
# ------------------------------
CATEGORY_KEYWORDS = {
    "food_staple": ["米", "飯", "麵", "粽", "粥", "麥", "穀", "油"],
    "food_snack": ["餅", "糖", "餅乾", "零食", "巧克力", "堅果", "點心"],
    "food_dessert": ["蛋糕", "甜", "布丁", "派", "冰", "冰淇淋"],
    "drink": ["飲", "茶", "咖啡", "汁", "酒", "奶", "粉", "豆漿"],
    "beauty": ["面膜", "乳", "霜", "精華", "化妝", "美容", "保養", "洗髮"],
    "household": ["清潔", "洗衣", "垃圾", "衛生", "紙", "拖把", "掃把"],
    "kitchen": ["鍋", "盤", "碗", "筷", "餐具", "烤盤", "廚"],
    "furniture": ["沙發", "床", "椅", "桌", "櫃", "收納", "衣櫥"],
    "appliance": ["電鍋", "微波", "冰箱", "洗衣機", "冷氣", "烤箱", "吹風機"],
    "electronics": ["手機", "電腦", "筆電", "耳機", "音響", "充電", "USB", "Wifi", "攝影"],
    "stationery": ["筆", "紙", "筆記", "本", "書寫", "文具", "資料夾"],
    "pet": ["狗", "貓", "寵", "犬", "貓砂", "飼料", "貓狗"],
    "baby": ["嬰", "寶寶", "幼兒", "奶瓶", "尿布", "成長", "學習"],
    "clothing": ["衣", "褲", "裙", "外套", "鞋", "襪", "帽", "洋裝", "T恤"],
    "sport": ["球", "運動", "健身", "瑜伽", "登山", "跑步", "自行車"],
    "automotive": ["車", "汽車", "機車", "輪胎", "機油", "汽油", "車架"],
    "garden": ["花", "植", "園藝", "盆栽", "肥料", "土", "種子"],
    "hardware": ["螺絲", "工具", "五金", "刀", "剪", "鋸", "鉗"],
    "festival": ["春聯", "燈籠", "禮盒", "贈品", "節", "年", "中秋", "端午", "聖誕"],
    "health": ["保健", "維生素", "補充", "藥", "醫", "護", "血", "體重"],
    "finance": ["禮券", "禮卡", "儲值", "票券", "gift", "voucher"],
    "books": ["書", "漫畫", "雜誌", "小說", "教材"],
    "travel": ["行李", "旅行", "背包", "露營", "旅遊", "裝備"],
}
CATEGORY_LIST = sorted(CATEGORY_KEYWORDS.keys()) + ["other"]
CATEGORY_INDEX = {cat: idx for idx, cat in enumerate(CATEGORY_LIST)}

UNIT_PATTERNS = {
    "weight": ["公斤", "kg", "公克", "克", "g", "mg", "磅", "斤"],
    "volume": ["毫升", "ml", "公升", "l", "cc", "oz"],
    "length": ["公分", "cm", "毫米", "mm", "公尺", "m", "inch", "吋"],
    "area": ["平方", "坪", "㎡", "m2"],
    "temperature": ["℃", "度c", "°c"],
    "quantity": ["入", "包", "盒", "瓶", "罐", "袋", "支", "顆", "台", "雙", "件", "片", "條", "組"],
    "currency": ["元", "nt$", "n.t", "$"],
}

COLOR_KEYWORDS = [
    "紅", "橙", "黃", "綠", "藍", "紫", "黑", "白", "灰", "棕", "咖啡", "粉", "金", "銀",
    "red", "orange", "yellow", "green", "blue", "purple", "black", "white", "grey", "gray",
    "brown", "pink", "gold", "silver", "navy", "ivory"
]

GENDER_KEYWORDS = {
    "female": ["女", "女性", "女生", "婦", "女士", "girl", "woman", "women", "ladies"],
    "male": ["男", "男性", "男生", "紳士", "男士", "boy", "man", "men", "gentleman"],
}

AGE_KEYWORDS = {
    "baby": ["嬰", "幼兒", "寶寶", "新生", "幼"],
    "kids": ["童", "兒童", "小孩", "學童", "國小"],
    "teen": ["青少年", "teen"],
    "adult": ["成人", "大人"],
    "senior": ["銀髮", "長者", "老人", "高齡"],
}

SEASON_KEYWORDS = {
    "spring": ["春", "spring"],
    "summer": ["夏", "summer"],
    "autumn": ["秋", "autumn"],
    "winter": ["冬", "winter"],
}

FESTIVAL_KEYWORDS = {
    "chinese_new_year": ["新年", "過年", "春節"],
    "dragon_boat": ["端午", "龍舟", "粽"],
    "mid_autumn": ["中秋", "月餅"],
    "christmas": ["聖誕", "christmas"],
    "valentine": ["情人", "valentine", "七夕"],
}

SIZE_KEYWORDS = ["XS", "S", "M", "L", "XL", "2L", "XXL", "3XL", "4XL", "5XL", "6XL"]
IMPORT_KEYWORDS = ["日本", "韓國", "美國", "德國", "法國", "義大利", "進口", "原裝"]
DISCOUNT_KEYWORDS = ["折", "SALE", "特價", "優惠", "買一送一", "送", "折扣"]
LUXURY_KEYWORDS = [
    "鑽石", "戒指", "黃金", "金條", "金飾", "金器", "錶", "手錶", "腕錶", "名錶",
    "相機", "單眼", "鏡頭", "攝影機", "珠寶", "寶石", "首飾", "精品", "名牌", "奢華"
]
MODEL_PATTERN = re.compile(r"[A-Za-z]{1,4}-?\d{2,}|\d{2,}[A-Za-z]{1,4}")
HASH_CODE_PATTERN = re.compile(r"#[0-9A-Za-z\-]+")
DIMENSION_PATTERN = re.compile(r"(\d+(?:\.\d+)?)\s?(cm|公分|mm|毫米|m|公尺|inch|吋)", re.IGNORECASE)
WEIGHT_PATTERN = re.compile(r"(\d+(?:\.\d+)?)\s?(kg|公斤|g|公克|mg|磅|斤)", re.IGNORECASE)
VOLUME_PATTERN = re.compile(r"(\d+(?:\.\d+)?)\s?(ml|毫升|l|公升|cc|oz)", re.IGNORECASE)


def make_hgb(**kwargs) -> HistGradientBoostingRegressor:
    params = dict(random_state=RANDOM_STATE)
    params.update(kwargs)
    return HistGradientBoostingRegressor(**params)


def make_gb(**kwargs) -> GradientBoostingRegressor:
    params = dict(random_state=RANDOM_STATE)
    params.update(kwargs)
    return GradientBoostingRegressor(**params)


# ------------------------------
# Feature Extraction Helpers
# ------------------------------

def build_character_features(names: pd.Series) -> pd.DataFrame:
    log_print("   > Building character-level features...")
    series = names.fillna("")
    char_counts = series.str.len().astype(np.int32)
    spaces = series.str.count("\s")
    digits = series.str.count(r"\d")
    alpha = series.str.count(r"[A-Za-z]")
    upper = series.str.count(r"[A-Z]")
    lower = series.str.count(r"[a-z]")
    chinese = series.str.count(r"[\u4e00-\u9fff]")
    symbols = char_counts - (spaces + digits + alpha + chinese)
    punctuation = series.str.count(r"[\!\?\.,;:]")

    features = pd.DataFrame({
        "name_length": char_counts,
        "char_no_space": (char_counts - spaces).astype(np.int32),
        "space_count": spaces.astype(np.int32),
        "digit_count": digits.astype(np.int32),
        "alpha_count": alpha.astype(np.int32),
        "alpha_upper_count": upper.astype(np.int32),
        "alpha_lower_count": lower.astype(np.int32),
        "chinese_count": chinese.astype(np.int32),
        "symbol_count": symbols.astype(np.int32),
        "punctuation_count": punctuation.astype(np.int32),
        "exclamation_count": series.str.count("!").astype(np.int32),
        "question_count": series.str.count("\?").astype(np.int32),
        "hash_count": series.str.count("#").astype(np.int32),
        "slash_count": series.str.count(r"/").astype(np.int32),
        "dash_count": series.str.count(r"-").astype(np.int32),
        "plus_count": series.str.count(r"\+").astype(np.int32),
        "ampersand_count": series.str.count(r"&").astype(np.int32),
        "digit_ratio": np.where(char_counts > 0, digits / char_counts, 0.0),
        "alpha_ratio": np.where(char_counts > 0, alpha / char_counts, 0.0),
        "chinese_ratio": np.where(char_counts > 0, chinese / char_counts, 0.0),
        "symbol_ratio": np.where(char_counts > 0, symbols / char_counts, 0.0),
    }, dtype=np.float32)

    first_chars = series.str[0].fillna("")
    features["starts_with_digit"] = first_chars.str.contains(r"^\d").astype(np.int32)
    features["starts_with_alpha"] = first_chars.str.contains(r"^[A-Za-z]").astype(np.int32)
    features["starts_with_chinese"] = first_chars.str.contains(r"^[\u4e00-\u9fff]").astype(np.int32)
    features["starts_with_symbol"] = (~(
        features["starts_with_digit"].astype(bool) |
        features["starts_with_alpha"].astype(bool) |
        features["starts_with_chinese"].astype(bool)
    )).astype(np.int32)

    last_chars = series.str[-1].fillna("")
    features["ends_with_digit"] = last_chars.str.contains(r"\d$").astype(np.int32)
    features["ends_with_alpha"] = last_chars.str.contains(r"[A-Za-z]$").astype(np.int32)
    features["ends_with_chinese"] = last_chars.str.contains(r"[\u4e00-\u9fff]$").astype(np.int32)

    unique_chars = series.apply(lambda x: len(set(x)) if x else 0)
    features["unique_char_count"] = unique_chars.astype(np.int32)
    features["unique_char_ratio"] = np.where(char_counts > 0, unique_chars / char_counts, 0.0)

    return features.astype(np.float32)


def build_numeric_features(names: pd.Series) -> pd.DataFrame:
    log_print("   > Parsing numeric patterns...")
    series = names.fillna("")
    number_lists = series.str.findall(r"\d+(?:\.\d+)?")

    def convert_numbers(lst):
        if not lst:
            return []
        try:
            return [float(x) for x in lst]
        except ValueError:
            converted = []
            for x in lst:
                try:
                    converted.append(float(x))
                except ValueError:
                    continue
            return converted

    numbers = number_lists.apply(convert_numbers)
    counts = number_lists.apply(len).astype(np.int32)

    sums = numbers.apply(sum)
    maxs = numbers.apply(lambda x: max(x) if x else 0.0)
    mins = numbers.apply(lambda x: min(x) if x else 0.0)
    means = numbers.apply(lambda x: np.mean(x) if x else 0.0)
    medians = numbers.apply(lambda x: np.median(x) if x else 0.0)
    stds = numbers.apply(lambda x: np.std(x) if len(x) > 1 else 0.0)
    ranges = numbers.apply(lambda x: max(x) - min(x) if len(x) >= 2 else 0.0)
    sums_log = numbers.apply(lambda x: sum(np.log1p(x)) if x else 0.0)

    decimal_counts = number_lists.apply(lambda lst: sum("." in item for item in lst)).astype(np.int32)
    ge_1000 = numbers.apply(lambda x: sum(1 for n in x if n >= 1000))
    years = numbers.apply(lambda x: sum(1 for n in x if 1900 <= n <= 2035))
    two_digit = numbers.apply(lambda x: sum(1 for n in x if 10 <= n < 100))
    three_digit = numbers.apply(lambda x: sum(1 for n in x if 100 <= n < 1000))
    unique_counts = number_lists.apply(lambda lst: len(set(lst)))

    digit_lengths = number_lists.apply(lambda lst: [len(x.replace('.', '')) for x in lst])
    max_digit_len = digit_lengths.apply(lambda lst: max(lst) if lst else 0)
    avg_digit_len = digit_lengths.apply(lambda lst: np.mean(lst) if lst else 0.0)

    features = pd.DataFrame({
        "number_count": counts,
        "number_sum": sums,
        "number_max": maxs,
        "number_min": mins,
        "number_mean": means,
        "number_median": medians,
        "number_std": stds,
        "number_range": ranges,
        "number_log_sum": sums_log,
        "number_decimal_count": decimal_counts,
        "number_ge_1000": np.array(ge_1000, dtype=np.int32),
        "number_year_count": np.array(years, dtype=np.int32),
        "number_two_digit": np.array(two_digit, dtype=np.int32),
        "number_three_digit": np.array(three_digit, dtype=np.int32),
        "number_unique_count": np.array(unique_counts, dtype=np.int32),
        "number_max_digit_len": np.array(max_digit_len, dtype=np.int32),
        "number_avg_digit_len": avg_digit_len.astype(np.float32),
        "number_density": np.where(names.str.len() > 0, counts / names.str.len(), 0.0),
    }, dtype=np.float32)

    half_pattern = series.str.contains(r"\d+/\d+")
    percent_pattern = series.str.contains(r"%")
    features["number_fraction_flag"] = half_pattern.astype(np.int32)
    features["number_percentage_flag"] = percent_pattern.astype(np.int32)
    features["number_series_flag"] = (counts >= 3).astype(np.int32)

    return features


def build_unit_features(names: pd.Series) -> pd.DataFrame:
    log_print("   > Detecting measurement units...")
    series = names.fillna("").str.lower()
    data = {}
    total_counts = np.zeros(len(series), dtype=np.float32)

    for unit_type, patterns in UNIT_PATTERNS.items():
        escaped = [re.escape(p.lower()) for p in patterns]
        pattern = "|".join(escaped)
        counts = series.str.count(pattern, flags=re.IGNORECASE)
        data[f"unit_{unit_type}_count"] = counts.astype(np.float32)
        data[f"unit_{unit_type}_flag"] = (counts > 0).astype(np.int32)
        total_counts += counts.values.astype(np.float32)

    data["unit_total_count"] = total_counts
    data["unit_diversity"] = np.array([
        sum(1 for unit_type in UNIT_PATTERNS if data[f"unit_{unit_type}_flag"][idx])
        for idx in range(len(series))
    ], dtype=np.float32)

    def aggregate_measurements(pattern, converter):
        res = []
        for name in names.fillna(""):
            matches = pattern.findall(name)
            if not matches:
                res.append([])
            else:
                converted = []
                for value, unit in matches:
                    converted.append(converter(float(value), unit))
                res.append(converted)
        return res

    def to_cm(value, unit):
        unit = unit.lower()
        if unit in {"cm", "公分"}:
            return value
        if unit in {"mm", "毫米"}:
            return value / 10
        if unit in {"m", "公尺"}:
            return value * 100
        if unit in {"inch", "吋"}:
            return value * 2.54
        return value

    def to_kg(value, unit):
        unit = unit.lower()
        if unit in {"kg", "公斤"}:
            return value
        if unit in {"g", "公克"}:
            return value / 1000
        if unit in {"mg"}:
            return value / 1_000_000
        if unit in {"磅"}:
            return value * 0.453592
        if unit in {"斤"}:
            return value * 0.6
        return value

    def to_liter(value, unit):
        unit = unit.lower()
        if unit in {"l", "公升"}:
            return value
        if unit in {"ml", "毫升", "cc"}:
            return value / 1000
        if unit in {"oz"}:
            return value * 0.0295735
        return value

    dims = aggregate_measurements(DIMENSION_PATTERN, to_cm)
    weights = aggregate_measurements(WEIGHT_PATTERN, to_kg)
    volumes = aggregate_measurements(VOLUME_PATTERN, to_liter)

    def stats(lst):
        if not lst:
            return 0.0, 0.0, 0.0, 0.0
        arr = np.array(lst, dtype=np.float32)
        return arr.mean(), arr.max(), arr.min(), arr.std() if len(arr) > 1 else 0.0

    dim_mean, dim_max, dim_min, dim_std = zip(*(stats(x) for x in dims))
    wt_mean, wt_max, wt_min, wt_std = zip(*(stats(x) for x in weights))
    vol_mean, vol_max, vol_min, vol_std = zip(*(stats(x) for x in volumes))

    data.update({
        "dimension_mean_cm": np.array(dim_mean, dtype=np.float32),
        "dimension_max_cm": np.array(dim_max, dtype=np.float32),
        "dimension_min_cm": np.array(dim_min, dtype=np.float32),
        "dimension_std_cm": np.array(dim_std, dtype=np.float32),
        "dimension_count": np.array([len(x) for x in dims], dtype=np.float32),
        "weight_mean_kg": np.array(wt_mean, dtype=np.float32),
        "weight_max_kg": np.array(wt_max, dtype=np.float32),
        "weight_min_kg": np.array(wt_min, dtype=np.float32),
        "weight_std_kg": np.array(wt_std, dtype=np.float32),
        "weight_count": np.array([len(x) for x in weights], dtype=np.float32),
        "volume_mean_l": np.array(vol_mean, dtype=np.float32),
        "volume_max_l": np.array(vol_max, dtype=np.float32),
        "volume_min_l": np.array(vol_min, dtype=np.float32),
        "volume_std_l": np.array(vol_std, dtype=np.float32),
        "volume_count": np.array([len(x) for x in volumes], dtype=np.float32),
    })

    return pd.DataFrame(data).astype(np.float32)


def build_color_gender_features(names: pd.Series) -> pd.DataFrame:
    log_print("   > Capturing color, gender, age, season signals...")
    series = names.fillna("")
    lowered = series.str.lower()

    color_pattern = "|".join(re.escape(c.lower()) for c in COLOR_KEYWORDS)
    color_counts = lowered.str.count(color_pattern)
    dominant_colors = series.apply(lambda x: next((c for c in COLOR_KEYWORDS if c.lower() in x.lower()), "none"))

    gender_features = {}
    for gender, keywords in GENDER_KEYWORDS.items():
        pattern = "|".join(re.escape(k.lower()) for k in keywords)
        counts = lowered.str.count(pattern)
        gender_features[f"gender_{gender}_count"] = counts.astype(np.float32)
        gender_features[f"gender_{gender}_flag"] = (counts > 0).astype(np.int32)

    age_features = {}
    for age_group, keywords in AGE_KEYWORDS.items():
        pattern = "|".join(re.escape(k.lower()) for k in keywords)
        counts = lowered.str.count(pattern)
        age_features[f"age_{age_group}_count"] = counts.astype(np.float32)
        age_features[f"age_{age_group}_flag"] = (counts > 0).astype(np.int32)

    season_features = {}
    for season, keywords in SEASON_KEYWORDS.items():
        pattern = "|".join(re.escape(k.lower()) for k in keywords)
        counts = lowered.str.count(pattern)
        season_features[f"season_{season}_count"] = counts.astype(np.float32)
        season_features[f"season_{season}_flag"] = (counts > 0).astype(np.int32)

    festival_features = {}
    for fest, keywords in FESTIVAL_KEYWORDS.items():
        pattern = "|".join(re.escape(k.lower()) for k in keywords)
        counts = lowered.str.count(pattern)
        festival_features[f"festival_{fest}_count"] = counts.astype(np.float32)
        festival_features[f"festival_{fest}_flag"] = (counts > 0).astype(np.int32)

    import_counts = lowered.str.count("|".join(re.escape(k.lower()) for k in IMPORT_KEYWORDS))
    discount_counts = lowered.str.count("|".join(re.escape(k.lower()) for k in DISCOUNT_KEYWORDS))
    size_counts = lowered.str.count("|".join(re.escape(k.lower()) for k in SIZE_KEYWORDS))
    luxury_counts = lowered.str.count("|".join(re.escape(k.lower()) for k in LUXURY_KEYWORDS))

    features = {
        "color_hit_count": color_counts.astype(np.float32),
        "color_flag": (color_counts > 0).astype(np.int32),
        "dominant_color": dominant_colors,
        "import_flag": (import_counts > 0).astype(np.int32),
        "discount_flag": (discount_counts > 0).astype(np.int32),
        "size_keyword_flag": (size_counts > 0).astype(np.int32),
        "luxury_flag": (luxury_counts > 0).astype(np.int32),
        "luxury_keyword_count": luxury_counts.astype(np.float32),
        "model_code_flag": series.apply(lambda x: int(bool(MODEL_PATTERN.search(x)))).astype(np.int32),
        "hash_code_flag": series.apply(lambda x: int(bool(HASH_CODE_PATTERN.search(x)))).astype(np.int32),
    }
    features.update(gender_features)
    features.update(age_features)
    features.update(season_features)
    features.update(festival_features)
    return pd.DataFrame(features)


def build_keyword_features(names: pd.Series) -> pd.DataFrame:
    log_print("   > Building token/keyword structural features...")
    series = names.fillna("")
    tokens = series.apply(quick_tokenize)

    token_lengths = tokens.apply(lambda lst: [len(tok) for tok in lst])
    token_count = tokens.apply(len)
    unique_token_count = tokens.apply(lambda lst: len(set(lst)))

    avg_token_len = token_lengths.apply(lambda lst: np.mean(lst) if lst else 0.0)
    max_token_len = token_lengths.apply(lambda lst: max(lst) if lst else 0)
    min_token_len = token_lengths.apply(lambda lst: min(lst) if lst else 0)

    numeric_token_count = tokens.apply(lambda lst: sum(tok.isdigit() for tok in lst))
    alpha_token_count = tokens.apply(lambda lst: sum(tok.isalpha() for tok in lst))
    chinese_token_count = tokens.apply(lambda lst: sum(bool(re.fullmatch(r"[\u4e00-\u9fff]+", tok)) for tok in lst))

    first_token = tokens.apply(lambda lst: lst[0].lower() if lst else "")
    last_token = tokens.apply(lambda lst: lst[-1].lower() if lst else "")
    second_token = tokens.apply(lambda lst: lst[1].lower() if len(lst) > 1 else "")

    features = pd.DataFrame({
        "token_count": token_count.astype(np.int32),
        "token_unique_count": unique_token_count.astype(np.int32),
        "token_unique_ratio": np.where(token_count > 0, unique_token_count / token_count, 0.0),
        "token_numeric_count": numeric_token_count.astype(np.int32),
        "token_alpha_count": alpha_token_count.astype(np.int32),
        "token_chinese_count": chinese_token_count.astype(np.int32),
        "token_avg_len": avg_token_len.astype(np.float32),
        "token_max_len": max_token_len.astype(np.int32),
        "token_min_len": min_token_len.astype(np.int32),
    })

    features["first_token_hash"] = first_token.apply(lambda x: stable_hash(x, 2048)).astype(np.int32)
    features["last_token_hash"] = last_token.apply(lambda x: stable_hash(x, 2048)).astype(np.int32)
    features["second_token_hash"] = second_token.apply(lambda x: stable_hash(x, 2048)).astype(np.int32)

    uppercase_tokens = series.apply(lambda x: re.findall(r"[A-Z]{2,}", x))
    features["uppercase_token_count"] = uppercase_tokens.apply(len).astype(np.int32)
    features["uppercase_token_avg_len"] = uppercase_tokens.apply(lambda lst: np.mean([len(tok) for tok in lst]) if lst else 0.0)

    bundle_words = series.str.count(r"組|set|套|雙|對", flags=re.IGNORECASE)
    gift_words = series.str.count(r"禮|贈|gift|present", flags=re.IGNORECASE)
    limited_words = series.str.count(r"限量|限定|首發|獨家", flags=re.IGNORECASE)

    features["bundle_flag"] = (bundle_words > 0).astype(np.int32)
    features["gift_flag"] = (gift_words > 0).astype(np.int32)
    features["limited_flag"] = (limited_words > 0).astype(np.int32)

    return features.astype(np.float32)


def build_category_features(names: pd.Series) -> pd.DataFrame:
    log_print("   > Deriving category-based features...")
    series = names.fillna("")
    lowered = series.str.lower()

    cat_scores = []
    for cat in CATEGORY_KEYWORDS:
        keywords = CATEGORY_KEYWORDS[cat]
        pattern = "|".join(re.escape(kw.lower()) for kw in keywords)
        counts = lowered.str.count(pattern)
        cat_scores.append(counts.astype(np.float32))
    cat_scores_df = pd.DataFrame({cat: cat_scores[idx] for idx, cat in enumerate(CATEGORY_KEYWORDS)}, dtype=np.float32)

    score_values = cat_scores_df.values
    score_sums = score_values.sum(axis=1, keepdims=True)

    with np.errstate(divide="ignore", invalid="ignore"):
        proportions = np.divide(score_values, score_sums, where=score_sums != 0)
    proportions[np.isnan(proportions)] = 0.0

    entropy = -np.sum(proportions * np.log(proportions + 1e-12), axis=1)
    multi_hits = (score_values > 0).sum(axis=1)

    top_two = np.partition(score_values, -2, axis=1)[:, -2:]
    top1 = top_two[:, 1]
    top2 = top_two[:, 0]

    primary_idx = np.argmax(score_values, axis=1)
    primary_scores = top1
    secondary_idx = np.argsort(score_values, axis=1)[:, -2]
    secondary_scores = np.take_along_axis(score_values, np.expand_dims(secondary_idx, axis=1), axis=1).flatten()

    primary_idx = np.where(score_sums.flatten() == 0, len(CATEGORY_LIST) - 1, primary_idx)
    secondary_idx = np.where(score_sums.flatten() == 0, len(CATEGORY_LIST) - 1, secondary_idx)
    primary_scores = np.where(score_sums.flatten() == 0, 0, primary_scores)
    secondary_scores = np.where(score_sums.flatten() == 0, 0, secondary_scores)

    cat_flags = (score_values > 0).astype(np.int32)

    features = cat_scores_df.copy()
    for idx, cat in enumerate(CATEGORY_KEYWORDS):
        features[f"cat_flag_{cat}"] = cat_flags[:, idx]

    features["category_hit_count"] = multi_hits.astype(np.float32)
    features["category_score_sum"] = score_sums.flatten().astype(np.float32)
    features["category_entropy"] = entropy.astype(np.float32)
    features["primary_category_idx"] = primary_idx.astype(np.int32)
    features["primary_category"] = [CATEGORY_LIST[idx] for idx in primary_idx]
    features["primary_category_score"] = primary_scores.astype(np.float32)
    features["secondary_category_idx"] = secondary_idx.astype(np.int32)
    features["secondary_category"] = [CATEGORY_LIST[idx] for idx in secondary_idx]
    features["secondary_category_score"] = secondary_scores.astype(np.float32)
    features["category_score_ratio"] = np.where(score_sums.flatten() > 0, primary_scores / (score_sums.flatten() + 1e-6), 0.0)
    features["category_gap"] = (primary_scores - secondary_scores).astype(np.float32)
    features["category_confidence"] = np.where(primary_scores > 0, (primary_scores - secondary_scores) / (primary_scores + 1e-6), 0.0)
    features["category_has_multiple"] = (multi_hits >= 2).astype(np.int32)
    features["category_hit_ratio"] = multi_hits / len(CATEGORY_KEYWORDS)

    return features


def assemble_features(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    start = time.time()
    log_print(f"\n[FEATURE] Assembling engineered features for {dataset_name} ...")
    names = df["name"].fillna("")

    char_df = build_character_features(names)
    numeric_df = build_numeric_features(names)
    unit_df = build_unit_features(names)
    keyword_df = build_keyword_features(names)
    cat_df = build_category_features(names)
    cg_df = build_color_gender_features(names)

    features = pd.concat([char_df, numeric_df, unit_df, keyword_df, cat_df, cg_df], axis=1)
    features.fillna(0, inplace=True)

    log_print(f"   Features assembled: {features.shape[1]} columns")
    log_print(f"   Time spent: {format_seconds(time.time() - start)}")
    return features


def clean_for_tfidf(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\u4e00-\u9fff#+/\-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_with_jieba(text: str) -> str:
    return " ".join(jieba.lcut(text))


def safe_truncated_svd(matrix, n_components: int, random_state: int):
    max_components = min(n_components, matrix.shape[1]-1, matrix.shape[0]-1)
    if max_components < 5:
        max_components = min(5, matrix.shape[1])
    svd = TruncatedSVD(n_components=max_components, random_state=random_state)
    reduced = svd.fit_transform(matrix)
    return svd, reduced


# ------------------------------
# Main Pipeline
# ------------------------------

def main():
    global LOG_FILE

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"training_feature_rich_{timestamp}.log"
    LOG_FILE = open(log_filename, "w", encoding="utf-8")

    global_start = time.time()
    log_print("=" * 90)
    log_print("FEATURE-RICH PRICE PREDICTION PIPELINE")
    log_print("=" * 90)
    log_print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_print(f"Log file : {log_filename}")

    # 1. Load Data
    step_start = time.time()
    log_print("\n[1/9] Loading data ...")
    train_df = pd.read_csv("train.csv").reset_index(drop=True)
    test_df = pd.read_csv("test.csv").reset_index(drop=True)
    sample_submission = pd.read_csv("sample_submission.csv")
    log_print(f"   train shape: {train_df.shape}, test shape: {test_df.shape}")
    log_print(f"   price range: {train_df['price'].min()} - {train_df['price'].max()}")
    log_print(f"   price mean : {train_df['price'].mean():.2f}, median: {train_df['price'].median():.2f}")
    log_print(f"   Step time  : {format_seconds(time.time() - step_start)}")

    # 2. Feature Engineering
    step_start = time.time()
    train_features = assemble_features(train_df, "train")
    test_features = assemble_features(test_df, "test")
    log_print(f"   train features: {train_features.shape}, test features: {test_features.shape}")
    log_print(f"   Step time  : {format_seconds(time.time() - step_start)}")

    # Keep categorical columns for analysis, exclude from numeric modeling
    categorical_cols = [col for col in train_features.columns if train_features[col].dtype == "object"]
    log_print(f"   categorical feature columns: {categorical_cols}")

    # 3. Clean text & tokenize for TF-IDF
    step_start = time.time()
    log_print("\n[3/9] Cleaning and tokenizing text ...")
    tqdm.pandas(desc="clean_text")
    train_clean = train_df["name"].fillna("").progress_apply(clean_for_tfidf)
    test_clean = test_df["name"].fillna("").progress_apply(clean_for_tfidf)

    tqdm.pandas(desc="jieba")
    train_tokenized = train_clean.progress_apply(tokenize_with_jieba)
    test_tokenized = test_clean.progress_apply(tokenize_with_jieba)
    log_print(f"   Step time  : {format_seconds(time.time() - step_start)}")

    # 4. Vectorization
    step_start = time.time()
    log_print("\n[4/9] Building TF-IDF representations ...")

    word_vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=5,
        max_features=20000,
        sublinear_tf=True
    )
    char_vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 4),
        min_df=5,
        max_features=12000,
        sublinear_tf=True
    )

    log_print("   > Fitting word-level TF-IDF ...")
    X_word = word_vectorizer.fit_transform(train_tokenized)
    X_word_test = word_vectorizer.transform(test_tokenized)

    log_print("   > Fitting char-level TF-IDF ...")
    X_char = char_vectorizer.fit_transform(train_clean)
    X_char_test = char_vectorizer.transform(test_clean)

    word_components = 150 if X_word.shape[1] > 150 else max(10, X_word.shape[1] - 1)
    char_components = 100 if X_char.shape[1] > 100 else max(10, X_char.shape[1] - 1)

    log_print(f"   > Reducing word TF-IDF to {word_components} components via SVD ...")
    svd_word, word_reduced = safe_truncated_svd(X_word, word_components, RANDOM_STATE)
    word_reduced_test = svd_word.transform(X_word_test)

    log_print(f"   > Reducing char TF-IDF to {char_components} components via SVD ...")
    svd_char, char_reduced = safe_truncated_svd(X_char, char_components, RANDOM_STATE)
    char_reduced_test = svd_char.transform(X_char_test)

    log_print(f"   Step time  : {format_seconds(time.time() - step_start)}")

    # 5. Prepare numeric/dense matrices
    step_start = time.time()
    log_print("\n[5/9] Preparing dense feature matrices ...")
    numeric_cols = [col for col in train_features.columns if col not in categorical_cols]
    train_numeric = train_features[numeric_cols].astype(np.float32).values
    test_numeric = test_features[numeric_cols].astype(np.float32).values

    scaler = StandardScaler(with_mean=True)
    train_numeric_scaled = scaler.fit_transform(train_numeric)
    test_numeric_scaled = scaler.transform(test_numeric)

    dense_train = np.hstack([train_numeric_scaled, word_reduced, char_reduced])
    dense_test = np.hstack([test_numeric_scaled, word_reduced_test, char_reduced_test])

    sparse_train = hstack([X_word, X_char]).tocsr()
    sparse_test = hstack([X_word_test, X_char_test]).tocsr()
    log_print(f"   dense feature shape: {dense_train.shape}")
    log_print(f"   sparse feature shape: {sparse_train.shape}")
    log_print(f"   Step time  : {format_seconds(time.time() - step_start)}")

    # 6. Train/Validation Split
    step_start = time.time()
    log_print("\n[6/9] Creating validation split ...")
    y = train_df["price"].values
    price_bins = pd.qcut(train_df["price"], q=20, labels=False, duplicates="drop")
    train_idx, val_idx = train_test_split(
        np.arange(len(train_df)),
        test_size=0.15,
        random_state=RANDOM_STATE,
        stratify=price_bins
    )

    y_train, y_val = y[train_idx], y[val_idx]
    dense_train_train, dense_val = dense_train[train_idx], dense_train[val_idx]
    sparse_train_train, sparse_val = sparse_train[train_idx], sparse_train[val_idx]

    log_print(f"   train size: {len(train_idx)}, validation size: {len(val_idx)}")
    log_print(f"   Step time : {format_seconds(time.time() - step_start)}")

    # 7. Model Training & Validation
    results = {}
    results_detail = {}

    log_print("\n[7/9] Training models ...")
    # Ridge on sparse TF-IDF
    step_start = time.time()
    log_print("   > Ridge Regression (TF-IDF sparse) ...")
    ridge = Ridge(alpha=3.0, random_state=RANDOM_STATE)
    ridge.fit(sparse_train_train, y_train)
    ridge_val_pred = np.clip(ridge.predict(sparse_val), 1.0, None)
    ridge_train_pred = np.clip(ridge.predict(sparse_train_train), 1.0, None)

    ridge_metrics = {
        "train_smape": smape(y_train, ridge_train_pred),
        "val_smape": smape(y_val, ridge_val_pred),
        "val_mae": mean_absolute_error(y_val, ridge_val_pred),
        "val_rmse": math.sqrt(mean_squared_error(y_val, ridge_val_pred)),
    }
    log_print(f"     Ridge train SMAPE: {ridge_metrics['train_smape']:.4f}%")
    log_print(f"     Ridge val   SMAPE: {ridge_metrics['val_smape']:.4f}%")
    log_print(f"     Ridge val   MAE  : {ridge_metrics['val_mae']:.4f}")
    log_print(f"     Ridge val   RMSE : {ridge_metrics['val_rmse']:.4f}")
    log_print(f"     Time taken      : {format_seconds(time.time() - step_start)}")
    results['ridge'] = ridge_metrics
    results_detail['ridge'] = {
        "model_type": "ridge",
        "model": ridge,
    }

    # Tree-based models on dense engineered features with log target
    tree_model_factories = {
        "hist_gb_light": lambda: make_hgb(
            learning_rate=0.08,
            max_depth=10,
            max_leaf_nodes=63,
            min_samples_leaf=15,
            l2_regularization=0.5,
            early_stopping=True,
            max_iter=300,
            validation_fraction=0.1,
            n_iter_no_change=20
        ),
        "hist_gb_deep": lambda: make_hgb(
            learning_rate=0.06,
            max_depth=16,
            max_leaf_nodes=127,
            min_samples_leaf=10,
            l2_regularization=1.0,
            early_stopping=True,
            max_iter=500,
            validation_fraction=0.1,
            n_iter_no_change=30
        ),
        "hist_gb_shallow": lambda: make_hgb(
            learning_rate=0.12,
            max_depth=8,
            max_leaf_nodes=45,
            min_samples_leaf=25,
            l2_regularization=0.3,
            early_stopping=True,
            max_iter=250,
            validation_fraction=0.1,
            n_iter_no_change=15
        ),
        "hist_gb_stable": lambda: make_hgb(
            learning_rate=0.09,
            max_depth=12,
            max_leaf_nodes=95,
            min_samples_leaf=18,
            l2_regularization=0.8,
            early_stopping=True,
            max_iter=350,
            validation_fraction=0.1,
            n_iter_no_change=25
        ),
        "hist_gb_wide": lambda: make_hgb(
            learning_rate=0.05,
            max_depth=18,
            max_leaf_nodes=255,
            min_samples_leaf=8,
            l2_regularization=1.2,
            early_stopping=True,
            max_iter=600,
            validation_fraction=0.15,
            n_iter_no_change=40
        ),
        "hist_gb_fast": lambda: make_hgb(
            learning_rate=0.15,
            max_depth=6,
            max_leaf_nodes=40,
            min_samples_leaf=20,
            l2_regularization=0.2,
            early_stopping=True,
            max_iter=180,
            validation_fraction=0.1,
            n_iter_no_change=10
        ),
        "hist_gb_regularized": lambda: make_hgb(
            learning_rate=0.07,
            max_depth=14,
            max_leaf_nodes=150,
            min_samples_leaf=12,
            l2_regularization=2.0,
            early_stopping=True,
            max_iter=500,
            validation_fraction=0.12,
            n_iter_no_change=30
        ),
        "hist_gb_balanced": lambda: make_hgb(
            learning_rate=0.065,
            max_depth=12,
            max_leaf_nodes=90,
            min_samples_leaf=16,
            l2_regularization=0.9,
            early_stopping=True,
            max_iter=420,
            validation_fraction=0.1,
            n_iter_no_change=20
        ),
        "hist_gb_ultra": lambda: make_hgb(
            learning_rate=0.04,
            max_depth=20,
            max_leaf_nodes=320,
            min_samples_leaf=8,
            l2_regularization=1.8,
            early_stopping=True,
            max_iter=700,
            validation_fraction=0.15,
            n_iter_no_change=40
        ),
        "hist_gb_extreme": lambda: make_hgb(
            learning_rate=0.03,
            max_depth=24,
            max_leaf_nodes=420,
            min_samples_leaf=6,
            l2_regularization=2.5,
            early_stopping=True,
            max_iter=900,
            validation_fraction=0.18,
            n_iter_no_change=50
        ),
        "hist_gb_lowlr": lambda: make_hgb(
            learning_rate=0.02,
            max_depth=18,
            max_leaf_nodes=256,
            min_samples_leaf=5,
            l2_regularization=2.0,
            early_stopping=True,
            max_iter=1100,
            validation_fraction=0.2,
            n_iter_no_change=60
        )
    }

    tree_models = {name: factory() for name, factory in tree_model_factories.items()}

    tree_predictions_val = {}
    tree_predictions_train = {}

    y_train_log = np.log1p(y_train)
    val_predictions_log = {}

    total_tree_models = len(tree_models)
    for idx, (name, model) in enumerate(tree_models.items(), start=1):
        step_start = time.time()
        log_print(f"   > ({idx}/{total_tree_models}) Training {name} ...")
        model.fit(dense_train_train, y_train_log)
        val_pred_log = model.predict(dense_val)
        train_pred_log = model.predict(dense_train_train)
        val_predictions_log[name] = val_pred_log
        val_pred = np.clip(np.expm1(val_pred_log), 1.0, None)
        train_pred = np.clip(np.expm1(train_pred_log), 1.0, None)

        metrics = {
            "train_smape": smape(y_train, train_pred),
            "val_smape": smape(y_val, val_pred),
            "val_mae": mean_absolute_error(y_val, val_pred),
            "val_rmse": math.sqrt(mean_squared_error(y_val, val_pred))
        }
        results[name] = metrics
        tree_predictions_val[name] = val_pred
        tree_predictions_train[name] = train_pred
        results_detail[name] = {
            "model_type": "tree",
            "factory": tree_model_factories[name],
        }

        log_print(f"     train SMAPE: {metrics['train_smape']:.4f}%")
        log_print(f"     val   SMAPE: {metrics['val_smape']:.4f}%")
        log_print(f"     val   MAE  : {metrics['val_mae']:.4f}")
        log_print(f"     val   RMSE : {metrics['val_rmse']:.4f}")
        log_print(f"     Time taken : {format_seconds(time.time() - step_start)}")

    # Blends combining ridge and top tree models
    blend_configs = {
        "blend_ridge_hist_light": (0.55, "hist_gb_light"),
        "blend_ridge_hist_deep": (0.50, "hist_gb_deep"),
        "blend_ridge_hist_shallow": (0.60, "hist_gb_shallow"),
        "blend_ridge_hist_stable": (0.53, "hist_gb_stable"),
        "blend_ridge_hist_wide": (0.48, "hist_gb_wide"),
        "blend_ridge_hist_fast": (0.58, "hist_gb_fast"),
        "blend_ridge_hist_regularized": (0.52, "hist_gb_regularized"),
        "blend_ridge_hist_balanced": (0.54, "hist_gb_balanced"),
        "blend_ridge_hist_ultra": (0.47, "hist_gb_ultra"),
        "blend_ridge_hist_extreme": (0.45, "hist_gb_extreme"),
        "blend_ridge_hist_lowlr": (0.40, "hist_gb_lowlr"),
    }

    total_blends = len(blend_configs)
    for idx, (blend_name, (ridge_weight, tree_name)) in enumerate(blend_configs.items(), start=1):
        if tree_name not in tree_predictions_val:
            continue
        tree_weight = 1.0 - ridge_weight
        val_pred = ridge_weight * ridge_val_pred + tree_weight * tree_predictions_val[tree_name]
        train_pred = ridge_weight * ridge_train_pred + tree_weight * tree_predictions_train[tree_name]

        metrics = {
            "train_smape": smape(y_train, train_pred),
            "val_smape": smape(y_val, val_pred),
            "val_mae": mean_absolute_error(y_val, val_pred),
            "val_rmse": math.sqrt(mean_squared_error(y_val, val_pred))
        }
        results[blend_name] = metrics
        results_detail[blend_name] = {
            "model_type": "blend",
            "ridge_weight": ridge_weight,
            "tree_name": tree_name,
        }
        log_print(f"\n   > Blend ({idx}/{total_blends}) {blend_name}: ridge {ridge_weight:.2f} + {tree_name} {tree_weight:.2f}")
        log_print(f"     train SMAPE: {metrics['train_smape']:.4f}%")
        log_print(f"     val   SMAPE: {metrics['val_smape']:.4f}%")
        log_print(f"     val   MAE  : {metrics['val_mae']:.4f}")
        log_print(f"     val   RMSE : {metrics['val_rmse']:.4f}")

    best_name = min(results, key=lambda k: results[k]['val_smape'])
    best_metrics = results[best_name]
    log_print(f"\n[RESULT] Best validation model: {best_name} (SMAPE {best_metrics['val_smape']:.4f}%)")
    results['_best_model'] = {
        "name": best_name,
        "metrics": best_metrics,
    }

    with open("training_results_feature_rich.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # 8. Train on full data
    step_start = time.time()
    log_print("\n[8/9] Training on full dataset ...")
    best_detail = results_detail[best_name]

    if best_detail['model_type'] == 'ridge':
        log_print("   Best model is Ridge. Training on full data ...")
        ridge_full = Ridge(alpha=3.0, random_state=RANDOM_STATE)
        ridge_full.fit(sparse_train, y)
        final_test_pred = np.clip(ridge_full.predict(sparse_test), 1.0, None)
    elif best_detail['model_type'] == 'tree':
        log_print(f"   Best model is tree-based ({best_name}). Training on full data ...")
        tree_full = best_detail['factory']()
        tree_full.fit(dense_train, np.log1p(y))
        final_test_pred = np.clip(np.expm1(tree_full.predict(dense_test)), 1.0, None)
    elif best_detail['model_type'] == 'blend':
        log_print(f"   Best model is blend ({best_name}). Training components on full data ...")
        ridge_weight = best_detail['ridge_weight']
        tree_name = best_detail['tree_name']
        tree_factory = tree_model_factories[tree_name]

        ridge_full = Ridge(alpha=3.0, random_state=RANDOM_STATE)
        ridge_full.fit(sparse_train, y)
        ridge_test_pred = np.clip(ridge_full.predict(sparse_test), 1.0, None)

        tree_full = tree_factory()
        tree_full.fit(dense_train, np.log1p(y))
        tree_test_pred = np.clip(np.expm1(tree_full.predict(dense_test)), 1.0, None)

        final_test_pred = ridge_weight * ridge_test_pred + (1.0 - ridge_weight) * tree_test_pred
    else:
        raise ValueError(f"Unknown model type for {best_name}")

    submission = pd.DataFrame({
        "name": test_df["name"],
        "price": final_test_pred,
    })
    submission.to_csv("submission_feature_rich.csv", index=False)
    log_print(f"   Submission saved -> submission_feature_rich.csv")
    log_print(f"   Step time  : {format_seconds(time.time() - step_start)}")

    # 9. Analytics & Feature Importance
    step_start = time.time()
    log_print("\n[9/9] Generating analytics & summaries ...")

    category_summary = train_df.assign(primary_category=train_features["primary_category"]).groupby("primary_category")['price'].agg(["count", "mean", "median", "std", "min", "max"]).sort_values("mean", ascending=False)
    category_summary.to_csv("category_summary.csv")
    log_print("   category_summary.csv written")

    feature_numeric_cols = numeric_cols
    corr_series = train_features[feature_numeric_cols].corrwith(train_df['price'])
    corr_series.sort_values(key=lambda s: np.abs(s), ascending=False).to_csv("feature_price_correlation.csv", header=['correlation'])
    log_print("   feature_price_correlation.csv written")

    # Gradient Boosting feature importances (dense data)
    gbr_imp = make_gb(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8
    )
    gbr_imp.fit(dense_train, y)
    dense_feature_names = [f"num_{col}" for col in numeric_cols] + [f"svd_word_{i}" for i in range(word_reduced.shape[1])] + [f"svd_char_{i}" for i in range(char_reduced.shape[1])]
    gbr_importance = pd.DataFrame({
        "feature": dense_feature_names,
        "importance": gbr_imp.feature_importances_
    }).sort_values("importance", ascending=False)
    log_print("   feature_importance_gradient_boosting.csv written (top 150 features)")

    train_features.head(200).to_csv("feature_sample.csv", index=False)
    log_print("   feature_sample.csv written (first 200 rows)")

    # Save metadata
    metadata = {
        "numeric_feature_count": len(numeric_cols),
        "categorical_columns": categorical_cols,
        "svd_word_components": word_reduced.shape[1],
        "svd_char_components": char_reduced.shape[1],
        "best_model": results['_best_model'],
        "models": results,
    }
    with open("feature_engineering_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    log_print("   feature_engineering_metadata.json written")
    log_print(f"   Step time  : {format_seconds(time.time() - step_start)}")

    # Closing remarks
    total_time = format_seconds(time.time() - global_start)
    log_print("\n" + "=" * 90)
    log_print("PIPELINE COMPLETE")
    log_print(f"Total runtime: {total_time}")
    log_print("Outputs:")
    log_print("  - submission_feature_rich.csv")
    log_print("  - training_results_feature_rich.json")
    log_print("  - category_summary.csv")
    log_print("  - feature_price_correlation.csv")
    log_print("  - feature_importance_gradient_boosting.csv")
    log_print("  - feature_sample.csv")
    log_print("  - feature_engineering_metadata.json")
    log_print(f"  - {log_filename}")
    log_print("=" * 90)

    if LOG_FILE is not None:
        LOG_FILE.close()


if __name__ == "__main__":
    main()
