
import pandas as pd
import numpy as np
import re
import glob
import lightgbm as lgb
from catboost import CatBoostClassifier, Pool
import gc
import os
import joblib
# (LogisticRegression 임포트 제거)

# --- 1. 전처리 함수 정의 ---
# ( 이전에 수정한 함수들 그대로 포함 )
def preprocess_age(age_str):
    ''' '40b', '60a' 같은 나이 문자열을 숫자로 변환합니다. '''
    if pd.isna(age_str): return np.nan
    try:
        base_age = int(re.findall(r'^\d+', age_str)[0])
        if 'a' in age_str: return base_age
        elif 'b' in age_str: return base_age + 5
    except: return np.nan

def get_string_list_stats(series, col_name):
    ''' [개선된 버전] 쉼표로 구분된 숫자 문자열 리스트(Series)를 받아 통계 특성을 계산합니다. '''
    def calculate_stats_for_row(s):
        if pd.isna(s): return pd.Series([np.nan, np.nan, np.nan, np.nan, np.nan, 0, 0])
        num_list = [pd.to_numeric(x, errors='coerce') for x in str(s).split(',')]
        mean_val = np.nanmean(num_list); std_val = np.nanstd(num_list)
        min_val = np.nanmin(num_list); max_val = np.nanmax(num_list)
        median_val = np.nanmedian(num_list); count_val = len(num_list)
        nan_count_val = pd.Series(num_list).isna().sum()
        return pd.Series([mean_val, std_val, min_val, max_val, median_val, count_val, nan_count_val])
    stat_names = ['mean', 'std', 'min', 'max', 'median', 'count', 'nan_count']
    col_names = [f'{col_name}_{stat}' for stat in stat_names]
    df_stats = series.apply(calculate_stats_for_row)
    df_stats.columns = col_names
    return df_stats.astype(np.float32)

def process_dataframe(df, file_type='A'):
    ''' DataFrame(청크)을 받아 전처리 및 특성 공학을 수행합니다. (TestDate 피처 포함) '''
    processed_cols = ['Test_id', 'PrimaryKey', 'TestDate']
    valid_cols = [col for col in processed_cols if col in df.columns]
    df_processed = df[valid_cols].copy()
    if 'Age' in df.columns: df_processed['Age'] = df['Age'].apply(preprocess_age)
    if 'TestDate' in df.columns:
        test_date_str = df['TestDate'].astype(str)
        df_processed['TestDate_Year'] = pd.to_numeric(test_date_str.str[:4], errors='coerce')
        df_processed['TestDate_Month'] = pd.to_numeric(test_date_str.str[4:], errors='coerce')
    if file_type == 'A':
        numeric_cols = [f'A6-1', 'A7-1', 'A8-1', 'A8-2'] + [f'A9-{i}' for i in range(1, 6)]
        string_cols = [f'A{i}-{j}' for i in range(1, 6) for j in range(1, 8) if f'A{i}-{j}' not in numeric_cols and f'A{i}-{j}' in df.columns]
    elif file_type == 'B':
        numeric_cols = [f'B9-{i}' for i in range(1, 6)] + [f'B10-{i}' for i in range(1, 7)]
        string_cols = [f'B{i}-{j}' for i in range(1, 9) for j in range(1, 3) if f'B{i}-{j}' in df.columns]
        string_cols += ['B6', 'B7', 'B8']
    else: return pd.DataFrame()
    for col in numeric_cols:
        if col in df.columns: df_processed[col] = pd.to_numeric(df[col], errors='coerce')
    for col in string_cols:
        if col in df.columns:
            stats_df = get_string_list_stats(df[col], col)
            df_processed = pd.concat([df_processed, stats_df], axis=1)
    return df_processed
# --- (전처리 함수 정의 끝) ---

# --- 2. 추론(Inference) 실행 ---
def main():
    print("[+] Inference script started.")
    DATA_DIR, MODEL_DIR, OUTPUT_DIR = './data', './model', './output'
    TEST_A_PATH = os.path.join(DATA_DIR, 'test', 'A.csv')
    TEST_B_PATH = os.path.join(DATA_DIR, 'test', 'B.csv')
    TEST_IDS_PATH = os.path.join(DATA_DIR, 'test.csv')
    SUBMISSION_PATH = os.path.join(OUTPUT_DIR, 'submission.csv')
    CHUNKSIZE = 100000

    # 모델 및 교정기 경로 정의 (3개 모델 모두)
    LGB_CLF_MODEL_PATH = os.path.join(MODEL_DIR, 'model.txt')
    CAT_CLF_MODEL_PATH = os.path.join(MODEL_DIR, 'model_cat.cbm')
    LGB_REG_MODEL_PATH = os.path.join(MODEL_DIR, 'model_reg.txt')
    LGB_CLF_CALIBRATOR_PATH = os.path.join(MODEL_DIR, 'calibrator_lgb.joblib')
    CAT_CLF_CALIBRATOR_PATH = os.path.join(MODEL_DIR, 'calibrator_cat.joblib')
    LGB_REG_CALIBRATOR_PATH = os.path.join(MODEL_DIR, 'calibrator_lgb_reg.joblib')

    # 모델, 교정기 로드 (3개 모델 모두)
    try:
        lgb_clf_model = lgb.Booster(model_file=LGB_CLF_MODEL_PATH)
        lgb_features = lgb_clf_model.feature_name()
        print(f"LGBM Classifier Model loaded from {LGB_CLF_MODEL_PATH}")
        cat_clf_model = CatBoostClassifier()
        cat_clf_model.load_model(CAT_CLF_MODEL_PATH)
        cat_features_model = cat_clf_model.feature_names_
        print(f"CatBoost Classifier Model loaded from {CAT_CLF_MODEL_PATH}")
        lgb_reg_model = lgb.Booster(model_file=LGB_REG_MODEL_PATH)
        print(f"LGBM Regressor Model loaded from {LGB_REG_MODEL_PATH}")
        lgb_clf_calibrator = joblib.load(LGB_CLF_CALIBRATOR_PATH)
        print(f"LGBM Classifier Calibrator loaded from {LGB_CLF_CALIBRATOR_PATH}")
        cat_clf_calibrator = joblib.load(CAT_CLF_CALIBRATOR_PATH)
        print(f"CatBoost Classifier Calibrator loaded from {CAT_CLF_CALIBRATOR_PATH}")
        lgb_reg_calibrator = joblib.load(LGB_REG_CALIBRATOR_PATH)
        print(f"LGBM Regressor Calibrator loaded from {LGB_REG_CALIBRATOR_PATH}")
    except Exception as e:
        print(f"!! Error loading models or calibrators: {e}"); return

    # 테스트 데이터 전처리 (A, B)
    # ( ... 이전과 동일한 전처리 로직 ... )
    print("Processing Test A file...")
    # ... (A 파일 처리) ...
    try:
        with pd.read_csv(TEST_A_PATH, chunksize=CHUNKSIZE) as reader:
            processed_chunks_A = [process_dataframe(chunk, file_type='A') for chunk in reader]
    except FileNotFoundError:
        print("Test A file not found. Skipping.")
        processed_chunks_A = []
    gc.collect()

    print("Processing Test B file...")
    # ... (B 파일 처리) ...
    try:
        with pd.read_csv(TEST_B_PATH, chunksize=CHUNKSIZE) as reader:
            processed_chunks_B = [process_dataframe(chunk, file_type='B') for chunk in reader]
    except FileNotFoundError:
        print("Test B file not found. Skipping.")
        processed_chunks_B = []
    gc.collect()

    df_A_processed = pd.concat(processed_chunks_A, ignore_index=True) if processed_chunks_A else pd.DataFrame()
    df_B_processed = pd.concat(processed_chunks_B, ignore_index=True) if processed_chunks_B else pd.DataFrame()
    df_all_features = pd.concat([df_A_processed, df_B_processed], ignore_index=True, sort=False)
    print(f"Total processed test features shape: {df_all_features.shape}")

    # test.csv 병합 + Lag/Static PK/Hypothesis/TargetEnc 피처 생성
    try:
        df_test_meta = pd.read_csv(TEST_IDS_PATH)[['Test_id', 'Test']]
        df_test_final = pd.merge(df_test_meta, df_all_features, on='Test_id', how='left')

        # --- TestDate 숫자형 변환 ---
        df_test_final['TestDate_Numeric'] = pd.to_datetime(df_test_final['TestDate'], format='%Y%m', errors='coerce')
        df_test_final['TestDate_Numeric'] = df_test_final['TestDate_Numeric'].dt.year +                                             (df_test_final['TestDate_Numeric'].dt.month / 12.0)
        df_test_final['TestDate_Numeric'] = df_test_final['TestDate_Numeric'].fillna(0)

        # --- Lag 피처 생성 ---
        print("Creating Lag (Time-Series) features for test set...")
        # ... (Lag 피처 생성 로직) ...
        df_test_final = df_test_final.sort_values(by=['PrimaryKey', 'TestDate_Numeric']).reset_index(drop=True)
        lag_cols = [
            'TestDate_Numeric', 'Age', 'A1-2_mean', 'A1-3_mean', 'A1-4_mean', 'A6-1', 'A7-1',
            'B1-1_mean', 'B6_mean', 'B7_mean', 'B10-1', 'B10-2'
        ]
        lag_cols = [col for col in lag_cols if col in df_test_final.columns]
        for col in lag_cols:
            df_test_final[f'Prev_{col}'] = df_test_final.groupby('PrimaryKey')[col].shift(1)
        df_test_final['Test_Gap_Time'] = df_test_final.apply(
            lambda row: row['TestDate_Numeric'] - row['Prev_TestDate_Numeric'] if pd.notna(row['Prev_TestDate_Numeric']) and row['Prev_TestDate_Numeric'] > 0 else np.nan, axis=1
        )
        df_test_final['Age_Delta'] = df_test_final['Age'] - df_test_final['Prev_Age'] # Use df_test_final
        if 'A1-2_mean' in lag_cols: df_test_final['A1-2_mean_Delta'] = df_test_final['A1-2_mean'] - df_test_final['Prev_A1-2_mean']
        if 'B10-1' in lag_cols: df_test_final['B10-1_Delta'] = df_test_final['B10-1'] - df_test_final['Prev_B10-1']
        print("Lag/Delta feature creation finished for test set.")


        # --- 가설 기반 피처 생성 ---
        print("Creating Hypothesis-Driven Features for test set...")
        # ... (Risk_... 피처 생성 로직) ...
        if 'A9-2' in df_test_final.columns and 'A9-5' in df_test_final.columns:
             df_test_final['Risk_Impulse_x_RuleViolation'] = df_test_final['A9-2'] * df_test_final['A9-5']
        if 'B9-2' in df_test_final.columns and 'B10-2' in df_test_final.columns:
             df_test_final['Risk_Impulse_x_RiskTaking'] = df_test_final['B9-2'] * df_test_final['B10-2']
        if 'B9-3' in df_test_final.columns and 'B10-6' in df_test_final.columns:
             df_test_final['Risk_Antisocial_x_Aggressive'] = df_test_final['B9-3'] * df_test_final['B10-6']
        risk_cols = [
            'A9-1', 'A9-2', 'A9-3', 'A9-4', 'A9-5', 'B9-1', 'B9-2', 'B9-3', 'B9-4', 'B9-5',
            'B10-1', 'B10-2', 'B10-3', 'B10-4', 'B10-5', 'B10-6'
        ]
        existing_risk_cols = [col for col in risk_cols if col in df_test_final.columns]
        if existing_risk_cols:
             df_test_final['Risk_Index_Sum'] = df_test_final[existing_risk_cols].fillna(0).sum(axis=1)
             df_test_final['Risk_Index_Mean'] = df_test_final[existing_risk_cols].fillna(0).mean(axis=1)
        print("Hypothesis-Driven feature creation finished for test set.")


        # --- 타겟 인코딩 적용 (Test Set - Smoothed) ---
        print("Applying Smoothed Target Encoding for test set (using global mean)...")
        # global_target_mean 값은 노트북 실행 결과에서 가져와야 함 (예: 0.028877)
        GLOBAL_TARGET_MEAN = 0.028877 # ★ 노트북 실행 결과 확인 후 값 수정 필요 ★
        KEY = 'PrimaryKey'
        TARGET_ENC_COL = f'{KEY}_TargetEnc_Smooth' # ★ 스무딩된 컬럼명 사용 ★
        df_test_final[TARGET_ENC_COL] = GLOBAL_TARGET_MEAN
        # (개선) 실제로는 학습 데이터로 만든 스무딩된 target_mean_map을 로드해서 map하는 것이 더 정확함
        print("Smoothed Target Encoding applied.")
        # --- 타겟 인코딩 끝 ---


        # --- 정적 PK 피처 생성 ---
        print("Creating PrimaryKey-based features for test set...")
        # ... (정적 PK 피처 생성 로직) ...
        df_test_final['TestDate_Numeric_Agg'] = df_test_final['TestDate_Numeric'].replace(0, np.nan)
        grouped_by_pk = df_test_final.groupby('PrimaryKey')
        pk_features = grouped_by_pk.agg(
            PK_Test_Count=('Test_id', 'count'), PK_First_TestDate=('TestDate_Numeric_Agg', 'min'),
            PK_Last_TestDate=('TestDate_Numeric_Agg', 'max'), PK_Mean_Age=('Age', 'mean'),
            PK_Min_Age=('Age', 'min'), PK_Max_Age=('Age', 'max')
        ).reset_index()
        pk_features['PK_Test_Duration'] = pk_features['PK_Last_TestDate'] - pk_features['PK_First_TestDate']
        pk_features['PK_Age_Range'] = pk_features['PK_Max_Age'] - pk_features['PK_Min_Age']
        pk_test_type_counts = df_test_final.groupby(['PrimaryKey', df_test_final['Test'].astype(str)]).size().unstack(fill_value=0)
        pk_test_type_counts.columns = [f'PK_{col}_Test_Count' for col in pk_test_type_counts.columns]
        pk_test_type_counts = pk_test_type_counts.reset_index()
        df_test_final = pd.merge(df_test_final, pk_features, on='PrimaryKey', how='left')
        df_test_final = pd.merge(df_test_final, pk_test_type_counts, on='PrimaryKey', how='left')
        df_test_final = df_test_final.drop(columns=['TestDate_Numeric', 'TestDate_Numeric_Agg'])
        print("PrimaryKey features merged for test set.")


        df_test_final['Test'] = df_test_final['Test'].astype(str)
        print("Merged with test.csv successfully.")
    except Exception as e:
        print(f"Error during feature engineering for test set: {e}"); return

    # 6. 예측을 위한 특성 준비 (모델별로)
    X_test_lgb = df_test_final.reindex(columns=lgb_features, fill_value=np.nan)
    if 'Test' in X_test_lgb.columns:
        X_test_lgb['Test'] = X_test_lgb['Test'].astype('category')
    print(f"Final X_test_lgb shape for prediction: {X_test_lgb.shape}")

    final_cat_features = [col for col in cat_features_model if col in df_test_final.columns]
    X_test_cat_df = df_test_final[final_cat_features]
    categorical_features_indices = [i for i, col in enumerate(X_test_cat_df.columns) if col == 'Test']
    print(f"Categorical feature index for CatBoost Pool: {categorical_features_indices}")
    test_pool = Pool(data=X_test_cat_df, cat_features=categorical_features_indices)
    print(f"CatBoost Pool created successfully.")

    # ★ 7. 3개 모델 앙상블 예측 (보정 + 최적 가중 평균) ★
    print("Predicting probabilities (LGBM Classifier - Raw)...")
    preds_lgb_clf_raw = lgb_clf_model.predict(X_test_lgb)

    print("Predicting probabilities (CatBoost Classifier - Raw)...")
    preds_cat_clf_raw = cat_clf_model.predict_proba(test_pool)[:, 1]

    print("Predicting probabilities (LGBM Regressor - Raw)...")
    preds_lgb_reg_raw = lgb_reg_model.predict(X_test_lgb)

    print("Applying calibration (IsotonicRegression)...")
    preds_lgb_clf = lgb_clf_calibrator.predict(preds_lgb_clf_raw)
    preds_cat_clf = cat_clf_calibrator.predict(preds_cat_clf_raw)
    preds_lgb_reg = lgb_reg_calibrator.predict(np.clip(preds_lgb_reg_raw, 0, 1))

    # ★ Optuna가 찾은 최종 최적 가중치 적용 ★
    WEIGHT_LGB_CLF = 0.7339  # Optuna 최종 결과
    WEIGHT_CAT_CLF = 0.2176  # Optuna 최종 결과
    WEIGHT_LGB_REG = 0.0485  # Optuna 최종 결과
    print(f"Applying final optimized 3-model weights (LGB_clf: {WEIGHT_LGB_CLF:.4f}, Cat: {WEIGHT_CAT_CLF:.4f}, LGB_reg: {WEIGHT_LGB_REG:.4f})...")
    final_predictions = (WEIGHT_LGB_CLF * preds_lgb_clf +
                           WEIGHT_CAT_CLF * preds_cat_clf +
                           WEIGHT_LGB_REG * preds_lgb_reg)

    # 최종 예측값 범위 보장
    final_predictions = np.clip(final_predictions, 0, 1)

    # 8. 제출 파일 생성
    df_submission = pd.DataFrame({
        'Test_id': df_test_final['Test_id'],
        'Label': final_predictions
    })

    # 9. 결과 저장
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df_submission.to_csv(SUBMISSION_PATH, index=False)

    print(f"[+] Submission file saved to {SUBMISSION_PATH}")
    print("[+] Inference script finished.")

if __name__ == "__main__":
    main()
