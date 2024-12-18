import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_log_error
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
import holidays
from sklearn.cluster import KMeans
import xgboost as xgb

# 1. 数据加载
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# 2. 数据预处理
# 去掉异常值 (如行程时间小于1秒或超过24小时的样本)
# train = train[(train['trip_duration'] > 1) & (train['trip_duration'] <= 24 * 3600)]

# 对目标变量进行对数变换
train["log_trip_duration"] = np.log1p(train["trip_duration"])

# 合并数据集以统一处理
train["is_train"] = 1
test["is_train"] = 0
test["trip_duration", "dropoff_datetime"] = np.nan
full_data = pd.concat([train, test])

# 3. 特征工程
def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    创建时间相关特征
    Args:
        df: 包含pickup_datetime列的DataFrame
    Returns:
        添加了时间特征的DataFrame
    """
    df = df.copy()
    
    # 转换时间格式
    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])
    
    # 基础时间特征
    df["pickup_hour"] = df["pickup_datetime"].dt.hour
    df["pickup_dayofweek"] = df["pickup_datetime"].dt.dayofweek
    
    # 周期性时间特征（使用sin和cos转换使特征连续）
    df["hour_sin"] = np.sin(df["pickup_hour"] * (2 * np.pi / 24))
    df["hour_cos"] = np.cos(df["pickup_hour"] * (2 * np.pi / 24))
    df["dayofweek_sin"] = np.sin(df["pickup_dayofweek"] * (2 * np.pi / 7))
    df["dayofweek_cos"] = np.cos(df["pickup_dayofweek"] * (2 * np.pi / 7))
    
    # 高峰时段标记
    df["is_rush_hour"] = df["pickup_hour"].isin([7,8,9,17,18,19]).astype(int)
    
    # 工作日/周末标记
    df["is_weekend"] = df["pickup_dayofweek"].isin([5, 6]).astype(int)
    
    # 时间段分类
    def get_time_period(hour: int) -> int:
        if 5 <= hour <= 11:
            return 0  # 早上
        elif 12 <= hour <= 17:
            return 1  # 下午
        elif 18 <= hour <= 22:
            return 2  # 晚上
        else:
            return 3  # 凌晨
    
    df["time_period"] = df["pickup_hour"].apply(get_time_period)
    
    # 假期标记
    us_holidays = holidays.US()
    df["is_holiday"] = df["pickup_datetime"].dt.date.map(
        lambda x: x in us_holidays
    ).astype(int)
    
    return df

# 应用时间特征处理
full_data = create_time_features(full_data)

# 计算地理距离 (Haversine公式)
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # 地球半径（公里）
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)
    a = (
        np.sin(delta_phi / 2) ** 2
        + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2) ** 2
    )
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

# 计算曼哈顿距离（城市街区距离）
def manhattan_distance(lat1, lon1, lat2, lon2):
    # 使用haversine公式分别计算南北和东西方向的距离
    lat_distance = haversine(lat1, lon1, lat2, lon1)  # 南北方向
    lon_distance = haversine(lat1, lon1, lat1, lon2)  # 东西方向
    return lat_distance + lon_distance


# 计算方位角
def calculate_bearing(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    d_lon = lon2 - lon1
    y = np.sin(d_lon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(d_lon)
    bearing = np.degrees(np.arctan2(y, x))
    # 将角度转换到0-360度范围内
    return (bearing + 360) % 360


# 添加距离特征
full_data["euclidean_distance"] = haversine(
    full_data["pickup_latitude"],
    full_data["pickup_longitude"],
    full_data["dropoff_latitude"],
    full_data["dropoff_longitude"],
)
full_data["manhattan_distance"] = manhattan_distance(
    full_data["pickup_latitude"],
    full_data["pickup_longitude"],
    full_data["dropoff_latitude"],
    full_data["dropoff_longitude"],
)
full_data["bearing"] = calculate_bearing(
    full_data["pickup_latitude"],
    full_data["pickup_longitude"],
    full_data["dropoff_latitude"],
    full_data["dropoff_longitude"],
)

# 添加距离比率特征
full_data["distance_ratio"] = full_data["manhattan_distance"] / (
    full_data["euclidean_distance"] + 1
)

# 将store_and_fwd_flag转换为数值型
full_data["store_and_fwd_flag"] = (full_data["store_and_fwd_flag"] == "Y").astype(int)

# 填补测试集缺失值
full_data["euclidean_distance"] = full_data["euclidean_distance"].fillna(0)
full_data["manhattan_distance"] = full_data["manhattan_distance"].fillna(0)
full_data["bearing"] = full_data["bearing"].fillna(-1)
full_data["distance_ratio"] = full_data["distance_ratio"].fillna(1)

# 在特征工程部分修改PCA处理
# 将上车和下车位置垂直堆叠
pickup_locations = full_data[['pickup_latitude', 'pickup_longitude']].copy()
dropoff_locations = full_data[['dropoff_latitude', 'dropoff_longitude']].copy()

# 垂直堆叠所有位置点
all_locations = pd.concat([pickup_locations, dropoff_locations], axis=0)

# 标准化位置数据
scaler = StandardScaler()
locations_scaled = scaler.fit_transform(all_locations)

# 应用PCA
pca = PCA(n_components=2)
pca.fit(locations_scaled)

# 分别转换上车和下车位置
pickup_pca = pca.transform(scaler.transform(pickup_locations))
dropoff_pca = pca.transform(scaler.transform(dropoff_locations))

# 添加PCA特征到数据集
full_data['pickup_loc_pca1'] = pickup_pca[:, 0]
full_data['pickup_loc_pca2'] = pickup_pca[:, 1]
full_data['dropoff_loc_pca1'] = dropoff_pca[:, 0]
full_data['dropoff_loc_pca2'] = dropoff_pca[:, 1]

# 在PCA处理之后添加聚类分析
# 确定聚类数量（可以根据实际情况调整）
n_clusters = 10

# 使用KMeans对标准化后的位置数据进行聚类
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(locations_scaled)

# 分别对上车点和下车点进行聚类预测
pickup_clusters = kmeans.predict(scaler.transform(pickup_locations))
dropoff_clusters = kmeans.predict(scaler.transform(dropoff_locations))

# 添加聚类标签到数据集
full_data['pickup_cluster'] = pickup_clusters
full_data['dropoff_cluster'] = dropoff_clusters

# 拆分训练集和测试集
train = full_data[full_data["is_train"] == 1]
test = full_data[full_data["is_train"] == 0]

X = train[
    [
        "pickup_hour",
        "pickup_dayofweek",
        "hour_sin",
        "hour_cos",
        "dayofweek_sin",
        "dayofweek_cos",
        "is_rush_hour",
        "is_weekend",
        "time_period",
        "is_holiday",
        "euclidean_distance",
        "manhattan_distance",
        "bearing",
        "distance_ratio",
        "passenger_count",
        "pickup_loc_pca1",
        "pickup_loc_pca2",
        "dropoff_loc_pca1",    # 新增下车位置PCA特征
        "dropoff_loc_pca2",    # 新增下车位置PCA特征
        "pickup_cluster",     # 新增聚类特征
        "dropoff_cluster"     # 新增聚类特征
    ]
]

# 更新测试集特征列表
X_test = test[X.columns]

y = train["log_trip_duration"]

# 4. 模型训练
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# XGBoost参数配置
xgb_params = {
    'objective': 'reg:squarederror',
    'tree_method': 'gpu_hist',  # 使用GPU加速
    'gpu_id': 0,               # GPU设备ID
    'learning_rate': 0.1,
    'max_depth': 8,
    'min_child_weight': 3,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'n_estimators': 1000,
    'random_state': 42
}

# 创建XGBoost模型
model = xgb.XGBRegressor(**xgb_params)

# 使用早停机制训练模型
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)], 
    eval_metric='rmse',
    early_stopping_rounds=50,
    verbose=100
)

# 验证模型（使用最佳迭代次数）
val_pred = model.predict(X_val)
val_pred = np.expm1(val_pred)
val_y_true = np.expm1(y_val)
rmsle = np.sqrt(mean_squared_log_error(val_y_true, val_pred))
print(f"Validation RMSLE: {rmsle}")

# 特征重要性分析
importance_df = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
})
print("\nTop 10 Important Features:")
print(importance_df.sort_values('importance', ascending=False).head(10))

# 5. 预测测试集（使用最佳迭代次数）
test_pred = model.predict(X_test)
test_pred = np.expm1(test_pred)

# 6. 保存提交结果
submission = pd.DataFrame({"id": test["id"], "trip_duration": test_pred})
submission.to_csv("submission.csv", index=False)
