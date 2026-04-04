-- 交通宏观表：记录车辆离场时的汇总信息
CREATE TABLE IF NOT EXISTS vehicle_macro (
    id INTEGER PRIMARY KEY AUTOINCREMENT, -- 按入场绝对时间排序的序号 (第一辆为1)
    tracker_id INTEGER UNIQUE,            -- 保留 tracker_id 以便与 vehicle_micro 关联
    vehicle_type TEXT,                    -- 车型: LDV / HDV
    energy_type TEXT,                     -- 能源类型: Normal / Electric
    entry_time REAL,                      -- 入场时间 (绝对时间戳)
    exit_time REAL,                       -- 离场时间 (绝对时间戳)
    average_speed REAL,                   -- 平均车速 (m/s)
    dominant_opmodes TEXT                 -- 主导工况序列 (JSON 数组)
);

-- 交通微观表：记录车辆逐帧的高频时序轨迹
CREATE TABLE IF NOT EXISTS vehicle_micro (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tracker_id INTEGER,
    frame_id INTEGER,
    timestamp REAL,      -- 精确绝对时间戳
    ipm_x REAL,          -- 逆透视变换后的 X 坐标 (米)
    ipm_y REAL,          -- 逆透视变换后的 Y 坐标 (米)
    FOREIGN KEY(tracker_id) REFERENCES vehicle_macro(tracker_id)
);

-- 建立索引加速后续的时序回放和查询
CREATE INDEX IF NOT EXISTS idx_micro_tracker ON vehicle_micro(tracker_id);
CREATE INDEX IF NOT EXISTS idx_micro_timestamp ON vehicle_micro(timestamp);

-- 1Hz 微观环境数据表
CREATE TABLE IF NOT EXISTS env_micro_1hz (
    timestamp INTEGER PRIMARY KEY,  -- 严格对齐到秒级的 Unix 时间戳
    ground_temp REAL,               -- 地面温度 (热成像仪)
    air_temp REAL,                  -- 空气温度 (气象站)
    humidity REAL,                  -- 湿度
    wind_speed REAL,                -- 风速
    wind_dir REAL,                  -- 风向
    pm25 REAL,                      -- PM2.5 瞬时浓度
    pm10 REAL                       -- PM10 瞬时浓度
);

-- 1min 宏观环境数据表
CREATE TABLE IF NOT EXISTS env_macro_1min (
    timestamp INTEGER PRIMARY KEY,  -- 每分钟的起始时间戳 (例如秒数为 00 的时刻)
    ground_temp_avg REAL,
    air_temp_avg REAL,
    humidity_avg REAL,
    wind_speed_avg REAL,
    wind_dir_avg REAL,              -- 注意：风向的平均可能需要向量平均计算
    pm25_integral REAL,             -- PM2.5 积分值 (表征该分钟内的总通量)
    pm10_integral REAL              -- PM10 积分值
);
