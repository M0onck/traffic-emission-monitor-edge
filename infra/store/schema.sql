-- 宏观表：记录车辆离场时的汇总信息
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

-- 微观表：记录车辆逐帧的高频时序轨迹
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
