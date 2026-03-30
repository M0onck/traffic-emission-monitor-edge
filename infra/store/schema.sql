-- 宏观表：记录车辆离场时的汇总信息
CREATE TABLE IF NOT EXISTS vehicle_macro (
    tracker_id INTEGER PRIMARY KEY,
    first_frame INTEGER,
    last_seen_frame INTEGER,
    class_id INTEGER,
    class_name TEXT,
    plate_number TEXT,
    plate_color TEXT,
    max_speed REAL,
    average_speed REAL,
    total_distance_m REAL
);

-- 微观表：记录车辆逐帧的高频时序轨迹
CREATE TABLE IF NOT EXISTS vehicle_micro (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tracker_id INTEGER,
    frame_id INTEGER,
    timestamp REAL,      -- 精确绝对时间戳
    ipm_x REAL,          -- 逆透视变换后的 X 坐标 (米)
    ipm_y REAL,          -- 逆透视变换后的 Y 坐标 (米)
    speed REAL,          -- 瞬时速度 (m/s)
    accel REAL,          -- 瞬时加速度 (m/s^2)
    vsp REAL,            -- 瞬时比功率 (kW/tonne)
    FOREIGN KEY(tracker_id) REFERENCES vehicle_macro(tracker_id)
);

-- 建立索引加速后续的时序回放和查询
CREATE INDEX IF NOT EXISTS idx_micro_tracker ON vehicle_micro(tracker_id);
CREATE INDEX IF NOT EXISTS idx_micro_timestamp ON vehicle_micro(timestamp);
