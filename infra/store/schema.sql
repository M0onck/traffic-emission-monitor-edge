-- ==============================================================================
-- 1. 监测任务总表 (Session Task)
-- 用途: 记录每次采集任务的宏观元数据，实现数据的批次化管理
-- ==============================================================================
CREATE TABLE IF NOT EXISTS Session_Task (
    session_id TEXT PRIMARY KEY,          -- 会话唯一标识 (时间戳字符串)
    start_time REAL,                      -- 任务开始时间戳
    end_time REAL,                        -- 任务结束时间戳 (任务进行中为 NULL)
    location_desc TEXT,                   -- 监测点位描述 (例如 "交叉路口A")
    status TEXT                           -- 当前状态 ('recording', 'flushing', 'completed')
);

-- ==============================================================================
-- 2. 环境微观原始表 (Environment Raw)
-- 用途: 1Hz高频实时记录气象与颗粒物，前端UI可直接查此表画图，后台用来计算特定时间窗口内滚动极小值基线
-- ==============================================================================
CREATE TABLE IF NOT EXISTS Env_Raw (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT,
    timestamp REAL,                       -- 精确时间戳 (1Hz)
    pm25_raw REAL,                        -- PM2.5 瞬时浓度
    pm10_raw REAL,                        -- PM10 瞬时浓度
    wind_speed REAL,                      -- 风速 (m/s)
    wind_dir REAL,                        -- 风向角
    air_temp REAL,                        -- 空气温度
    humidity REAL,                        -- 相对湿度
    ground_temp REAL,                     -- 地面温度 (热成像仪)
    FOREIGN KEY(session_id) REFERENCES Session_Task(session_id)
);

-- ==============================================================================
-- 3. 车辆微观轨迹与特征表 (Vehicle Raw)
-- 用途: 车辆离场后写入。存储其全生命周期的完整信息，特别是经过 S-G 滤波后的轨迹
-- ==============================================================================
CREATE TABLE IF NOT EXISTS Veh_Raw (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT,
    tracker_id INTEGER,                   -- 视觉管道赋予的唯一追踪ID
    vehicle_type TEXT,                    -- 车型: 使用 'LDV' 或 'HDV' 适配排放研究
    energy_type TEXT,                     -- 能源类型: 'Normal' / 'Electric' (用于动态质量修正)
    entry_time REAL,                      -- 入场时间戳
    exit_time REAL,                       -- 离场时间戳
    trajectory_blob TEXT,                 -- 存储平滑后高频轨迹序列的 JSON (包含 [t, x, y, v, a, vsp])
    FOREIGN KEY(session_id) REFERENCES Session_Task(session_id)
);

-- ==============================================================================
-- 4. 交通宏观表 (Vehicle Summary - 前端 UI 看板专用)
-- 用途: 车辆离场后写入。作为前端实时数据看板的展示源
-- ==============================================================================
CREATE TABLE IF NOT EXISTS Veh_Sum (
    id INTEGER PRIMARY KEY AUTOINCREMENT, 
    session_id TEXT,                      -- 关联当前任务
    tracker_id INTEGER,            
    vehicle_type TEXT,                    -- 车型: LDV / HDV
    energy_type TEXT,                     -- 能源类型: Normal / Electric
    entry_time REAL,                      
    exit_time REAL,                       
    average_speed REAL,                   -- 平均车速 (m/s)
    dominant_opmodes TEXT,                -- 主导工况序列 (JSON 数组)
    FOREIGN KEY(session_id) REFERENCES Session_Task(session_id)
);

-- ==============================================================================
-- 5. 多源数据对齐表 (Aligned Dataset)
-- 用途: 由延迟对齐引擎生成，直接包含对齐处理好的变量，导出为 CSV 即可喂给模型
-- ==============================================================================
CREATE TABLE IF NOT EXISTS Aligned_Dataset (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT,
    aligned_timestamp REAL,               -- 对齐后的绝对时间戳 (T_now - x s)
    
    -- 结果变量 (Y)
    pm10_raw REAL,                        -- 该时刻表观浓度
    pm10_baseline REAL,                   -- 过去n分钟滑动极小值基线
    delta_c REAL,                         -- 扬尘增量
    
    -- 核心驱动变量 (T)
    sigma_vsp REAL,                       -- 交通瞬态综合做功通量 (包含电动车质量修正)
    
    -- 气象协变量 (X)
    w_cross REAL,                         -- 有效横风扰动分量
    delta_tv REAL,                        -- 路气虚温差
    
    FOREIGN KEY(session_id) REFERENCES Session_Task(session_id)
);

-- ==============================================================================
-- 创建索引加速查询 (特别是按时间窗口查询和绘图)
-- ==============================================================================
CREATE INDEX IF NOT EXISTS idx_env_session_time ON Env_Raw(session_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_veh_sum_session ON Veh_Sum(session_id);
CREATE INDEX IF NOT EXISTS idx_veh_raw_session_time ON Veh_Raw(session_id, exit_time);
CREATE INDEX IF NOT EXISTS idx_aligned_session_time ON Aligned_Dataset(session_id, aligned_timestamp);
