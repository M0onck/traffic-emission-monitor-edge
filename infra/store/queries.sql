-- infra/store/queries.sql

-- ==============================================================================
-- 1. Session & Raw Data 插入
-- ==============================================================================

-- name: insert_session
INSERT INTO Session_Task (session_id, start_time, location_desc, status)
VALUES (?, ?, ?, 'recording');

-- name: complete_session
UPDATE Session_Task
SET end_time = ?, status = 'completed'
WHERE session_id = ?;

-- name: insert_env_raw
INSERT INTO Env_Raw (
    session_id, timestamp, pm25_raw, pm10_raw, 
    wind_speed, wind_dir, air_temp, humidity, ground_temp
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);

-- name: insert_veh_raw
INSERT INTO Veh_Raw (
    session_id, tracker_id, vehicle_type, energy_type,
    entry_time, exit_time, trajectory_blob
) VALUES (?, ?, ?, ?, ?, ?, ?);

-- name: insert_veh_sum
INSERT INTO Veh_Sum (
    session_id, tracker_id, vehicle_type, energy_type, 
    entry_time, exit_time, average_speed, dominant_opmodes, settlement_status
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);

-- ==============================================================================
-- 2. 对齐引擎专用查询 (Snapshot Support)
-- ==============================================================================

-- name: get_nearest_env_raw
-- 职责: 获取离目标时间戳最近的一条环境记录，带有最大容差限制
SELECT * FROM Env_Raw 
WHERE session_id = ? 
  AND ABS(timestamp - ?) <= ?
ORDER BY ABS(timestamp - ?) ASC 
LIMIT 1;

-- name: get_active_vehicles_during
-- 职责: 获取在特定时间点处于场内的所有车辆及其原始轨迹
SELECT tracker_id, vehicle_type, trajectory_blob 
FROM Veh_Raw 
WHERE session_id = ? 
  AND entry_time <= ? 
  AND exit_time >= ?;

-- name: insert_aligned_snapshot
-- 职责: 插入重构后的物理快照数据
INSERT INTO Aligned_Snapshots (
    session_id, aligned_timestamp, 
    air_temp, ground_temp, humidity, wind_speed, wind_dir, pm25, pm10,
    active_vehicle_count, vehicles_data
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);

-- ==============================================================================
-- 3. 统计与清理
-- ==============================================================================

-- name: get_session_stats
SELECT 
    COUNT(*) as total_vehicles,
    AVG(exit_time - entry_time) as avg_duration
FROM Veh_Sum 
WHERE session_id = ?;
