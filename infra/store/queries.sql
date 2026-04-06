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

-- name: insert_aligned_dataset
INSERT INTO Aligned_Dataset (
    session_id, aligned_timestamp, 
    pmc_raw, pmc_baseline, delta_c_flux, 
    e_traffic, 
    d_trans, w_cross, delta_tv
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
