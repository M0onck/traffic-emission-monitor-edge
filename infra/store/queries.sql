-- name: insert_session
INSERT INTO Session_Task (session_id, start_time, location_desc, status)
VALUES (?, ?, ?, 'recording');

-- name: complete_session
UPDATE Session_Task
SET end_time = ?, status = 'completed'
WHERE session_id = ?;

-- name: insert_macro
INSERT OR REPLACE INTO vehicle_macro (
    tracker_id, vehicle_type, energy_type, entry_time, exit_time,
    average_speed, dominant_opmodes
) VALUES (?, ?, ?, ?, ?, ?, ?);

-- name: insert_micro
INSERT INTO vehicle_micro (
    tracker_id, frame_id, timestamp, ipm_x, ipm_y
) VALUES (?, ?, ?, ?, ?);
