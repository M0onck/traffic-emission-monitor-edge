-- name: insert_macro
INSERT OR REPLACE INTO vehicle_macro (
    tracker_id, first_frame, last_seen_frame, class_id, class_name,
    plate_number, plate_color, max_speed, average_speed, total_distance_m
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);

-- name: insert_micro
INSERT INTO vehicle_micro (
    tracker_id, frame_id, timestamp, ipm_x, ipm_y
) VALUES (?, ?, ?, ?, ?);
