"""Flask Web Application for Multi-Well Perfusion Control and Imaging."""

from collections import deque
import glob
import io
import os
from pathlib import Path
import subprocess
import threading
import time
import traceback
import webbrowser

import cv2
from flask import Flask, Response, jsonify, render_template, request, send_file, send_from_directory
import numpy as np
from pycromanager import stop_headless

from multi_well_perfusion import *


# ==========================================
# HARDWARE & STATE MANAGEMENT CLASS
# ==========================================

class HardwareController:
    """Encapsulates Micro-Manager core, serial connections, and hardware state."""

    print("Initializing hardware...")

    def __init__(self):
        # Hardware objects
        self.core = None
        self.ser = None
        self.ser_flow = None
        self.ecu = None

        # Threading Locks
        self.mm_lock = threading.Lock()
        self.core_lock = threading.Lock()
        self.telemetry_lock = threading.Lock()

        # Operational states
        self.live_mode = False
        self.last_image = None
        self.last_button_state = "RELEASED"
        self.latest_telemetry = {
            "waste1": "1",
            "waste2": "1",
            "button": "RELEASED",
            "flow_rate": 0.0
        }

        # Rolling history buffers
        self.waste1_history = deque(maxlen=5)
        self.waste2_history = deque(maxlen=5)

    def initialize_hardware(self):
        """Initializes serial connections, ECU, MMCore, and background listeners."""
        self.ser_flow = setup_serial_esp32()
        self.ser = setup_serial()
        _, self.ecu = setup_logging_ecu()
        self.core = pycro_core()
        self.core.enable_debug_log(False)

        send_command(self.ser, b"C 0 0 0\n")

        daemon_thread = threading.Thread(
            target=self.serial_background_listener,
            daemon=True
        )
        daemon_thread.start()

    def serial_background_listener(self):
        """Dedicated background loop to handle reading telemetry from ser_flow.

        Prevents thread collisions and resource blocking while maintaining
        a rolling history window for sensor tendency calculations.
        """
        print("Background Serial Listener Started.")

        while True:
            try:
                if self.ser_flow and self.ser_flow.is_open:
                    line = self.ser_flow.readline().decode('utf-8', errors='ignore').strip()
                    if not line:
                        continue

                    parts = line.split('\t')

                    button_state = "RELEASED"
                    w1, w2, flow = "1", "1", 0.0

                    # Parse values from key-value labels
                    for part in parts:
                        if "Button:" in part:
                            button_state = part.split(':')[1]
                        elif "Waste1:" in part:
                            w1 = part.split(':')[1]
                        elif "Waste2:" in part:
                            w2 = part.split(':')[1]
                        elif "FlowRate:" in part:
                            try:
                                flow = float(part.split(':')[1])
                            except ValueError:
                                pass

                    # Parse numeric values safely for rolling window tracking
                    try:
                        waste1_numeric = int(w1)
                    except ValueError:
                        waste1_numeric = 1

                    try:
                        waste2_numeric = int(w2)
                    except ValueError:
                        waste2_numeric = 1

                    # Update shared state safely with telemetry lock
                    with self.telemetry_lock:
                        self.latest_telemetry["waste1"] = w1
                        self.latest_telemetry["waste2"] = w2
                        self.latest_telemetry["button"] = button_state
                        self.latest_telemetry["flow_rate"] = flow

                        self.waste1_history.append(waste1_numeric)
                        self.waste2_history.append(waste2_numeric)

                    # Trigger emergency abort if hardware button is pressed
                    if button_state == "PRESSED" and self.last_button_state == "RELEASED":
                        print("!!! Emergency Stop Button Pressed !!! Triggering Abort Mechanism.")
                        self.abort()

                    self.last_button_state = button_state

            except Exception as e:
                print(f"Error in background serial thread: {e}")
                time.sleep(1)

    def abort(self):
        """Executes the exact sequence required to safely stop all hardware."""
        try:
            if self.ecu:
                self.ecu.disable_all()
            if self.ser:
                send_command(self.ser, b"1S\n")
                send_command(self.ser, b"2S\n")
                send_command(self.ser, b"C 0 0 0\n")
                send_command_and_wait(self.ser, b"M 0\n")
                send_command_and_wait(self.ser, b"A 0\n")
            if self.core:
                self.core.set_config("fluorescence", "off")
                self.core.set_xy_position(0, 0)
            print("Hardware Abort Sequence Completed Successfully.")
        except Exception as e:
            print("Error encountered during asynchronous abort execution:", e)

    def shutdown(self):
        """Safely powers off components and cleans up device instances."""
        try:
            if self.ecu:
                self.ecu.disable_all()
            if self.core:
                self.core.set_config("fluorescence", "off")
                self.core.unload_all_devices()
            stop_headless()
            if self.ser:
                send_command(self.ser, b"SOCKET_OFF\n")
                self.ser.close()
            if self.ser_flow:
                self.ser_flow.close()
        except Exception as e:
            print("Clean exit handled:", e)


# Instantiate Flask App and Controller
app = Flask(__name__)
hw = HardwareController()

def open_browser():
    webbrowser.open_new_tab("http://127.0.0.1:5000/")

# Hardware constants
FOV_10X = 1192.8  # µm
FOV_20X = 1192.8 / 2  # µm

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def process_and_encode_jpeg(arr, quality=50):
    """Fast vectorized auto-contrast and TurboJPEG/OpenCV encoding.

    Args:
        arr (np.ndarray): Input image array to process.
        quality (int): JPEG compression quality (0-100).

    Returns:
        io.BytesIO: Buffer containing encoded JPEG byte array.
    """
    p_low, p_high = np.percentile(arr, (1, 99))
    if p_high > p_low:
        arr = np.clip((arr - p_low) / (p_high - p_low) * 255.0, 0, 255)
    else:
        arr = np.zeros_like(arr)

    img_uint8 = arr.astype(np.uint8)
    success, encoded_img = cv2.imencode('.jpg', img_uint8, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return io.BytesIO(encoded_img.tobytes())


# ==========================================
# ROUTE HANDLERS
# ==========================================

# --- Web UI Interface ---

@app.route("/")
def index():
    """Renders main dashboard interface."""
    group = hw.core.get_channel_group()
    return render_template("index.html", group=group)


# --- Basic Hardware Controls ---

@app.route("/set_channel", methods=["POST"])
def set_channel():
    """Sets light source channel configuration and exposure."""
    data = request.json
    hw.core.set_config(data["group"], data["channel"])
    exposure = data.get("exposure")
    if exposure:
        try:
            hw.core.set_exposure(float(exposure))
        except ValueError:
            pass
    return jsonify({"status": "ok"})


@app.route("/set_live_mode", methods=["POST"])
def set_live_mode():
    """Toggles continuous sequence acquisition mode and LED ring state on or off."""
    data = request.get_json() or {}
    requested_live = data.get("live", False)
    enable_fluo = data.get("state", False)

    with hw.mm_lock:
        try:
            if requested_live and not hw.live_mode:
                # If fluorescence is enabled, turn off BF (C 0 0 0). Otherwise, turn on BF.
                set_BF_cmd = "C 0 0 0" if enable_fluo else "C 255 255 255"
                command = f"{set_BF_cmd}\n".encode('utf-8')
                send_command(hw.ser, command)

                hw.core.start_continuous_sequence_acquisition(0)
                hw.live_mode = True
                print("[LIVE] Continuous sequence acquisition started.")

            elif not requested_live and hw.live_mode:
                if hw.core.is_sequence_running():
                    hw.core.stop_sequence_acquisition()

                # Turn OFF LED ring once at end
                send_command(hw.ser, b"C 0 0 0\n")

                hw.live_mode = False
                print("[LIVE] Continuous sequence acquisition stopped.")

            return jsonify({"status": "success", "live_mode": hw.live_mode})
        except Exception as e:
            print(f"\n[ERROR] /set_live_mode failed: {e}")
            return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/toggle_fluorescence", methods=["POST"])
def toggle_fluorescence():
    """Toggles fluorescence excitation hardware state and syncs the LED ring."""
    data = request.get_json() or {}
    enable_fluo = data.get("state", False)

    with hw.mm_lock:
        try:
            was_sequence_running = hw.core.is_sequence_running()
            if was_sequence_running:
                hw.core.stop_sequence_acquisition()

            # Set Micro-Manager fluorescence config
            hw.core.set_config("fluorescence", "on_70P" if enable_fluo else "off")
            hw.core.wait_for_system()

            # If live mode is currently running, immediately update the LED ring state
            if hw.live_mode:
                set_BF_cmd = "C 0 0 0" if enable_fluo else "C 255 255 255"
                send_command(hw.ser, f"{set_BF_cmd}\n".encode('utf-8'))

            if was_sequence_running:
                hw.core.start_continuous_sequence_acquisition(0)

            return jsonify({"status": "success", "fluorescence": enable_fluo})
        except Exception as e:
            print("\n[ERROR] /toggle_fluorescence failed:")
            traceback.print_exc()
            if locals().get('was_sequence_running') and not hw.core.is_sequence_running():
                try:
                    hw.core.start_continuous_sequence_acquisition(0)
                except Exception:
                    pass
            return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/set_brightness", methods=['POST'])
def set_brightness():
    """Adjusts LED ring hardware brightness."""
    if not hw.ser:
        return jsonify({"status": "error", "message": "Serial port not connected"}), 500
    data = request.get_json()
    brightness = data.get('brightness', 0)
    command = f"B {brightness}\n".encode('utf-8')
    try:
        hw.ser.write(command)
        return jsonify({"status": "success", "sent": command.decode('utf-8').strip()})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# --- Image Snap Routes ---

@app.route("/snap", methods=["GET"])
def snap_image():
    """Captures a single brightfield frame."""
    if not hw.live_mode:
        send_command(hw.ser, b"C 255 255 255\n")
        hw.core.snap_image()
        send_command(hw.ser, b"C 0 0 0\n")
        java_img = hw.core.get_tagged_image()
    else:
        java_img = hw.core.get_last_tagged_image()

    if java_img is None:
        return "No image", 500

    arr = np.reshape(java_img.pix, newshape=[java_img.tags['Height'], java_img.tags['Width']])
    buf = process_and_encode_jpeg(arr, quality=85)
    return send_file(buf, mimetype='image/jpeg')


@app.route("/snap_fluo", methods=["GET"])
def snap_fluo():
    """Captures a single fluorescent frame."""
    if not hw.live_mode:
        hw.core.set_config("fluorescence", "on_70P")

    hw.core.snap_image()
    java_img = hw.core.get_tagged_image()

    if not hw.live_mode:
        hw.core.set_config("fluorescence", "off")

    if java_img is None:
        return "No image", 500

    arr = np.reshape(java_img.pix, newshape=[java_img.tags['Height'], java_img.tags['Width']])
    buf = process_and_encode_jpeg(arr, quality=85)
    return send_file(buf, mimetype='image/jpeg')


@app.route("/snap_live", methods=["GET"])
def snap_live():
    """Retrieves current frame during active live view streaming."""
    java_img = None

    with hw.core_lock:
        try:
            if hw.core.is_sequence_running():
                java_img = hw.core.get_last_tagged_image()
            else:
                hw.core.snap_image()
                java_img = hw.core.get_tagged_image()
        except Exception as e:
            print(f"[ERROR] MMCore acquisition failed: {e}")
            return "Acquisition error", 500

    if java_img is None:
        return "No image", 500

    try:
        arr = np.reshape(java_img.pix, newshape=[java_img.tags['Height'], java_img.tags['Width']])
        buf = process_and_encode_jpeg(arr, quality=40)
        return send_file(buf, mimetype='image/jpeg')
    except Exception as e:
        print(f"[ERROR] Image processing failed: {e}")
        return "Processing error", 500


# --- Stage & Focus Movement ---

@app.route("/move_xy", methods=["POST"])
def move_xy():
    """Executes relative XY stage movement."""
    data = request.json
    hw.core.set_relative_xy_position(float(data["x"]), float(data["y"]))
    return jsonify({"status": "ok"})


@app.route("/move_z", methods=["POST"])
def move_z():
    """Executes relative Z focus stage movement."""
    data = request.json
    hw.core.set_relative_position(float(data["z"]))
    return jsonify({"status": "ok"})


@app.route('/z-stream')
def z_stream():
    """Streams continuous Server-Sent Events containing current Z position."""
    def generate_z_data():
        while True:
            try:
                z_pos = hw.core.get_position()
                val = round(z_pos, 3)
            except Exception:
                val = "Error"
            yield f"data: {json.dumps({'value': val})}\n\n"
            time.sleep(0.1)
    return Response(generate_z_data(), mimetype='text/event-stream')


@app.route('/flow-stream')
def flow_stream():
    """Streams continuous Server-Sent Events containing current flow rate from background thread."""

    def generate_flow_data():
        while True:
            try:
                # Thread-safe read of telemetry state updated by background thread
                with hw.telemetry_lock:
                    flow_val = hw.latest_telemetry.get("flow_rate", 0.0)

                val = round(flow_val, 2)
            except Exception as e:
                val = "Error"

            yield f"data: {json.dumps({'value': val})}\n\n"
            time.sleep(0.1)  # 10Hz update rate

    return Response(generate_flow_data(), mimetype='text/event-stream')

@app.route("/run_autofocus", methods=["POST"])
def run_autofocus():
    """Executes standard range software autofocus routine."""
    data = request.json
    outcome = software_autofocus(hw.core, hw.ser, float(data["range"]), float(data["step"]))
    return jsonify({"saf_outcome": outcome})


@app.route("/run_large_range_autofocus", methods=["POST"])
def run_large_range_autofocus():
    """Executes wide-range software autofocus routine."""
    outcome = software_autofocus_large_range(hw.core, hw.ser, 1000, 50)
    return jsonify({"saf_outcome_large": outcome})


# --- FOV Management ---

@app.route("/log_experiment_event", methods=["POST"])
def log_experiment_event():
    data = request.json
    exp_name = data.get("exp_name")
    path_to_exp_name = data.get("path_to_exp_name") or str(Path(BASE_PATH) / exp_name)
    event_key = data.get("event_key", "Warning: ")
    message = data.get("message", "")

    log_file_path = Path(path_to_exp_name) / f"{exp_name}_log.csv"

    # Write to existing log file using log_event
    log_event(log_file_path, event_key, message)

    return jsonify({"status": "logged"}), 200

@app.route("/run_determine_best_well_center_coords", methods=["POST"])
def run_determine_best_well_center_coords():
    """Calculates or auto-detects optimal well-center coordinates."""
    data = request.json
    result = determine_best_well_center_coords(
        int(data["plate_type"]), int(data["current_well"]), hw.ser, hw.core,
        FOV_10X, data.get("order_type"), float(data["range"]), float(data["step"]),
        data["exp_name"], data["auto_FOV"]
    )

    if data["auto_FOV"]:
        current_coord, saf_value = result

        # Check for invalid FOV or SAF values
        if current_coord is None or saf_value is None:
            return jsonify({
                "success": False,
                "error": "No valid SAF or best FOV found for this well."
            }), 200

        return jsonify({
            "success": True,
            "current_coord_x": round(current_coord[0][0], 2),
            "current_coord_y": round(current_coord[0][1], 2),
            "saf_value": round(saf_value, 2),
        })

    return jsonify({
        "success": True,
        "current_coord_x": round(result[0], 2),
        "current_coord_y": round(result[1], 2),
        "saf_value": 0
    })


@app.route("/move_to_initial_well_center", methods=["POST"])
def move_to_initial_well_center():
    """Drives stage to estimated center of selected well."""
    data = request.get_json()
    coords = determine_well_centers(int(data["plate_type"]), data.get("order_type", "snake"))
    x_center, y_center = coords[int(data["current_well"]) - 1]

    hw.core.set_xy_position(x_center, y_center)
    while hw.core.device_busy("XYStage"):
        time.sleep(1)

    return jsonify({
        "status": "success",
        "well": data["current_well"],
        "initial_center": [x_center, y_center]
    })


@app.route("/save_manual_fov_coords", methods=["POST"])
def save_manual_fov_coords():
    """Saves user-selected manual field of view coordinates."""
    data = request.get_json() or {}
    with hw.core_lock:
        try:
            x_pos = round(hw.core.get_x_position(), 2)
            y_pos = round(hw.core.get_y_position(), 2)
            focus_dev = hw.core.get_focus_device()
            z_pos = round(hw.core.get_position(focus_dev), 2) if focus_dev else 0.0

            return jsonify({
                "status": "success",
                "current_well": data.get("current_well"),
                "current_coord_x": x_pos,
                "current_coord_y": y_pos,
                "saf_value": z_pos
            })
        except Exception as e:
            print("\n[ERROR] /save_manual_fov_coords failed:")
            traceback.print_exc()
            return jsonify({"status": "error", "message": str(e)}), 500


# --- Fluidics & Calibration ---
from multi_well_perfusion import RES_MAP

@app.route('/test_flow_start', methods=['POST'])
def test_flow_start():
    """Initiates fluid flow test on targeted reservoir."""
    data = request.get_json() or {}
    try:
        reservoir_no = int(data["reservoir_no"])
    except (KeyError, ValueError, TypeError):
        return jsonify({"status": "error", "message": "Invalid or missing reservoir_no"}), 400

    physical_pin = RES_MAP[reservoir_no]
    hw.ecu.enable(physical_pin)
    hw.ecu.enable(RES_WASH if physical_pin in [1, 2, 3] else RES_SOLS)
    return jsonify({"status": "success", "message": f"Test flow for reservoir {reservoir_no} started"})


@app.route('/test_flow_stop', methods=['POST'])
def test_flow_stop():
    """Stops all active test flows."""
    hw.ecu.disable_all()
    return jsonify({"status": "success", "message": "Test flow stopped"})


@app.route("/check_sensor_status", methods=["GET"])
def check_sensor_status():
    """Evaluates reservoir fluid level and returns sensor status alerts."""
    try:
        with hw.telemetry_lock:
            w1_snapshot = list(hw.waste1_history)
            w2_snapshot = list(hw.waste2_history)
        alerts = get_sensor_tendency(w1_snapshot, w2_snapshot)
        return jsonify({"status": "success", "alerts": alerts})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/get_perfusion_time", methods=["POST"])
def get_perfusion_time():
    """Calculates mean required perfusion time from raw calibration points."""
    data = request.json
    filtered = [x for x in data["perfusion_times"] if x > 0]
    perfusion_time_mean = round(np.mean(filtered) + 2, 2)
    return jsonify({"perfusion_time_mean": perfusion_time_mean})


@app.route("/calibration", methods=["POST"])
def calibration():
    """Executes volumetric flow rate calibration per reservoir."""
    data = request.json or {}
    sequence = data.get("sequence", [])
    current_res = data.get("current_reservoir")
    perfusion_time = None

    if current_res == 1:
        perfusion_time = flow_rate_calibration(
            1, hw.ecu, hw.ser, hw.latest_telemetry, hw.telemetry_lock, target_volume_ul=V_WASH_UL
        )
    elif current_res in RES_MAP:
        # Check if the mapped physical pin is present in the sequence
        target_pin = RES_MAP[current_res]
        if target_pin in sequence:
            multiplier = 2 if current_res in [2, 3, 4] else 1
            raw_time = flow_rate_calibration(
                current_res, hw.ecu, hw.ser, hw.latest_telemetry, hw.telemetry_lock, target_volume_ul=V_WASH_UL
            )
            perfusion_time = raw_time * multiplier if raw_time is not None else None

    # Fallback to default estimated time instead of None if skipped/uncalibrated
    if perfusion_time is None:
        perfusion_time = 0

    return jsonify({"current_perfusion_time": round(perfusion_time, 2)})


@app.route("/emptying", methods=["POST"])
def emptying():
    """Runs automated emptying command sequence for waste containers."""
    try:
        data = request.get_json() or {}
        target_str = str(data.get("waste_id", "all")).lower()
        pumps = [int(p) for p in ["1", "2"] if p in target_str] or [1, 2]

        send_command(hw.ser, b'DELAY 2000\n')
        for pump in pumps:
            cmd = f"{pump}D-11000\n".encode('utf-8')
            send_command(hw.ser, f"VALVE{pump}_ON\n".encode('utf-8'))
            send_command_and_wait(hw.ser, cmd, expected_response="Dispensing Complete")
            send_command(hw.ser, f"VALVE{pump}_OFF\n".encode('utf-8'))

        send_command(hw.ser, b'DELAY 18000\n')

        with hw.telemetry_lock:
            if 1 in pumps:
                hw.waste1_history.clear()
            if 2 in pumps:
                hw.waste2_history.clear()

        return jsonify({"status": "success", "message": f"Emptying procedure complete for Waste {target_str}"}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/clean-syringe-pumps', methods=['POST'])
def clean_syringe_pumps():
    try:
        # Pump 1
        send_command(hw.ser, b"DELAY 5000")
        send_command(hw.ser, b'1D5000\n')
        send_command(hw.ser, b"VALVE1_ON")
        send_command_and_wait(hw.ser, b'1D-5000\n', expected_response="Dispensing Complete")
        send_command(hw.ser, b"VALVE1_OFF")

        # Pump 2
        send_command(hw.ser, b"DELAY 5000")
        send_command(hw.ser, b'2D5000\n')
        send_command(hw.ser, b"VALVE2_ON")
        send_command_and_wait(hw.ser, b'2D-5000\n', expected_response="Dispensing Complete")
        send_command(hw.ser, b"VALVE2_OFF")
        return jsonify({"status": "success", "message": "Cleaning finished"}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# --- Experiment Orchestration ---

@app.route("/check_experiment_name", methods=["POST"])
def check_experiment_name():
    """Verifies whether target experiment output directory exists, creating it if unique."""
    exp_name = request.get_json().get("exp_name", "").strip()
    if not exp_name:
        return jsonify(
            {"exists": False, "error": "No experiment name provided"}
        ), 400

    exp_dir = Path(BASE_PATH) / exp_name
    exists = exp_dir.exists()

    if not exists:
        exp_dir.mkdir(parents=True, exist_ok=True)

    # Return the full string path so the client/caller can pass it to subsequent steps
    return jsonify({"exists": exists, "path_to_exp_name": str(exp_dir)})


@app.route("/estimate_experiment_time", methods=["POST"])
def estimate_experiment_time():
    """Estimates total runtime duration for planned experiment sequence."""
    try:
        data = request.get_json(force=True) or {}

        # Safely extract and parse numeric inputs with fallbacks
        volumes = np.array(data.get("volumes", [0]), dtype=float)
        seq_len = len(data.get("sequence", []))

        raw_perf_time = data.get("perfusion_time_mean")
        perf_time = float(raw_perf_time) if raw_perf_time is not None and float(raw_perf_time) > 0 else 33.72

        observation_time = float(data.get("observation_time", 0))
        plate_type = int(data.get("plate_type", 6))
        scan_range = float(data.get("range", 0))
        scan_step = float(data.get("step", 1)) if float(data.get("step", 1)) > 0 else 1.0

        # Calculations
        total_acq_time = int(
            (np.sum(volumes) * round(perf_time) * 2)
            + observation_time * seq_len
            + np.sum(volumes) * T_CONSTANT_LIQUID_OUT
            + 10 * seq_len
        )

        time_per_well = (
                12
                + (scan_range / scan_step)
                + 1
                + total_acq_time
                + 1
                + (V_WASH_UL / 1000 * (1 / perf_time))
        )

        total_seconds = int(
            plate_type * time_per_well
            + 12
            + (V_WASH_UL / 1000 * (1 / perf_time))
            + 30
        )

        return jsonify({"estimated_seconds": total_seconds})

    except Exception as e:
        # Prevents HTML 500 responses by returning structured JSON on error
        return jsonify({"error": str(e)}), 500


@app.route("/run_setup", methods=["POST"])
def run_setup():
    """Initializes workspace directory structures and parameter sets for experiment."""
    d = request.json

    # Expect path_to_exp_name from frontend (from /check_experiment_name)
    path_to_exp_name = d.get(
        "path_to_exp_name", str(Path(BASE_PATH) / d["exp_name"])
    )

    path, num_imgs, sol_dict = setup(
        hw.core,
        d["exp_name"],
        path_to_exp_name,
        float(d["observation_time"]),
        int(d["time_interval_ms"]),
        d["solution_names"],
        d["volumes"],
        d["sequence"],
        int(float(d["exposure_ms"])),
        float(d["perfusion_time_mean"]),
        int(d["plate_type"]),
        d["auto_FOV"],
        d["manu_FOV"],
    )

    # Ensure path is explicitly converted to a string for JSON safety
    return jsonify(
        {
            "path_to_exp_name": str(path),
            "num_images_per_well_max": num_imgs,
            "solution_dict": sol_dict,
        }
    )


@app.route("/determine_well_centers", methods=["POST"])
def get_well_centers():
    data = request.get_json() or {}

    n_wells = data.get("n_wells") or data.get("plate_type")
    order_type = data.get("order_type", "snake")
    range_um = float(data.get("range", 100))
    step_size = float(data.get("step", 10))

    if not n_wells:
        return jsonify({"error": "Missing required parameter 'n_wells' or 'plate_type'"}), 400

    well_center_coordinates = []
    autofocus_coordinates = []
    valid_wells = []

    try:
        # Get base well coordinates
        coords = determine_well_centers(n_wells=int(n_wells), order_type=order_type)

        for idx, current_coord in enumerate(coords):
            current_well = idx + 1

            try:
                # Move to well center coordinate
                hw.core.set_xy_position(current_coord[0], current_coord[1])
                while hw.core.device_busy("XYStage"):
                    time.sleep(0.1)

                saf_value = None

                # 1. Coarse SAF: 1000 um range
                try:
                    saf_value = software_autofocus_large_range(hw.core, ser=hw.ser, range_um=1000, step_size=50)
                    if isinstance(saf_value, str):
                        saf_value = None
                except Exception as e:
                    print(f"Coarse SAF (1000 um) failed on well {current_well}: {e}")

                # 2. Coarse SAF: 2000 um range (fallback)
                if saf_value is None:
                    try:
                        saf_value = software_autofocus_large_range(hw.core, ser=hw.ser, range_um=2000, step_size=100)
                        if isinstance(saf_value, str):
                            saf_value = None
                    except Exception as e:
                        print(f"Coarse SAF (2000 um) failed on well {current_well}: {e}")

                # 3. Fine SAF using UI-selected range & step
                if saf_value is not None:
                    try:
                        saf_value = software_autofocus(hw.core, ser=hw.ser, range_um=range_um, step_size=step_size)
                        if isinstance(saf_value, str):
                            saf_value = None
                    except Exception as e:
                        print(f"Fine SAF failed on well {current_well}: {e}")

                # SKIP CONDITION: SAF failure or null value
                if saf_value is None:
                    print(f"[Version C] Well {current_well} skipped - Autofocus failed.")
                    continue

                # Store valid results
                well_center_coordinates.push(current_coord) if hasattr(well_center_coordinates, 'push') else well_center_coordinates.append(current_coord)
                autofocus_coordinates.append(round(saf_value, 2))
                valid_wells.append(current_well)

            except Exception as well_err:
                # Log hardware or movement error and continue to next well
                print(f"[Version C] Exception processing well {current_well}: {well_err}")
                continue

        return jsonify({
            "success": True,
            "well_center_coordinates": well_center_coordinates,
            "autofocus_coordinates": autofocus_coordinates,
            "valid_wells": valid_wells
        }), 200

    except ValueError as err:
        return jsonify({"error": str(err)}), 400
    except Exception as err:
        return jsonify({"error": f"Internal server error: {str(err)}"}), 500

@app.route("/run_perfusion_experiment", methods=["POST"])
def run_perfusion_experiment():
    """Executes perfusion sequence pipeline on targeted single well."""
    d = request.json
    print("well_coords:", d["well_coords"])
    print("saf_coordinates:", d["saf_coordinates"])
    perfusion_sequence(
        hw.core, hw.ser, hw.ecu, int(d["current_well"]), d["exp_name"], d["path_to_exp_name"],
        int(d["num_images_per_well_max"]), d["solution_dict"], d["auto_FOV"], d["manu_FOV"],
        float(d["range"]), float(d["step"]), d["well_coords"], float(d["observation_time"]),
        int(d["time_interval_ms"]), d["volumes"], d["sequence"], int(float(d["exposure_ms"])),
        d["saf_coordinates"], hw.latest_telemetry, hw.telemetry_lock
    )
    return jsonify({"status": "started"})


@app.route("/run_finishing", methods=["POST"])
def run_finishing():
    """Runs final system flush and cleanup routine after experiment completion."""
    d = request.json
    finishing(hw.core, hw.ser, hw.ecu, d["exp_name"], d["path_to_exp_name"], d["well_coords"], hw.latest_telemetry, hw.telemetry_lock)
    return jsonify({"status": "started"})


@app.route("/abort_experiment", methods=["POST"])
def abort_experiment():
    """Triggers immediate hardware abort sequence."""
    hw.abort()
    return jsonify({"status": "aborting"})


# --- Analysis & Data Serving ---

@app.route('/analyze', methods=['POST'])
def analyze_experiment():
    """Triggers offline background segmentation and image processing workers."""
    exp_name = request.form.get('exp_name')
    if not exp_name:
        return jsonify({"status": "error", "message": "Please provide a valid Experiment Name."}), 400

    try:
        exp_dir = Path(BASE_PATH) / exp_name
        analysis_dir = exp_dir / "analysis"
        analysis_dir.mkdir(parents=True, exist_ok=True)

        plate_type = parse_log_file(exp_name)
        wells_to_process = generate_well_list(plate_type)
        processed_grid = []
        ran_new_analysis = False

        for well in wells_to_process:
            well_num, well_label = well['index'], well['label']
            out_json = analysis_dir / f"result_well_{well_num}.json"
            overlay_img = analysis_dir / f"well_{well_num}_overlay.png"
            traj_img = analysis_dir / f"well_{well_num}_trajectory.png"

            if out_json.exists() and overlay_img.exists() and traj_img.exists():
                processed_grid.append({
                    'label': well_label,
                    'overlay_url': f"/data_route/{exp_name}/analysis/well_{well_num}_overlay.png",
                    'trajectory_url': f"/data_route/{exp_name}/analysis/well_{well_num}_trajectory.png"
                })
                continue

            bf_matches = glob.glob(str(exp_dir / f"{exp_name}_well_{well_num}_BF.tif"))
            if not bf_matches:
                continue

            matching_dirs = [item for item in glob.glob(str(exp_dir / f"{exp_name}_well_{well_num}_*")) if Path(item).is_dir()]
            if not matching_dirs:
                continue

            fluo_files = [str(f) for f in Path(matching_dirs[0]).iterdir() if "fluo" in f.name.lower() and f.suffix.lower() in [".tif", ".tiff"]]
            if not fluo_files:
                continue

            try:
                subprocess.run([
                    "python", "cellpose_worker.py", "--mode", "analysis",
                    "--bf_image", bf_matches[0], "--fluo_image", fluo_files[0],
                    "--output", str(out_json), "--well", str(well_num),
                    "--iteration", "0", "--analysis_dir", str(analysis_dir)
                ], check=True)
                ran_new_analysis = True
            except subprocess.CalledProcessError as e:
                print(f"Cellpose tracking runtime failed on well {well_label}: {e}")
                continue

            processed_grid.append({
                'label': well_label,
                'overlay_url': f"/data_route/{exp_name}/analysis/well_{well_num}_overlay.png",
                'trajectory_url': f"/data_route/{exp_name}/analysis/well_{well_num}_trajectory.png"
            })

        summary_img = analysis_dir / "all_wells_means.png"
        if not summary_img.exists():
            try:
                generate_cumulative_plot(analysis_dir)
            except Exception as err:
                print(f"[ANALYSIS ROUTE] Error during summary plotting: {err}")

        return jsonify({"status": "success", "plate_type": plate_type, "grid": processed_grid, "is_cached": not ran_new_analysis}), 200
    except Exception as e:
        print(f"Critical Exception inside analysis loop: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/data_route/<exp_name>/analysis/<filename>')
def serve_analysis_images(exp_name, filename):
    """Serves analysis outputs and generated plot graphics directly."""
    analysis_dir = Path(BASE_PATH) / exp_name / "analysis"
    return send_from_directory(directory=str(analysis_dir), path=filename)


@app.route('/metadata/<exp_name>', methods=['GET'])
def get_experiment_metadata(exp_name):
    """Retrieves experimental metadata parameters."""
    try:
        return jsonify({"status": "success", "plate_type": parse_log_file(exp_name)}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/exit", methods=["POST"])
def exit_app():
    """Cleanly stops hardware routines and terminates Flask application."""
    hw.shutdown()
    func = request.environ.get('werkzeug.server.shutdown')
    if func:
        func()

    # Schedule process exit shortly after returning the HTTP response
    def shutdown_server():
        os._exit(0)

    threading.Timer(0.5, shutdown_server).start()
    return jsonify({"status": "closed"})


# ==========================================
# ENTRY POINT
# ==========================================

if __name__ == "__main__":
    hw.initialize_hardware()
    threading.Timer(1.5, open_browser).start()
    app.run(debug=True, use_reloader=False, host="127.0.0.1", port=5000)