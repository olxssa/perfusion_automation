from dask.array.chunk import argmax
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fov = 1192.8

from pycromanager import start_headless, Core

def pycro_core():
    mm_app_path = 'D:\ProgramFiles\Micro-Manager-2.0_Olyssa'
    config_file = mm_app_path + "\MMConfig_Basler_SOLA_ASI-XYZ_PixelSize_10x_20x.cfg"

    # Start the headless process (Java backend)
    start_headless(mm_app_path, config_file, python_backend=False)

    # Now use Pycro-Manager APIs as normal
    core = Core()

    print("Micromanager configuration loaded successfully.")

    return core

import logging
from ecu import ECUManager
from memetis.pcon.ecui import ECUI

def setup_logging_ecu():

    # Set up logging
    log = logging.getLogger()
    log.setLevel(logging.WARNING)
    logging.getLogger("ecu").setLevel(logging.CRITICAL)
    logHandler = logging.StreamHandler()
    logFormat = logging.Formatter(
        '%(asctime)s %(levelname)s - %(name)s: %(message)s')
    logHandler.setFormatter(logFormat)
    log.addHandler(logHandler)

    # Connect to ECU-I
    manager = ECUManager(accept_unknown_hardware=True)
    ecus = []
    for ecu in manager.get_all():
        if "ECU-I" in ecu.firmware_name:
            ecus.append(ECUI(ecu))
    if not ecus:
        raise RuntimeError("No ECU-I found.")
    for ecu in ecus:
        log.info("ECU UUID Short: %s", ecu.uuid_short)
        log.info("A4 Serial Numbers: %s", ecu.a4_serial_numbers)
    print("ECU connected successfully. Channels available:", ecu.number_of_actuators)

    # Set current profile for each ECU
    # for ecu in ecus:
    #     current_profile = ecu.get_current_profile()
    #     log.info(
    #         "Old current Profile for %s: %s", ecu.uuid_short, current_profile)
    #     ecu.set_current_profile(
    #         peak_current=0,
    #         peak_time=0,
    #         hold_current=500,
    #         hold_time=1,
    #         lock_time=0
    #     )
    #     log.info(
    #         "New current Profile for %s: %s", ecu.uuid_short, ecu.get_current_profile())

    return log, ecu

import serial

import threading
serial_lock = threading.Lock()

def send_command(ser, command):
    # Ensure only one Flask request talks to the serial port at a time
    with serial_lock:

        if ser.write_timeout is None:
            ser.write_timeout = 2.0

        try:
            ser.reset_input_buffer()

            if isinstance(command, str):
                packet = f"{command}\n".encode('utf-8')
            elif isinstance(command, bytes):
                # If it already has a newline, use it as-is; otherwise append it
                packet = command if command.endswith(b'\n') else command + b'\n'
            else:
                raise TypeError("Command must be a string or bytes object")

            ser.reset_input_buffer()

            ser.write(packet)

            ser.flush()
            time.sleep(0.5)  # Small buffer for microcontroller to process
            resp = ser.readline()
            print(resp)
            time.sleep(.1)
        except serial.SerialTimeoutException:
            print(f"Serial Error: Write operation timed out on command {command}!")

            # Try to re-open the port here if it dropped
            ser = reconnect_serial(ser)
            return ser, "Error: Connection dropped and reconnected"

        except Exception as e:
            print(f"Serial Error: {e}")


def reconnect_serial(old_ser):
    """Safely closes the old serial port and attempts to open a new one."""
    if old_ser is not None:
        try:
            old_ser.close()
            print("Closed old serial port reference.")
        except Exception:
            pass  # Ignore errors during forced closure

    time.sleep(1.0)

    print("Attempting to re-initialize serial port...")
    try:
        return setup_serial()
    except Exception as e:
        print(f"Serial Error: {e}")

def setup_serial():
    try:
        ser = serial.Serial('COM6', 9600, timeout=0.1, rtscts=True)
        print("Arduino connected.")
        ser.reset_input_buffer()
        ser.reset_output_buffer()
        time.sleep(2)
        send_command(ser, b"SOCKET_ON\n")
        time.sleep(.2)
        send_command(ser, b"B 50\n")
        time.sleep(2)
        send_command_and_wait(ser, b"A 0\n")
        time.sleep(1)
        send_command_and_wait(ser, b"M 0\n")
        time.sleep(1)
        send_command(ser, b"1S")
        time.sleep(1)
        send_command(ser, b"2S\n")
        return ser
    except Exception as e:
        print(f"Failed to initialize serial port: {e}")
        return None

def setup_serial_esp32():
    ser_flow = serial.Serial(port='COM7', baudrate=115200, timeout=0.1)

    time.sleep(2)

    print("ESP32 Connected.")
    return ser_flow

import cv2
from scipy.optimize import curve_fit

def lorentzian(x, amplitude, xo, gamma, baseline):
    """
    amplitude: Peak height above baseline
    xo: Center of the peak (best Z)
    gamma: Scale parameter (Half-Width at Half-Maximum)
    baseline: Vertical offset
    """
    return baseline + amplitude * (gamma**2 / ((x - xo)**2 + gamma**2))

def software_autofocus_large_range(core, ser, range_um, step_size):
    z_record, focus_scores, z_center = software_autofocus_scores(core, ser, range_um, step_size)

    x, y = np.array(z_record), np.array(focus_scores)
    print("focus_scores =", focus_scores)
    print("z_record =", z_record)

    # Reject if the edges are higher than the center, it's a valley (minimum) instead of a peak (maximum).
    # This usually indicates sensor oversaturation flattening the true focus peak.
    if (y[0] + y[-1]) / 2.0 > y[len(y) // 2]:
        core.set_position(z_center)
        core.wait_for_device("XYStage")
        print("Minimum instead of maximum found indicating a sensor oversaturation."
                         "Please decrease light intensity via shortening the exposure time or decreasing BF "
                         "source brightness. Setting z back to original z.")
        return ("Minimum instead of maximum found indicating a sensor oversaturation."
                         "Please decrease light intensity via shortening the exposure time or decreasing BF "
                         "source brightness. Setting z back to original z.")

    try:
        # The center of the Gaussian (x0) is popt[1]
        best_z = z_record[argmax(focus_scores)]

        fig, ax = plt.subplots(figsize=(6, 4))
        plt.plot(z_record, focus_scores, marker='o')
        plt.plot(best_z, max(focus_scores), marker='x', color='red', markersize=15)
        plt.xlabel("Absolute z Position (µm)")
        plt.ylabel("Focus Score")
        plt.title(f"Range: {range_um} µm, Step Size: {step_size} µm")
        plt.tight_layout()
        plt.show()

        print("best z calculated large range: ", best_z)
        core.set_position(best_z)
        core.wait_for_device("XYStage")

        return best_z

    except (RuntimeError, ValueError) as e:
        print(f"Autofocus failed due to: {e}. Setting z back to original z.")
        core.set_position(z_center)
        core.wait_for_device("XYStage")
        print("Autofocus failed - no maximum found. Setting z back to original z. "
            "Please try focusing using the Live button and z wheel first and using different parameters.")
        return (
            "Autofocus failed - no maximum found. Setting z back to original z. "
            "Please try focusing using the Live button and z wheel first and using different parameters."
        )

def software_autofocus(core, ser, range_um, step_size):
    z_record, focus_scores, z_center = software_autofocus_scores(core, ser, range_um, step_size)

    x, y = np.array(z_record), np.array(focus_scores)
    print("focus_scores =", focus_scores)
    print("z_record =", z_record)

    # Estimate initial parameters including the baseline offset
    baseline_est = np.min(y)
    amplitude_est = np.max(y) - baseline_est
    mean_est = x[np.argmax(y)]
    gamma_est = step_size * 0.5  # Lorentzians have narrower peaks

    p0 = [amplitude_est, mean_est, gamma_est, baseline_est]
    bounds = (
        [0, x.min(), 0, 0],
        # Restrict max gamma to a fraction of the range to force a sharp peak
        [amplitude_est * 3, x.max(), step_size * 2.0, baseline_est * 1.5]
    )

    # 1. Reject if the edges are higher than the center, it's a valley (minimum) instead of a peak (maximum).
    # This usually indicates sensor oversaturation flattening the true focus peak.
    if (y[0] + y[-1]) / 2.0 > y[len(y) // 2]:
        core.set_position(z_center)
        core.wait_for_device("XYStage")
        print("Minimum instead of maximum found indicating a sensor oversaturation."
                         "Please decrease light intensity via shortening the exposure time or decreasing BF "
                         "source brightness. Setting z back to original z.")
        return("Minimum instead of maximum found indicating a sensor oversaturation."
                         "Please decrease light intensity via shortening the exposure time or decreasing BF "
                         "source brightness. Setting z back to original z.")

    try:
        # Fit using the updated 4-parameter model with bounds
        popt = curve_fit(lorentzian, x, y, p0=p0, bounds=bounds)[0]
        print("popt =", popt)
        # The center of the Gaussian (x0) is popt[1]
        best_z = float(popt[1])
        fit_amplitude = popt[0]

        # If it passes validation, proceed with plotting and moving
        x_fine = np.linspace(x.min(), x.max(), 200)
        y_fine = lorentzian(x_fine, *popt)

        fig, ax = plt.subplots(figsize=(6, 4))
        plt.plot(z_record, focus_scores, marker='o')
        plt.plot(best_z, np.max(y_fine), marker='x', color='red', markersize=15)
        plt.plot(x_fine, y_fine, linestyle='--', color='green')
        plt.xlabel("absolute z Position")
        plt.ylabel("focus score")
        plt.title(f"Range: {range_um} µm, Step Size: {step_size} µm")
        plt.tight_layout()
        plt.show()

        # 2. Reject if the computed peak is on the absolute edge of the scan range
        edge_tolerance = step_size * 1.5
        if (best_z <= x.min() + edge_tolerance) or (best_z >= x.max() - edge_tolerance):
            core.set_position(z_center)
            core.wait_for_device("XYStage")
            print("Calculated peak is on the edge of the scan range. True focus is likely outside this window. "
                "Setting z back to original z.")
            return(
                "Calculated peak is on the edge of the scan range. True focus is likely outside this window. "
                "Setting z back to original z.")

        ## 3. Reject if the peak isn't distinct enough from the noise floor (here at least 5 % above baseline)
        #if fit_amplitude < (popt[3] * 0.05):
        #    core.set_position(z_center)
        #    core.wait_for_device("XYStage")
        #    return ("Peak prominence is too low; data looks like background noise. "
        #                     "True focus is likely outside this window. Setting z back to original z.")

        print("best z calculated: ", best_z)
        core.set_position(best_z)
        core.wait_for_device("XYStage")

    except (RuntimeError, ValueError) as e:
        print(f"Autofocus failed due to: {e}. Setting z back to original z.")
        core.set_position(z_center)
        core.wait_for_device("XYStage")
        return (
            "Autofocus failed - no maximum found. Setting z back to original z. "
            "Please try focusing using the Live button and z wheel first and using different parameters."
        )

def software_autofocus_scores(core, ser, range_um, step_size):
    z_center = core.get_position()
    print("Z center:", z_center)

    current_exposure = core.get_exposure()
    #core.set_exposure(50)

    focus_scores, z_record = [], []
    z_start, z_end = z_center - range_um * 0.5, z_center + range_um * 0.5
    z_positions = np.arange(z_start, z_end + step_size, step_size)

    def image_process_fn(image, metadata):
        length = 3400
        z = z_positions[metadata['Axes']['z']]
        rand = (image.shape[0] - length) / 2
        c1, c2 = int(rand), int(rand + length)
        arr_cropped = image[c1:c2, c1:c2]

        # Apply smoothing to suppress pure high-frequency camera noise
        im_filtered = cv2.medianBlur(arr_cropped, 3)
        im_filtered = cv2.GaussianBlur(arr_cropped, (5, 5), 0)
        # Compute variance of Laplacian on un-normalized data
        score = cv2.Laplacian(im_filtered, cv2.CV_64F, ksize=3).var()

        focus_scores.append(score)
        z_record.append(z)
        return image, metadata

    events = multi_d_acquisition_events(z_start=z_start, z_end=z_end, z_step=step_size)

    send_command(ser, b"C 255 255 255\n")
    time.sleep(.1)
    with Acquisition(image_process_fn=image_process_fn, show_display=False, save_dir=r"D:\Automated_Bath_Perfusion_Experiments\SAF_tests") as acq:
        acq.acquire(events)
    send_command(ser, b"C 0 0 0\n")

    core.set_exposure(current_exposure)

    return z_record, focus_scores, z_center

import json
import subprocess
import cv2
from scipy.optimize import curve_fit
from pathlib import Path

BASE_PATH = r"D:\Automated_Bath_Perfusion_Experiments"

def determine_fluorescent_cells_count(ser, core, exp_name, current_well, iteration):
    """
    Snaps images, saves them to disk, and calls the isolated Cellpose worker
    while tracking the current well and iteration.
    """

    path_to_exp_name = Path(BASE_PATH) / exp_name / "segmentation"
    path_to_exp_name.mkdir(parents=True, exist_ok=True)

    bf_path = os.path.join(path_to_exp_name, rf"bf_well{current_well}_fov{iteration}.tif")
    fluo_path = os.path.join(path_to_exp_name, rf"fluo_well{current_well}_fov{iteration}.tif")
    output_json = os.path.join(path_to_exp_name, rf"result_well{current_well}.json")

    # Clean up old results file
    if os.path.exists(output_json):
        os.remove(output_json)

    # 1. Snap and transform your images
    im_bf = bf_snap_image_transform_crop(ser, core)
    im_fluo = fluo_snap_image_transform_crop(core)

    # 2. Save live frames to disk
    tifffile.imwrite(bf_path, im_bf)
    tifffile.imwrite(fluo_path, im_fluo)

    # 3. Run the Cellpose analyzer (Passing well and iteration explicitly)
    print(f"Launching isolated Cellpose process for Well {current_well}, Iteration {iteration}/8")
    try:
        subprocess.run([
            "python", "cellpose_worker.py",
            "--mode", "segmentation",
            "--bf_image", bf_path,
            "--fluo_image", fluo_path,
            "--output", str(output_json),
            "--well", str(current_well),
            "--iteration", "0"
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Cellpose worker failed with error: {e}")
        return 0, 0

    # 4. Read the results back
    if os.path.exists(output_json):
        with open(output_json, "r") as f:
            data = json.load(f)
        return data["total_cells"], data["fluorescent_cell_count"]
    else:
        print("Error: Worker finished but no JSON output was found.")
        return 0, 0

def determine_best_fov(ser, core, fov_um, current_coord, range_um, step_size, current_well, exp_name):
    core.set_xy_position(current_coord[0], current_coord[1])
    while True:
        if not core.device_busy("XYStage"):
            break
        time.sleep(1)

    new_coords = None
    saf_value = None

    try:
        # Large range 1000 um
        saf_value = software_autofocus_large_range(core, ser=ser, range_um=1000, step_size=50)
        if isinstance(saf_value, str):
            saf_value = None

        # Try again large range 2000 um
        if saf_value is None:
            saf_value = software_autofocus_large_range(core, ser=ser, range_um=2000, step_size=100)
            if isinstance(saf_value, str):
                saf_value = None
        print("SAF value:", saf_value)

        # Fine range according to chosen parameters
        if saf_value is not None:
            saf_value = software_autofocus(core, ser=ser, range_um=range_um, step_size=step_size)
            if isinstance(saf_value, str):
                saf_value = None
        print("SAF value:", saf_value)
    except Exception as e:
        print(f"SAF value error: {e}")

    # Check if Autofocus failed
    if saf_value is None:
        print(f"Error: SAF value was not found for well {current_well}.")
        return None, None

    current_x, current_y = current_coord[0], current_coord[1]

    directions = [(1, 0), (0, -1), (-1, 0), (0, 1)]
    dir_idx = 0
    step_length = 1
    steps_taken_in_this_length = 0
    leg_count = 0

    for i in range(1):
        total_cells, fluorescent_cell_count = determine_fluorescent_cells_count(ser, core, exp_name, current_well, i)
        print(i, total_cells, fluorescent_cell_count)

        if (fluorescent_cell_count > 20) and (fluorescent_cell_count < 80):
            new_coords = [core.get_x_position(), core.get_y_position()]
            break
        else:
            dx, dy = directions[dir_idx]
            current_x += dx * fov_um
            current_y += dy * fov_um
            core.set_xy_position(current_x, current_y)

            while True:
                if not core.device_busy("XYStage"):
                    break
                time.sleep(1)

            steps_taken_in_this_length += 1
            if steps_taken_in_this_length == step_length:
                steps_taken_in_this_length = 0
                dir_idx = (dir_idx + 1) % 4
                leg_count += 1
                if leg_count == 2:
                    step_length += 1
                    leg_count = 0

    # Check if FOV search failed
    if new_coords is None:
        print(f"No best coordinates were found for well {current_well}.")
        return None, None
    else:
        return new_coords, saf_value

import random
import string

def determine_best_well_center_coords(n_wells, current_well, ser, core, fov_um, order_type, range_um, step_size, exp_name, auto_FOV=False):
    coords = determine_well_centers(n_wells, order_type)
    current_coord = coords[current_well - 1]
    print("Current coord: ", current_coord)

    if auto_FOV:
        current_coord, saf_value = determine_best_fov(ser, core, fov_um, current_coord, range_um, step_size, current_well, exp_name)
        print("[[current_coord], saf_value]", [[current_coord], saf_value])
        return [[current_coord], saf_value]
    else:
        path_to_exp_name = Path(BASE_PATH) / exp_name / "segmentation"
        path_to_exp_name.mkdir(parents=True, exist_ok=True)
        return current_coord

def auto_contrast(img):

    img = img.astype(np.float32)

    p_low, p_high = np.percentile(img, (0.01, 99.99))

    # prevent division by zero
    if p_high <= p_low:
        return np.zeros_like(img, dtype=np.uint8)

    # clip first
    img = np.clip(img, p_low, p_high)

    # scale to 0-255 safely
    img = (img - p_low) / (p_high - p_low)
    img = np.clip(img, 0, 1)  # clamp NaNs / out-of-range
    img = (img * 255).astype(np.uint8)

    return img

def tube_position(ser, tube_pos):
    try:
        if tube_pos == "down":
            ser.write(b"A 145\n")
            ser.flush()
            time.sleep(1)
            resp = ser.readline()
            print(resp)
            time.sleep(.1)
        else:
            ser.write(b"A 75\n")
            ser.flush()
            time.sleep(1)
            resp = ser.readline()
            print(resp)
            time.sleep(.1)

        ser.flush()

    except Exception as e:
        print(f"Serial Error in vacuum: {e}")

T_CONSTANT_LIQUID_OUT = 5 # at Delay 10000
V_WASH_UL = 1000 # ul (internal volume of system) 750 ul

# ECU Channel Assignment Constants
RES1 = 1
RES2 = 2
RES3 = 6
RES4 = 7
RES5 = 8
RES6 = 3
RES_WASH = 4
RES_SOLS = 5

# Map reservoir number directly to physical ECU pins
RES_MAP = {
    1: RES1,  # Pin 1
    2: RES2,  # Pin 2
    3: RES3,  # Pin 6
    4: RES4,  # Pin 7
    5: RES5,  # Pin 8
    6: RES6,  # Pin 3
}

def flow_rate_calibration(reservoir_no, ecu, ser, latest_telemetry, telemetry_lock, target_volume_ul=V_WASH_UL):
    send_command_and_wait(ser, b"A 0\n")
    time.sleep(0.5)
    send_command_and_wait(ser, b"M 80\n")
    send_command(ser, b"DELAY 15000\n")
    send_command(ser, b"1F\n")

    time.sleep(1)

    # Enable physical target channel
    physical_pin = RES_MAP[reservoir_no]
    ecu.enable(physical_pin)

    # Route based on physical pin ID (Pins 1, 2, and 3 use RES_WASH)
    companion_pin = RES_WASH if physical_pin in [1, 2, 3] else RES_SOLS
    ecu.enable(companion_pin)

    perfusion_time = perfuse_volume(ecu, target_volume_ul, latest_telemetry, telemetry_lock)

    send_command(ser, b"1S\n")
    send_command_and_wait(ser, b"M 0\n")

    return perfusion_time

def perfuse_volume(ecu, target_volume_ul, latest_telemetry, telemetry_lock):
    t0 = time.time()
    current_volume = 0.0
    last_time = time.time()
    flow_rates = []

    try:
        while current_volume < target_volume_ul:
            # Safely check the dictionary reference passed in
            with telemetry_lock:
                flow_rate = latest_telemetry["flow_rate"]

            if flow_rate > 10000:
                time.sleep(0.01)
                continue

            flow_rates.append(flow_rate)

            now = time.time()
            dt = now - last_time
            last_time = now

            added_volume = (flow_rate / 60.0) * dt
            current_volume += added_volume

            #print(f"Flow: {flow_rate:.2f} | Total: {current_volume:.2f}/{target_volume_ul} ul", end='\r')
            time.sleep(0.02)

    finally:
        ecu.disable_all()
        #mean_rate = np.mean(flow_rates) if flow_rates else 0.0
        #std_rate = np.std(flow_rates) if flow_rates else 0.0
        #print(
        #    f"Stopped. Flow-rate: {mean_rate:.2f} SD: {std_rate:.2f} | Total: {current_volume:.2f}/{target_volume_ul} ul",
        #    end='\r')

        return time.time() - t0

def solution(input_channel, ecu, target_volume_ul, latest_telemetry, telemetry_lock):
    physical_pin = RES_MAP[input_channel]

    ecu.enable(physical_pin)
    if physical_pin in [1, 2, 3]:
        ecu.enable(RES_WASH)
    else:
        ecu.enable(RES_SOLS)
    perfuse_volume(ecu, target_volume_ul, latest_telemetry, telemetry_lock)

def vacuum(ser, time_liquid_out):
    send_command(ser, b"DELAY 10000\n")
    send_command(ser, b"2F\n")

    time.sleep(time_liquid_out+2) # additional 2 s

    send_command(ser, b"2S\n")

def set_well_position(core, ser, well_coords):
    send_command_and_wait(ser, b"A 75\n")
    core.set_xy_position(well_coords[0], well_coords[1])
    while True:
        if not core.device_busy("XYStage"):
            break
        time.sleep(1)

import os, time, math, csv
import numpy as np
import tifffile
import threading
from pycromanager import Acquisition, multi_d_acquisition_events

def background_imaging_fixed_number(time_interval_ms, directory, exp_name, well_index, num_images):
    well_name = f"{exp_name}_well_{well_index + 1}_fluo70P"

    with Acquisition(directory=directory, name=well_name, saving_queue_size=100, show_display=False) as acq:
        events = multi_d_acquisition_events(
            num_time_points=num_images,
            time_interval_s=time_interval_ms / 1000,
        )
        acq.acquire(events)

def background_imaging_dynamic(
    time_interval_ms,
    path_to_exp_name,
    exp_name,
    well_index,
    stop_event,
    max_images,
):
    # Ensure directory is a string representation of the target path
    exp_dir = str(Path(path_to_exp_name))
    well_name = f"{exp_name}_well_{well_index + 1}_fluo70P"
    image_count = 0

    print(f"Max images scheduled: {max_images}")
    print(f"Saving acquisition sequence to directory: {exp_dir}")

    # Pycromanager creates a subfolder named `well_name` inside `directory`
    with Acquisition(
        directory=exp_dir, name=well_name, show_display=False
    ) as acq:
        while not stop_event.is_set() and image_count < max_images:
            start_time = time.time()

            event = multi_d_acquisition_events(num_time_points=1)[0]
            acq.acquire(event)

            image_count += 1

            # Account for acquisition execution time during sleep
            elapsed = time.time() - start_time
            sleep_time = max(0, (time_interval_ms / 1000) - elapsed)
            time.sleep(sleep_time)

        time.sleep(1)
        print(f"Imaging finished for {well_name}. Total images: {image_count}")

def log_event(log_file_path, message, value=""):
    """Centralized logger ensuring entries go into the experiment directory."""
    with open(log_file_path, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([datetime.now(), message, value])

def setup(
    core,
    exp_name,
    path_to_exp_name,
    observation_time,
    time_interval_ms,
    solution_names,
    volumes,
    sequence,
    exposure_ms,
    perfusion_time_mean,
    plate_type,
    auto_FOV,
    manu_FOV,
):
    exp_dir = Path(path_to_exp_name)
    log_file_path = exp_dir / f"{exp_name}_log.csv"

    core.set_exposure(exposure_ms)

    log_event(log_file_path, "Starting Perfusion Experiment: ", exp_name)
    print("Starting perfusion experiment: ", exp_name)
    log_event(log_file_path, "Saving all files to: ", str(exp_dir))
    log_event(log_file_path, "Plate-type: ", plate_type)
    log_event(log_file_path, "Solution names: ", solution_names)

    volumes_ul = [x * 1000 for x in volumes]
    log_event(log_file_path, "Volumes [ul]: ", volumes_ul)
    log_event(log_file_path, "Sequence: ", sequence)

    if auto_FOV:
        log_event(log_file_path, "FOV-determination via: AutoFOV")
    elif manu_FOV:
        log_event(log_file_path, "FOV-determination via: ManuFOV")
    else:
        log_event(log_file_path, "FOV-determination via: Well-center")

    total_acq_time = int(
        (np.sum(volumes) * perfusion_time_mean * 2)
        + observation_time * len(sequence)
        + np.sum(volumes) * T_CONSTANT_LIQUID_OUT
        + 10 * len(sequence)
    )
    print("Total acq time:", total_acq_time)

    num_images_per_well_max = math.ceil(
        total_acq_time * 1000 / time_interval_ms
    )
    print("Number of images per well:", num_images_per_well_max)

    solution_dict = {i + 1: value for i, value in enumerate(solution_names)}
    print("Solution dictionary:", solution_dict)

    return str(exp_dir), num_images_per_well_max, solution_dict


def perfusion_sequence(
    core,
    ser,
    ecu,
    current_well,
    exp_name,
    path_to_exp_name,
    num_images_per_well_max,
    solution_dict,
    auto_FOV,
    manu_FOV,
    range_um,
    step_size,
    well_coords,
    observation_time,
    time_interval_ms,
    volumes,
    sequence,
    exposure_ms,
    saf_coordinates,
    latest_telemetry,
    telemetry_lock,
):
    exp_dir = Path(path_to_exp_name)
    log_file_path = exp_dir / f"{exp_name}_log.csv"

    volumes_ul = [x * 1000 for x in volumes]
    solution_dict = {int(k): v for k, v in solution_dict.items()}

    i = current_well - 1

    log_event(log_file_path, f"Starting with well", i + 1)
    print("Starting with well", i + 1)

    # Set well position
    set_well_position(core, ser, well_coords[0])

    # Autofocus
    core.set_position(saf_coordinates[0])
    while True:
        if not core.device_busy("XYStage"):
            break
        time.sleep(1)

    # Snap BF reference image of cells
    send_command(ser, b"C 255 255 255\n")
    core.snap_image()
    java_img = core.get_tagged_image()
    arr = np.reshape(
        java_img.pix, newshape=[java_img.tags["Height"], java_img.tags["Width"]]
    )

    # File path targeting the subfolder directly
    bf_img_path = exp_dir / f"{exp_name}_well_{i + 1}_BF.tif"
    tifffile.imwrite(bf_img_path, np.array(arr))
    send_command(ser, b"C 0 0 0\n")

    log_event(
        log_file_path,
        f"Starting imaging for well {i + 1}, exposure time: {exposure_ms}, "
        f"imaging frequency: {time_interval_ms} ms, observation time: {observation_time} s",
    )
    print(
        f"Starting imaging for well {i + 1}, exposure time: {exposure_ms}, "
        f"imaging frequency: {time_interval_ms} ms, observation time: {observation_time} s"
    )

    core.set_config("fluorescence", "on_70P")

    stop_imaging = threading.Event()

    # Pass the explicit exp_dir path down into background_imaging_dynamic
    img_thread = threading.Thread(
        target=background_imaging_dynamic,
        args=(
            time_interval_ms,
            exp_dir,
            exp_name,
            i,
            stop_imaging,
            num_images_per_well_max,
        ),
    )
    img_thread.start()

    for s in range(len(sequence)):
        # Flushing fluidic system with solution
        #log_event(log_file_path,
        #          f"Flushing system with {volumes_ul[s]} ul of solution", solution_dict[sequence[s]])
        #print(f"Flushing system with {volumes_ul[s]} ul of solution", solution_dict[sequence[s]])
        #send_command_and_wait(ser, b"A 0\n")
        #send_command(ser, b'DELAY 20000\n')
        #send_command(ser, b'1F\n')
        #send_command_and_wait(ser, b"M 60")
        #solution(input_channel=sequence[s], ecu=ecu, target_volume_ul=V_WASH_UL,
        #         latest_telemetry=latest_telemetry, telemetry_lock=telemetry_lock)
        #send_command_and_wait(ser, b"M 0")
        #send_command(ser, b'1S\n')

        # Insert tubes into well
        send_command_and_wait(ser, b"A 145\n")

        # Vacuuming out old solution
        log_event(
            log_file_path, f"Vacuuming solution", solution_dict[sequence[s]]
        )
        print(
            "Vacuuming solution",
            solution_dict[sequence[s]],
            "for time",
            volumes[s] * T_CONSTANT_LIQUID_OUT,
            "s",
        )
        time_liquid_out = round(volumes[s] * T_CONSTANT_LIQUID_OUT, 1)
        vacuum(ser, time_liquid_out=time_liquid_out)

        # Perfuse solution
        log_event(
            log_file_path,
            f"Perfusing {volumes_ul[s]} ul of solution",
            solution_dict[sequence[s]],
        )
        print(
            f"Perfusing {volumes_ul[s]} ul of solution",
            solution_dict[sequence[s]],
        )
        solution(
            input_channel=sequence[s],
            ecu=ecu,
            target_volume_ul=volumes_ul[s],
            latest_telemetry=latest_telemetry,
            telemetry_lock=telemetry_lock,
        )

        # Observe cells for observation_time
        log_event(
            log_file_path,
            f"Starting acquisition time after perfusion",
            observation_time,
        )
        print(
            f"Starting acquisition time after perfusion", observation_time, "s \n"
        )
        time.sleep(observation_time)

    log_event(log_file_path, f"Stopping imaging for well", i + 1)
    print("Stopping imaging for well", i + 1)
    stop_imaging.set()
    img_thread.join()

    core.set_config("fluorescence", "off")

    print("\n")
    log_event(log_file_path, f"Flushing {V_WASH_UL:.0f} ul {solution_dict[2]} to waste")
    print(f"Flushing {V_WASH_UL:.0f} ul {solution_dict[2]} to waste")
    perf_time = flow_rate_calibration(
        2,
        ecu,
        ser,
        latest_telemetry,
        telemetry_lock,
        target_volume_ul=V_WASH_UL,
    )
    print(perf_time)

    print("Perfusion sequence complete for well", i + 1)


def finishing(
    core,
    ser,
    ecu,
    exp_name,
    path_to_exp_name,
    well_coords,
    latest_telemetry,
    telemetry_lock,
):
    exp_dir = Path(path_to_exp_name)
    log_file_path = exp_dir / f"{exp_name}_log.csv"

    set_well_position(core, ser, well_coords[0])

    log_event(log_file_path, f"Flushing {V_WASH_UL:.0f} ul wash to waste")
    print(f"Flushing {V_WASH_UL:.0f} ul wash to waste")
    flow_rate_calibration(
        1,
        ecu,
        ser,
        latest_telemetry,
        telemetry_lock,
        target_volume_ul=V_WASH_UL,
    )

    log_event(log_file_path, f"Experiment completed")
    print("Experiment completed")

def get_sensor_tendency(w1_readings, w2_readings):
    """
    Evaluates historical snapshots for both Waste 1 and Waste 2 reservoirs
    and returns localized alert objects indicating which waste is full.
    """
    if len(w1_readings) < 3 or len(w2_readings) < 3:
        raise RuntimeError("Waiting for background thread to aggregate stable sensor data.")

    alerts = []

    # 1. Evaluate Waste Reservoir 1
    w1_avg = sum(w1_readings) / len(w1_readings)
    if w1_avg <= 0.5:
        alerts.append({
            "waste_id": 1
        })

    # 2. Evaluate Waste Reservoir 2
    w2_avg = sum(w2_readings) / len(w2_readings)
    if w2_avg <= 0.5:
        alerts.append({
            "waste_id": 2
        })

    return alerts

def bf_snap_image_transform_crop(ser, core, length=3400):
    send_command(ser, b"C 255 255 255\n")
    time.sleep(.1)
    core.snap_image()
    java_img = core.get_tagged_image()
    arr = np.reshape(java_img.pix, newshape=[java_img.tags['Height'], java_img.tags['Width']])
    send_command(ser, b"C 0 0 0\n")
    rand = (arr.shape[0] - length) / 2
    c1, c2 = int(rand), int(rand + length)
    arr_cropped = arr[c1:c2, c1:c2]
    return arr_cropped

def fluo_snap_image_transform_crop(core, length=3400):
    core.set_config("fluorescence", "on_70P")
    time.sleep(.1)
    core.snap_image()
    java_img = core.get_tagged_image()
    arr = np.reshape(java_img.pix, newshape=[java_img.tags['Height'], java_img.tags['Width']])
    core.set_config("fluorescence", "off")
    rand = (arr.shape[0] - length) / 2
    c1, c2 = int(rand), int(rand + length)
    arr_cropped = arr[c1:c2, c1:c2]
    return arr_cropped

def determine_well_centers(n_wells, order_type="snake"):
    layout_map = {
        6: {
            "rows": 2,
            "cols": 3,
            "pitch": 39200,
            "a1": (0, 0),
        },
        12: {
            "rows": 3,
            "cols": 4,
            "pitch": 26010,
            "a1": (1234.3, -7228.5),
        },
        24: {
            "rows": 4,
            "cols": 6,
            "pitch": 19300,
            "a1": (-4357.6, -7371),
        },
    }

    if n_wells not in layout_map:
        raise ValueError("Only 6- and 24-well plates are supported.")

    # Extract layout
    rows = layout_map[n_wells]["rows"]
    cols = layout_map[n_wells]["cols"]
    pitch = layout_map[n_wells]["pitch"]
    x0, y0 = layout_map[n_wells]["a1"]

    well_coords = [
        (x0 + c * pitch, y0 + r * pitch)
        for r in range(rows)
        for c in range(cols)
    ]

    row_labels = string.ascii_uppercase
    well_names = [
        f"{row_labels[r]}{c + 1}"
        for r in range(rows)
        for c in range(cols)
    ]

    if order_type == "row-major":
        order_idx = list(range(len(well_names)))

    elif order_type == "column-major":
        order_idx = [r * cols + c for c in range(cols) for r in range(rows)]

    elif order_type == "snake":
        order_idx = []
        for r in range(rows):
            if r % 2 == 0:  # left → right
                order_idx.extend([r * cols + c for c in range(cols)])
            else:  # right → left
                order_idx.extend([r * cols + c for c in reversed(range(cols))])

    elif order_type == "random":
        order_idx = list(range(len(well_names)))
        random.shuffle(order_idx)

    else:
        raise ValueError("order_type must be 'row-major', 'column-major', 'snake', or 'random'")

    coords = [well_coords[i] for i in order_idx]

    return coords


import re
from datetime import datetime
from pathlib import Path


def parse_well_timings(log_path, well_num):
    """
    Parses the experiment log to find:
    1. The imaging frequency (in seconds) for the well.
    2. The relative times (in seconds) from imaging start when each solution was perfused or vacuumed.
    """
    if not Path(log_path).exists():
        return 1.0, []  # default 1s step, no events

    start_time = None
    img_freq_s = 1.0  # default
    events = []  # list of dicts: {"time": float, "solution": str, "type": str}

    # Time parser helper with quotes stripping and fallback formatting
    def parse_dt(ts_str):
        cleaned = ts_str.strip().strip('"').strip("'")  # Strip outer quotes
        try:
            return datetime.strptime(cleaned, "%Y-%m-%d %H:%M:%S.%f")
        except ValueError:
            # Fallback for timestamps without microseconds
            return datetime.strptime(cleaned, "%Y-%m-%d %H:%M:%S")

    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    well_active = False
    for line in lines:
        parts = line.split(',', 2)
        if len(parts) < 2:
            continue

        timestamp_str = parts[0]
        message = parts[1].strip().lower().strip('"')
        detail = parts[2].strip().strip('"') if len(parts) > 2 else ""

        # Scope parsing to the specific well
        if "starting with well" in message and detail.strip() == str(well_num):
            well_active = True
            continue
        elif "stopping imaging for well" in message and detail.strip() == str(well_num):
            well_active = False
            continue

        if well_active:
            # 1. Capture imaging start time and frequency
            if "starting imaging for well" in message or "starting imaging for well" in line.lower():
                try:
                    start_time = parse_dt(timestamp_str)
                    # Extract frequency using regex: e.g., "imaging frequency: 1000 ms"
                    freq_match = re.search(r"imaging frequency:\s*(\d+)\s*ms", line, re.IGNORECASE)
                    if freq_match:
                        img_freq_s = float(freq_match.group(1)) / 1000.0
                except Exception as e:
                    print(f"Error parsing start time on line '{line.strip()}': {e}")

            # 2. Capture perfusion times
            elif "perfusing" in message and start_time is not None:
                try:
                    perf_time = parse_dt(timestamp_str)
                    elapsed_seconds = (perf_time - start_time).total_seconds()
                    # Clean solution name
                    sol_name = parts[-1].strip().strip('"')
                    events.append({"time": elapsed_seconds, "solution": sol_name, "type": "perfusion"})
                except Exception as e:
                    print(f"Error parsing perfusion step on line '{line.strip()}': {e}")

            # 3. Capture vacuuming times
            elif "vacuuming" in message and start_time is not None:
                try:
                    vac_time = parse_dt(timestamp_str)
                    elapsed_seconds = (vac_time - start_time).total_seconds()
                    events.append({"time": elapsed_seconds, "solution": "aspirating", "type": "vacuum"})
                except Exception as e:
                    print(f"Error parsing vacuuming step on line '{line.strip()}': {e}")

    return img_freq_s, events


def get_global_y_limits(analysis_dir, default_lims=(-0.1, 2.0)):
    """
    Scans finished well result files and finds the absolute minimum and maximum
    values across ALL individual cell trajectories across ALL wells.
    This ensures that individual cell plots and cumulative plots share the exact same scale
    without clipping any cell's response.
    """
    analysis_dir = Path(analysis_dir)
    result_files = list(analysis_dir.glob("result_well_*.json"))

    g_min = float('inf')
    g_max = float('-inf')
    found_any = False

    for file_path in result_files:
        try:
            with open(file_path, 'r') as f:
                well_data = json.load(f)

            trajectories = well_data.get("trajectories", {})

            # Robust label selector (supports legacy and new schemas)
            fluo_labels = well_data.get("fluorescent_cells_valid", [])
            if not fluo_labels:
                fluo_labels = well_data.get("fluorescent_cells", [])
            if not fluo_labels and trajectories:
                fluo_labels = list(trajectories.keys())

            # Check the absolute min/max of every single active cell trajectory
            for label in fluo_labels:
                if label in trajectories:
                    traj_data = trajectories[label]
                    if traj_data:
                        g_min = min(g_min, np.min(traj_data))
                        g_max = max(g_max, np.max(traj_data))
                        found_any = True
        except Exception as e:
            print(f"[LIMITS FINDER] Error reading {file_path.name}: {e}")
            continue

    if not found_any:
        print(f"[LIMITS FINDER] No trajectory points gathered. Defaulting to: {default_lims}")
        return default_lims

    # Calculate a 15% safety margin
    spread = g_max - g_min if g_max != g_min else 1.0
    padding = spread * 0.15

    computed_limits = (float(g_min - padding), float(g_max + padding))
    print(f"[LIMITS FINDER] Unified absolute scaling computed from all cells/wells: {computed_limits}")
    return computed_limits


def generate_cumulative_plot(analysis_dir):
    """
    Scans the analysis directory for completed 'result_well_*.json' files,
    isolates fluorescent trajectories, and averages them per well.
    """
    analysis_dir = Path(analysis_dir)
    result_files = list(analysis_dir.glob("result_well_*.json"))

    print(f"[PLOTTER] Starting cumulative plot. Found result files: {[f.name for f in result_files]}")

    if not result_files:
        print("[PLOTTER] Aborting: No result_well_*.json files found in analysis directory.")
        return False

    # --- INLINE LOG PARSING HELPER (Aligned with parse_well_timings) ---
    def parse_well_timings_inline(log_path, well_num):
        from datetime import datetime
        log_path = Path(log_path)
        if not log_path.exists():
            print(f"[PLOTTER] Warning: Log file not found at {log_path}. Using default 1s steps.")
            return 1.0, []

        start_time = None
        img_freq_s = 1.0
        events = []

        def parse_dt(ts_str):
            cleaned = ts_str.strip().strip('"').strip("'")
            try:
                return datetime.strptime(cleaned, "%Y-%m-%d %H:%M:%S.%f")
            except ValueError:
                return datetime.strptime(cleaned, "%Y-%m-%d %H:%M:%S")

        try:
            with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
        except Exception as e:
            print(f"[PLOTTER] Failed to read log file: {e}")
            return img_freq_s, events

        well_active = False
        for line in lines:
            parts = line.split(',', 2)
            if len(parts) < 2:
                continue

            timestamp_str = parts[0]
            message = parts[1].strip().lower().strip('"')
            detail = parts[2].strip().strip('"') if len(parts) > 2 else ""

            if "starting with well" in message and detail.strip() == str(well_num):
                well_active = True
                continue
            elif "stopping imaging for well" in message and detail.strip() == str(well_num):
                well_active = False
                continue

            if well_active:
                if "starting imaging for well" in message or "starting imaging for well" in line.lower():
                    try:
                        start_time = parse_dt(timestamp_str)
                        freq_match = re.search(r"imaging frequency:\s*(\d+)\s*ms", line, re.IGNORECASE)
                        if freq_match:
                            img_freq_s = float(freq_match.group(1)) / 1000.0
                    except Exception as e:
                        print(f"[PLOTTER] Error parsing start time for well {well_num}: {e}")

                elif "perfusing" in message and start_time is not None:
                    try:
                        perf_time = parse_dt(timestamp_str)
                        elapsed_seconds = (perf_time - start_time).total_seconds()
                        sol_name = parts[-1].strip().strip('"')
                        events.append({"time": elapsed_seconds, "solution": sol_name, "type": "perfusion"})
                    except Exception as e:
                        print(f"[PLOTTER] Error parsing perfusion step on line '{line.strip()}': {e}")

                elif "vacuuming" in message and start_time is not None:
                    try:
                        vac_time = parse_dt(timestamp_str)
                        elapsed_seconds = (vac_time - start_time).total_seconds()
                        events.append({"time": elapsed_seconds, "solution": "aspirating", "type": "vacuum"})
                    except Exception as e:
                        print(f"[PLOTTER] Error parsing vacuuming step on line '{line.strip()}': {e}")

        return img_freq_s, events

    # --- MAIN PLOTTING LOGIC ---
    plt.figure(figsize=(8, 4.5))
    plt.style.use('dark_background')
    has_data = False

    # Trace log file dynamically
    exp_name = analysis_dir.parent.name
    log_path = analysis_dir.parent / f"{exp_name}_log.csv"

    # Get perfusion reference schedule using the first available well
    try:
        first_well_num = int(sorted(result_files)[0].stem.split("_")[-1])
        img_freq_s, reference_perfusions = parse_well_timings_inline(log_path, first_well_num)
        print(f"[PLOTTER] Extracted timing schema: Freq={img_freq_s}s, Events={reference_perfusions}")
    except Exception as e:
        print(f"[PLOTTER] Reference timing parser failed: {e}")
        img_freq_s, reference_perfusions = 1.0, []

    for file_path in sorted(result_files, key=lambda x: int(x.stem.split("_")[-1])):
        try:
            well_num = file_path.stem.split("_")[-1]
            print(f"[PLOTTER] Processing data file: {file_path.name}")

            with open(file_path, 'r') as f:
                well_data = json.load(f)

            trajectories = well_data.get("trajectories", {})

            # Robust label selector (supports legacy and new schemas)
            fluo_labels = well_data.get("fluorescent_cells_valid", [])
            if not fluo_labels:
                fluo_labels = well_data.get("fluorescent_cells", [])
            if not fluo_labels and trajectories:
                fluo_labels = list(trajectories.keys())

            fluo_intensities = [
                trajectories[label] for label in fluo_labels
                if label in trajectories
            ]

            if fluo_intensities:
                trajectories_matrix = np.array(fluo_intensities)
                well_mean_profile = np.mean(trajectories_matrix, axis=0)

                well_img_freq, _ = parse_well_timings_inline(log_path, int(well_num))
                time_axis = [t * well_img_freq for t in range(len(well_mean_profile))]

                plt.plot(time_axis, well_mean_profile,
                         label=f"Well {well_num}", marker='o', markersize=3, linewidth=1.5)
                has_data = True
                print(
                    f"[PLOTTER] Successfully plotted trajectory for Well {well_num} (frames={len(well_mean_profile)}).")
            else:
                print(f"[PLOTTER] Well {well_num} has no fluorescent trajectories.")

        except Exception as e:
            print(f"[PLOTTER] Error processing well file {file_path.name}: {e}")
            continue

    if not has_data:
        print("[PLOTTER] Aborting plot save: No valid well traces plotted.")
        plt.close()
        return False

    # --- APPLY UNIFIED SCALE ---
    y_min, y_max = get_global_y_limits(analysis_dir)
    plt.ylim(y_min, y_max)

    # --- DRAW TIMELINES WITH DYNAMICALLY MAPPED COLORS ---
    color_cycle = [
        "#3b82f6",  # Vibrant Blue
        "#ec4899",  # Magenta/Pink
        "#10b981",  # Emerald Green
        "#f59e0b",  # Amber/Orange
        "#a855f7",  # Purple
        "#06b6d4",  # Cyan
        "#f43f5e"   # Rose/Red
    ]
    solution_to_color_map = {}
    color_idx = 0

    # Built-in shortcut: X is data (seconds), Y is axes fraction (0.0 to 1.0)
    trans = plt.gca().get_xaxis_transform()

    for event in reference_perfusions:
        t_event = event["time"]
        raw_name = event["solution"].strip()
        norm_name = raw_name.lower()

        # Handle vacuum/aspiration vs standard perfusion coloring
        if event.get("type") == "vacuum":
            line_color = "#d4d4d8"  # Light grey for aspirating steps
        else:
            if norm_name not in solution_to_color_map:
                solution_to_color_map[norm_name] = color_cycle[color_idx % len(color_cycle)]
                color_idx += 1
            line_color = solution_to_color_map[norm_name]

        # Draw line
        plt.axvline(x=t_event, color=line_color, linestyle="--", alpha=0.6, linewidth=1.2)

        # Position text ending exactly at the top (Y=0.98 in axes fraction coordinate)
        plt.text(t_event + (img_freq_s * 0.15), 0.98, raw_name,
                 rotation=90, transform=trans,
                 horizontalalignment='left', verticalalignment='top',
                 color=line_color, fontsize=7, fontweight='bold', alpha=0.9)

    plt.title("Kinetic Profiles (All Processed Wells)", fontsize=11, fontweight='bold', pad=12, color='#e4e4e7')
    plt.xlabel("Time [seconds]", fontsize=9, color='#a1a1aa')
    plt.ylabel(r"Fluorescence Intensity ($\Delta$F/F$_0$)", fontsize=9, color='#a1a1aa')
    plt.grid(True, linestyle='--', alpha=0.15, color='#27272a')
    plt.ylim(y_min, y_max)
    plt.legend(loc='upper right', frameon=True, facecolor='#18181b', edgecolor='#27272a', fontsize=8)
    plt.tight_layout()

    output_path = analysis_dir / "all_wells_means.png"
    plt.savefig(output_path, dpi=200, facecolor='#18181b')
    plt.close()
    print(f"[PLOTTER] Saved cumulative plot safely to {output_path}")
    return True

def _read_csv_rows(f):
    plate_type = 6  # Default fallback if not found in file
    reader = csv.reader(f)

    for row in reader:
        if not row:
            continue

        # Check if any column in the row contains "plate-type:"
        for i, cell in enumerate(row):
            if "plate-type:" in cell.lower():
                # Based on the image, the number is likely in the next column
                if i + 1 < len(row):
                    try:
                        # Extract and clean the number from the next cell
                        plate_type = int(row[i + 1].strip())
                        return plate_type  # Return immediately once found
                    except ValueError:
                        # If conversion fails, keep checking or use default
                        pass

    return plate_type

def generate_well_list(plate_type):
    """Maps structural configurations depending on multi-well geometry."""
    if plate_type == 24:
        # 24-well layout: Rows A-D, Columns 1-6
        rows = ['A', 'B', 'C', 'D']
        cols = range(1, 7)
    elif plate_type == 12:
        # 12-well layout: Rows A-C, Columns 1-4
        rows = ['A', 'B', 'C']
        cols = range(1, 5)
    else:
        # Default 6-well layout: Rows A-B, Columns 1-3
        rows = ['A', 'B']
        cols = range(1, 4)

    well_list = []
    well_idx = 1
    for r in rows:
        for c in cols:
            well_list.append({
                'label': f"{r}{c}",
                'index': well_idx
            })
            well_idx += 1
    return well_list

def parse_log_file(exp_name):
    """Reads exp_name_log.csv to extract plate configuration details."""
    log_path = Path(BASE_PATH) / exp_name / f"{exp_name}_log.csv"
    plate_type = 6  # fallback default

    if not log_path.exists():
        print(f"Warning: Log file not found at {log_path}. Defaulting to 6-well plate.")
        return plate_type

    # --- UPDATED ENCODING HANDLER ---
    try:
        with open(log_path, mode='r', encoding='utf-8') as f:
            return _read_csv_rows(f)
    except UnicodeDecodeError:
        # Fall back to latin-1 to safely parse characters like µ without crashing
        with open(log_path, mode='r', encoding='latin-1') as f:
            return _read_csv_rows(f)

def send_command_and_wait(ser, command, expected_response="OK:", timeout=60.0):
    """
    Sends a serial command in a thread-safe manner and waits for
    a specific completion keyword before releasing the lock.
    """
    with serial_lock:

        if ser.write_timeout is None:
            ser.write_timeout = 2.0

        try:
            # Format command to bytes with trailing newline
            if isinstance(command, str):
                packet = f"{command}\n".encode('utf-8')
            elif isinstance(command, bytes):
                packet = command if command.endswith(b'\n') else command + b'\n'
            else:
                raise TypeError("Command must be a string or bytes object")

            # Flush input before sending to remove stale serial logs
            ser.reset_input_buffer()

            ser.write(packet)

            ser.flush()
            time.sleep(0.5)  # Small buffer for microcontroller to process

            # Poll input buffer until the target completion message is found or timeout occurs
            start_time = time.time()
            full_response = []

            while (time.time() - start_time) < timeout:
                if ser.in_waiting > 0:
                    line = ser.readline().decode('utf-8', errors='replace').strip()
                    if line:
                        print(f"send_command_and_wait: {line}")
                        full_response.append(line)

                        # Check for completion condition output by Arduino
                        if expected_response in line or "Dispensing Complete" in line:
                            return "\n".join(full_response)

                time.sleep(0.02)  # Yield CPU during long execution cycles

            print(f"Serial Error: Timed out waiting for response to {command}")
            return "Error: Timeout waiting for Arduino completion"

        except serial.SerialTimeoutException:
            print(f"Serial Error: Write operation timed out on command {command}!")
            return "Error: Connection dropped and reconnected"

        except Exception as e:
            print(f"Serial Error: {e}")
            return f"Error: {e}"