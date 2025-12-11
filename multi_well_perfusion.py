import os

from pymmcore_plus import CMMCorePlus, Device, DeviceAdapter, ConfigGroup

def setup_mm():
    """
    Sets up micro-manager instance using a micro-manager configuration file.

    Returns:
        core
    """

    # Micro manager directory
    mm_dir = 'D:\ProgramFiles\Micro-Manager-2.0'
    core = CMMCorePlus()
    # core_plus.setDeviceAdapterSearchPaths([mm_dir])
    core.loadSystemConfiguration(os.path.join(mm_dir, 'MMConfig_Basler_SOLA_ASI-XYZ_PixelSize.cfg'))
    print("Micromanager configuration loaded successfully.")

    return core

import logging
from ecu import ECUManager
from memetis.pcon.ecui import ECUI

def setup_logging_ecu():
    """
    Sets up ecu logging.

    Returns:
        log, ecu
    """

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
import time

def setup_serial():
    """
    Sets up serial communication.

    Returns:
        ser
    """

    # Connect to serial communication with arduino (pump + z-movement)
    ser = serial.Serial('COM6', 9600, timeout=1)
    time.sleep(2)  # wait for Arduino to reset
    ser.write(b"ANGLE 90\n")
    print("Arduino connected successfully.")

    return ser

from useq import MDASequence
from pymmcore_plus.mda import mda_listeners_connected
from pymmcore_plus.mda.handlers import ImageSequenceWriter
import glob
import cv2
import numpy as np
from scipy.optimize import curve_fit

def gauss(x, a, x0, sigma):
    return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))

def software_autofocus(core, range, step_size):
    """
        Perform software autofocus using z-stack images
        based on a range and step size in µm.
    """
    z = core.getZPosition()

    # define location and type of image saving
    writer = ImageSequenceWriter(r'C:\Users\Admin\Desktop\Olyssa\focus', extension=".png", overwrite=True)

    pos = core.getXYPosition()
    # acquire z-stack
    sequence = MDASequence(
        axis_order="tpgcz",
        stage_positions=[(pos[0], pos[1], z)],
        channels=[{'group': 'LED_light', 'config': 'on'}],
        z_plan={'above': range, 'below': range, 'step': step_size}
    )

    # run focus mda sequence
    with mda_listeners_connected(writer):
        core.mda.run(sequence)

    # calculate focus scores
    focus_images = glob.glob(r'C:\Users\Admin\Desktop\Olyssa\focus\*.png')
    focus_scores = []
    for f in focus_images:
        im = cv2.imread(rf'{f}')
        im_filtered = cv2.medianBlur(im, ksize=3)
        laplacian = cv2.Laplacian(im_filtered, ddepth=cv2.CV_64F, ksize=3)
        focus_score = laplacian.var()
        focus_scores.append(focus_score)

    # define x and y for fitting
    y = focus_scores
    x = np.linspace(z - range, z + range, len(y))

    # define gauss fit function
    mean = sum(x * y) / sum(y)
    sigma = np.sqrt(sum(y * (x - mean) ** 2) / sum(y))

    # optimize gauss fit
    popt, pcov = curve_fit(gauss, x, y, p0=[np.max(y), mean, sigma])

    # calculate maximum focus score of fit function
    x_fine = np.linspace(z - range, z + range, 100)
    y_max_fit = np.max(gauss(x_fine, *popt))
    pos = np.where(y_max_fit == gauss(x_fine, *popt))
    x_max_fit = x_fine[pos]
    y_max_data = np.max(focus_scores)
    x_max_data = x[np.where(focus_scores == y_max_data)]

    # set optimal z position
    core.setPosition(float(x_max_fit[0]))

    # remove focus images
    for file in focus_images:
        os.remove(file)

import random
def generate_well_centers(n_wells, order_type="snake"):
    """
    Generate well center coordinates for 6- or 24-well plates using only
    n_wells and order_type.

    Layout map includes:
        - rows, cols
        - pitch
        - a1_center

    Supports ordering:
        "row-major", "column-major", "snake", "random"

    Returns:
        coords
    """

    layout_map = {
        6: {
            "rows": 2,
            "cols": 3,
            "pitch": 39200,
            "a1": (-49605.2, -18609.1),
        },
        24: {
            "rows": 4,
            "cols": 6,
            "pitch": 19300,
            "a1": (-59336.2, -29119.1),
        },
    }

    if n_wells not in layout_map:
        raise ValueError("Only 6- and 24-well plates are supported.")

    # Extract layout
    rows   = layout_map[n_wells]["rows"]
    cols   = layout_map[n_wells]["cols"]
    pitch  = layout_map[n_wells]["pitch"]
    x0, y0 = layout_map[n_wells]["a1"]

    well_coords = [
        (x0 + c * pitch, y0 + r * pitch)
        for r in range(rows)
        for c in range(cols)
    ]

    row_labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    well_names = [
        f"{row_labels[r]}{c+1}"
        for r in range(rows)
        for c in range(cols)
    ]

    if order_type == "row-major":
        order_idx = list(range(len(well_names)))

    elif order_type == "column-major":
        order_idx = [r*cols + c for c in range(cols) for r in range(rows)]

    elif order_type == "snake":
        order_idx = []
        for r in range(rows):
            if r % 2 == 0:     # left → right
                order_idx.extend([r*cols + c for c in range(cols)])
            else:              # right → left
                order_idx.extend([r*cols + c for c in reversed(range(cols))])

    elif order_type == "random":
        order_idx = list(range(len(well_names)))
        random.shuffle(order_idx)

    else:
        raise ValueError("order_type must be 'row-major', 'column-major', 'snake', or 'random'")

    coords = [well_coords[i] for i in order_idx]
    return coords