# perfusion_automation

<img align="right" src="Figures/bath_perfusion_icon_bathtub.png" width="100"/>

*Design and implementation of an automated high-content screening cell perfusion system, including associated control software*

Public project repository about an automated multi-well bath perfusion and imaging system.
***

## Repository Content

### FreeCAD Files

### FPC Design Files

### Firmware

### Software
<img src="/Figures/web_app_structure.png" width="920">

After the build the setup can be either used via a custom-developed web application or script-based with a Jupyter Notebook. Furthermore, the scripts core_methods.py and cellpose_worker.py are neccessary for both to work. 

The software includes the following tools:
* Preview:
   * Preview window featuring modes live, BF snap and fluorescence snap
   * Choose ideal fluorescence exposure time for your experiment
   * Choose indeal BF light source brightness for the software autofocus to work with your samples
   * Manual stage movement
   * Choose ideal software autofocus parameters for your experiment
   * Test flow from reservoirs 1-5
* Perfusion Experiment:
   * Set folder location to save all experiment files
   * Multi-well plate-type tool
   * Set imaging parameters: observation time after perfusion and imaging frequency
   * Name reservoirs accordingly to perfusate solutions
   * Set solution sequence and volumes to be perfused
   * Additional tools:
      * Automatically determine best Field of View (FOV) for every well
      * Manually determine best Field of View (FOV) for every well
* Preliminary Analysis:
   * Cell segmentation of first fluorescence image for every well
   * Dynamic classification of cells that are fluorescent
   * Background subtraction and computation of fluorescence change over time for individual cells for every well
   * Comparison of mean of fluorescence change over time for all wells

### Data


***

## Documentation of the Project
### 1. Building a Modular Fluorescence Microscope

<img src="/Figures/light_path_w.png" width="500">

### 2. Constructing an Automated Bath Perfusion System for Screening
#### Fluidic Circuit
<img src="/Figures/fluidic_circuit_wh.png" width="500">

<img src="/Figures/setup.png" width="500">

#### Electrical Circuits
<img src="/Figures/electrical_circuit_w.png" width="500">
Wiring of the electrical circuit. A: RX System with Arduino Uno Rev 3 with Servos, Solenoid-Valves and Neopixel Ring. B: TX System with Xiao ESP32-S3 with Sensors and Abort button.

### 3. Developing a Custom Web App as a User Interface using Flask
#### System Architecture
<img src="/Figures/web_app_w.png" width="500">

#### Hardware Interfacing

#### Communication Protocols

### Calibration and Benchmarking

<img src="/Figures/Benchmarking_FlowRate_w.png" width="500">

<img src="/Figures/benchmarking_SAF.png" width="500">

### References

*** 

### Status quo: 
[![](https://img.youtube.com/vi/6NS1tuH9CTA/0.jpg)](https://youtu.be/6NS1tuH9CTA "Click to play on Youtube.com")

Click on image to play on Youtube.com