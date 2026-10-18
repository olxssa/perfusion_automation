# perfusion_automation

<img align="right" src="Figures/bath_perfusion_icon_bathtub.png" width="100"/>

*Design and implementation of an automated high-content screening cell perfusion system, including associated control software*

Here, the design, implementation and benchmarking of a fully integrated screening platform for automated bath perfusion experiments is documented. 
The system was developed to provide reproducible and sample-friendly solution exchange for kinetic biosensor characterization.

The platform consists of three primary systems: a modular fluorescence microscope, an automated bath perfusion add-on, and an integrated hardware and software control framework. 

## Repository Content

### FreeCAD Files
Using the software FreeCAD, the provided CAD files can be exported as .stl and 3D-printed using PLA, 15 % infill and 0.2 mm layer height. 
Alternatively, the files can be adapted to your liking.

### FPC Design Files
In order to create a more elegant design for controlling the microvalves, a custom Flexible Printed Circuit (FPC) was designed using KiCAD in accordance with the information given by the company memetis. 
The microvalves are controlled via a constant-current source. 
The valves are all connected in series and can each be bypassed on the circuit board using a MOSFET, with a tap between the valve contacts. 
This allows individual valves to be removed from the current path and taken out of the control circuit.

<img src="/Figures/perfusion_system_FPC.png" width="500">

Custom-designed flexible printed circuit (FPC) for ECU-I 8CH and eight microvalves positioned on two 4-valve adapters and one 2-valve adapter. 
(**A**) Top-view of perfusion arm where 8 individual cables were connected to the controller adapter. 
(**B**) The PCB layer of the custom-designed FPC in KiCAD. 
(**C**) Top-view of the perfusion arm with the FPC.

### Firmware and MMConfig
Low-level hardware management is split between two microcontrollers and using the firmware rx_system.ino and tx_system.ino. 
High-level microscope hardware control is achieved via Pycro-Manager [1], 
which bridges the Python backend to a headless Java-based Micro-Manager [2] (v2.0.3) instance over port 4721.
The Micro-Manager instance is created using the config file MMConfig.cfg.

### Software
<img src="/Figures/web_app_structure_.png" width="920">

After the build, the setup can be controlled via a custom-developed web application called "Automated Bath Perfusion App". 
The web application serves as a single-page dashboard designed to organize all operational controls into three dedicated panels arranged from left to right:
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
      * CenterFOV: By default, the well center coordinates are used as Field of View
      * AutoFOV: Automatically determine best Field of View for every well
      * ManuFOV: Manually determine best Field of View and focus position for every well
* Preliminary Analysis:
   * Cell segmentation of first fluorescence image for every well
   * Dynamic classification of cells that are fluorescent
   * Background subtraction and computation of fluorescence change over time for individual cells for every well
   * Comparison of mean of fluorescence change over time for all wells

### Data
The system's performance is validated by assessing pipetting procedures, 
demonstrating software autofocus stability and conducting automated multiplexed perfusion on cells expressing GFP and NEMO-based calcium sensors under agonist conditions.

## Documentation of the Project
### 1. Building a Modular Fluorescence Microscope
<img src="/Figures/light_path_w.png" width="500">

Schematic optical light path of the modular fluorescence microscope.

### 2. Constructing an Automated Bath Perfusion System for Screening
<img src="/Figures/Setup.png" width="500">

Implementation of the automated bath perfusion system. 
(**A**) The complete automated screening perfusion system integrated into the modular fluorescence microscope. 
(**B**)  The perfusion arm responsible for movement of tubes into single wells and for the perfusion and aspiration of the desired solution.

#### Fluidic Circuit
<img src="/Figures/fluidic_circuit_wh.png" width="500">

Wiring of the fluidic circuit. 
(**A**) Pressure distribution system. 
(**B**) Reservoirs with perfusion solutions. 
(**C**) Microvalve circuit and channel numbering. 
(**D**) Perfusion site with precise volume exchange.

#### Electrical Circuits
<img src="/Figures/electrical_circuit_w.png" width="500">

Wiring of the electrical circuits. 
(**A**) The RX system receives signals from the PC and controls the servo motors, the stepper motor drivers, relays (switchable power socket, 3/2-valve relays), and the Neopixel LED ring. 
(**B**) The TX system gathers information from the flow-rate sensor, the fluid level sensors and the stop button and sends the information to the PC.

### 3. Developing a Custom Web App as a User Interface using Flask
<img src="/Figures/web_app_wh.png" width="500">

Web App Architecture. Multi-layered system showing the interaction between green software layers, blue hardware nodes, and the physical microscope components in grey.

<img src="/Figures/perfusion_experiment_w.png" width="500">

Perfusion experiment workflow sequence. (**A**) Initialization and sensor validation. (**B**) Automated well
traversal, solution perfusion, and imaging loops. (**C**) Decontamination protocol and park execution.

### 4. Calibration and Benchmarking
<img src="/Figures/Benchmarking_FlowRate_w.png" width="500">

Flow-rate stability characterization. 
(**A**)  Flow-rate robustness and transient oscillations observed during operational tuning. 
(**B**)  Flow-rate decay resulting from increased system resistance upon integrating additional reservoirs. 
(**C**)  Hydrostatic flow-rate decay as reservoir fluid levels deplete (tested across 100 mL, 50 mL, and 10 mL capacities). 
Modularity allows for interchangeable reservoir sizes to accommodate varying experimental volume requirements.

<img src="/Figures/benchmarking_SAF.png" width="500">

Software autofocus performance throughout a 6-well plate. 
For each well, except for the starting position A1, a course scan followed by a fine scan are carried out as shown on the left, resulting in a focussed sample shown on the right of each well.

<img src="/Figures/GFP_w.png" width="500">

[//]: # (GFP Caption)

[//]: # (Add NEMO Data + NEMO Caption)

### References

[1] H. Pinkard, N. Stuurman, I. E. Ivanov, et al., “Pycro-manager: Open-source software for customized and
reproducible microscope control,” Nat Methods, vol. 18, 226–228, 2021. DOI: 10.1038/s41592-021-01087-6.

[2] A. D. Edelstein, M. A. Tsuchida, N. Amodaj, H. Pinkard, R. D. Vale, and N. Stuurman, “Advanced methods of
microscope control using μmanager software,” Journal of Biological Methods, 2014. DOI: 10.14440/jbm.2014.36.

### Demonstration Video: 
[![](https://img.youtube.com/vi/6NS1tuH9CTA/0.jpg)](https://youtu.be/6NS1tuH9CTA "Click to play on Youtube.com") Version: March 2026

Click on image to play on Youtube.com