# perfusion_automation
*Design and implementation of an automated high-content screening cell perfusion system, including associated control software*

Public project repository about an automated multi-well perfusion and imaging system. Except for the microscope frame, motorized stage and the main light source for fluorescence microscopy, the optical and electrical parts were either re-used or cheap to buy.

## Repository content

### 3D printed parts

### Software
<img src="/images/img_design.png" width="500">

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

## Documentation
### Building a fluorescence microscope
* Rapid Automated Modular Microscope System (RAMM) by ASI as a frame with motorized ASI xyz-stage and stage controller (~30 K)
* NeoPixel ring 24 LEDs
* Lumencore SOLA Light Engine (~8 K)
<img src="/images/light_path.png" width="500">

### Constructing an Automated Screening Perfusion System
#### Fluidic circuit
<img src="/images/fluidic_circuit.png" width="500">
Wiring of the fluidic circuit. A: Pressure distribution system. B: Reservoirs with perfusion solutions. C:
Micro-valve circuit and channel numbering. D: Perfusion site with precise volume exchange.

#### Electrical circuits
<img src="/images/electrical_circuit.png" width="500">
Wiring of the electrical circuit. A: RX System with Arduino Uno Rev 3 with Servos, Solenoid-Valves and Neopixel Ring. B: TX System with Xiao ESP32-S3 with Sensors and Abort button.

### Developing a custom web app as a user interface using Flask
#### System Architecture
<img src="/images/web_app.png" width="500">

#### Hardware Interfacing

#### Communication Protocols

#### Core Functional Methods
<img src="/images/perfusion_experiment.png" width="500">

### Status quo: 
[![](https://img.youtube.com/vi/6NS1tuH9CTA/0.jpg)](https://youtu.be/6NS1tuH9CTA "Click to play on Youtube.com")

Click on image to play on Youtube.com
