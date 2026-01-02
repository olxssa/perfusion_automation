# perfusion_automation
*Design and implementation of an automated high-content screening cell perfusion system, including associated control software*

Public project repository about an automated multi-well perfusion system.

[multi_well_perfusion.py](https://github.com/olxssa/perfusion_automation/blob/main/multi_well_perfusion.py) enables the workflow script-based only and includes:
* Setting up of micro-manager instance, communication with the electronic control unit (ECU) and serial communication
* Software autofocus
* Generation of well-center coordinates based on plate layout

## Building a fluorescence microscope
The fluorescence microscope was self-built using the Rapid Automated Modular Microscope System (RAMM) by ASI as a frame. Furthermore, the motorized ASI xyz-stage and stage controller were added.
![microscope_setup](/images/microscope_setup.png)
1: Sola Light Engine (Lumencore), 6: 20x/0.3 Ph1 Achromat, 10: Basler ace2, 11: Cairn Aura Phase Contrast Illuminator.

## Constructing an automated perfusion system
### Fluid dispensing: Wiring of tubes and electrically-controllable microvalves
![fluid_wiring](/images/fluid_wiring.png)

### Z-movement: Self-built linear translation stage
* Servo motor (controlled by Arduino Uno)
* Miniaturized guide and slide block: MGN9R and MGN9H by Dold Mechatronik
* 3D printed mount and gear wheel
* Microswitch as a proximity detection sensor

### Circuit for pump and servo control
![circuit](/images/circuit.png)
1: Solenoid valve, 2: Relay, 3: Power supply, 4: Servo motor.

## Developing a web app as a user interface using Flask
Status quo:
![gui](/images/gui.png)
