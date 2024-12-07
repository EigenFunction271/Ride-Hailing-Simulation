# Ride-Hailing Simulation

A Python-based simulation of a ride-hailing system that models driver-rider interactions, demand patterns, and system dynamics.

## Features

- Real-time visualization of drivers, riders, and system statistics
- Periodic demand patterns simulating rush hours and quiet periods
- Dynamic driver state management (cruising, assigned, with passenger, off duty)
- Intelligent driver behavior with clustering for efficient rider pickup
- Grid-based road network with optimized pathfinding
- Supply-demand balancing system

## Dependencies

```python
numpy
networkx
scipy
matplotlib
scikit-learn
```

## Key Parameters

- `MAP_SIZE`: Size of the grid-based road network (default: 30)
- `TOTAL_DRIVERS`: Number of drivers in the simulation (default: 200)
- `RIDER_GENERATION_RATE`: Base rate for new rider generation
- `N_RIDER_CLUSTERS`: Number of demand hotspots (default: 5)
- `N_DROPOFF_CLUSTERS`: Number of common dropoff locations (default: 5)
- `DEMAND_PERIOD`: Length of one complete demand cycle
- `BASE_RIDER_RATE`: Minimum rider generation rate
- `PEAK_RIDER_RATE`: Maximum rider generation rate

## Components

### Driver States
- **Cruising**: Searching for riders
- **Assigned**: En route to pickup
- **With Passenger**: Transporting rider to destination
- **Off Duty**: Taking breaks in designated rest areas

### Demand Generation
- Dynamic demand patterns following periodic cycles
- Clustered rider generation around hotspots
- Realistic dropoff location distribution

### Visualization
- Real-time map showing driver and rider positions
- Live statistics tracking:
  - Drivers with passengers
  - Waiting riders
  - Cruising drivers
  - Off-duty drivers

## Performance Optimizations

- Vectorized operations for distance calculations
- Efficient driver-rider matching using numpy
- K-means clustering for large-scale rider grouping
- State-based driver updates
- Optimized memory usage

## Usage

Run the simulation:

```python
python Main.py
```

The visualization window will show:
- Left: Map view with drivers and riders
- Right: Real-time statistics graphs

## Implementation Details

- Uses NetworkX for road network representation
- Matplotlib for real-time visualization
- NumPy for efficient numerical operations
- Scikit-learn for clustering algorithms
- SciPy for statistical distributions

## Notes

- The simulation is optimized for large numbers of drivers and riders
- Grid-based movement ensures realistic traffic patterns
- Supply-demand balancing adjusts driver availability automatically
```

