# Rankine Cycle Optimization

This package contains scripts for thermodynamic optimization of Rankine power cycles using scipy.

## Files

1. **rankine_cycle.py** - Your original RankineCycle class (module)
2. **rankine_optimization.py** - Optimization script (run this one!)
3. **README.md** - This file

## Requirements

```bash
pip install numpy matplotlib scipy CoolProp --break-system-packages
```

## Usage

Simply run the optimization script:

```bash
python rankine_optimization.py
```

This will:
- Optimize cycle parameters to maximize thermal efficiency
- Compare SLSQP (gradient-based) vs Differential Evolution (global) methods
- Generate 3 visualization files:
  - `optimization_contour_boiler.png` - Efficiency vs boiler P & T
  - `optimization_contour_pressure.png` - Efficiency vs pressure ratio
  - `optimization_sensitivity.png` - 1D sensitivity plots

## What You'll Learn

### Optimization Concepts
- **Objective functions**: Converting maximization → minimization
- **Bounds**: Simple box constraints on variables
- **Nonlinear constraints**: Superheat and quality requirements
- **Algorithm selection**: When to use gradient-based vs global methods

### scipy.optimize Methods
- **SLSQP**: Fast gradient-based, good for smooth problems
- **Differential Evolution**: Slower but finds global optimum

### Visualization
- Contour plots to understand optimization landscape
- Sensitivity analysis to identify critical parameters
- Comparing multiple optimization results

## Next Steps

Try these modifications to build your skills:

1. Change the bounds in `optimize_rankine_cycle()` - explore supercritical pressures
2. Add turbine/pump inefficiencies (eta < 1.0)
3. Modify constraints - try different superheat or quality requirements
4. Add a third constraint (e.g., maximum heat exchanger area)
5. Create a cost-based objective function instead of pure efficiency

## Understanding the Output

The script prints:
- Initial guess efficiency
- Optimization progress (iterations, function evaluations)
- Optimal parameters for each method
- Comparison table of methods
- Full cycle properties and performance at optimum

Example output:
```
Optimal Parameters (SLSQP):
  Boiler Pressure:    20.000 MPa
  Boiler Temperature: 600.00 °C
  Condenser Pressure: 5.00 kPa
  Maximum Efficiency: 46.543%
```

Happy optimizing! 🚀