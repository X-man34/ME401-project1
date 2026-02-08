"""
Rankine Cycle Optimization Script
==================================
Demonstrates scipy.optimize for thermodynamic cycle parameter optimization.

Purpose:
- Learn how to formulate engineering optimization problems
- Understand objective functions, constraints, and bounds
- Compare different optimization algorithms
- Visualize optimization landscapes

Author: [Your Name]
Date: February 2026
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, differential_evolution
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.filterwarnings('ignore')  # Suppress CoolProp warnings during optimization

# Import your existing RankineCycle class
# Make sure the rankine_cycle.py file is in the same directory!
from rankine_cycle import RankineCycle


# =============================================================================
# STEP 1: DEFINE THE OBJECTIVE FUNCTION
# =============================================================================

def objective_function(x):
    """
    Objective function for optimization.
    
    In optimization, we want to MINIMIZE the objective function.
    Since we want to MAXIMIZE efficiency, we return the NEGATIVE efficiency.
    
    Parameters:
    -----------
    x : array-like, shape (3,)
        Decision variables: [P_boiler, T_boiler, P_condenser]
        Units: [Pa, K, Pa]
    
    Returns:
    --------
    float : Negative thermal efficiency (to convert maximization to minimization)
    
    Note: This function will be called MANY times during optimization,
    so we want it to be as fast as possible. Avoid unnecessary calculations!
    """
    P_boiler, T_boiler, P_condenser = x
    
    try:
        # Create cycle with current parameters
        cycle = RankineCycle(P_boiler, T_boiler, P_condenser, 
                            eta_pump=1.0, eta_turbine=1.0)
        
        # Calculate performance
        perf = cycle.calculate_performance()
        
        # Return NEGATIVE efficiency (we minimize, so negative = maximize)
        return -perf['eta_thermal']
    
    except Exception as e:
        # If CoolProp fails (invalid state), return a large penalty
        # This guides the optimizer away from infeasible regions
        return 1e6


# =============================================================================
# STEP 2: DEFINE CONSTRAINTS
# =============================================================================

def constraint_superheat(x):
    """
    Constraint: Ensure adequate superheat at turbine inlet.
    
    Superheat prevents moisture damage to turbine blades. We want the turbine
    inlet to be sufficiently above the saturation temperature at boiler pressure.
    
    For scipy.optimize constraints:
    - constraint >= 0 is SATISFIED
    - constraint < 0 is VIOLATED
    
    Parameters:
    -----------
    x : array-like
        [P_boiler, T_boiler, P_condenser]
    
    Returns:
    --------
    float : Superheat margin [K] (must be >= minimum_superheat)
    
    TODO: Research typical superheat values for steam turbines (usually 50-200 K)
    and adjust the minimum_superheat value based on industry standards.
    """
    from CoolProp.CoolProp import PropsSI
    
    P_boiler, T_boiler, P_condenser = x
    minimum_superheat = 50  # K - minimum superheat requirement
    
    try:
        # Saturation temperature at boiler pressure
        T_sat_boiler = PropsSI('T', 'P', P_boiler, 'Q', 1, 'Water')
        
        # Actual superheat
        superheat = T_boiler - T_sat_boiler
        
        # Return margin: positive = satisfied, negative = violated
        return superheat - minimum_superheat
    
    except:
        return -1e6  # Large negative = violated


def constraint_quality_turbine_exit(x):
    """
    Constraint: Ensure turbine exit quality is not too low.
    
    If steam is too wet at turbine exit, it can cause erosion damage.
    Typical minimum quality is 0.85-0.90 (85-90% vapor by mass).
    
    Parameters:
    -----------
    x : array-like
        [P_boiler, T_boiler, P_condenser]
    
    Returns:
    --------
    float : Quality margin (must be >= 0 for quality >= minimum)
    
    TODO: Investigate how different condenser pressures affect turbine exit
    quality and whether this constraint is active at the optimum.
    """
    P_boiler, T_boiler, P_condenser = x
    minimum_quality = 0.88  # Typical industry standard
    
    try:
        cycle = RankineCycle(P_boiler, T_boiler, P_condenser)
        
        # Calculate quality at state 4 (turbine exit)
        from CoolProp.CoolProp import PropsSI
        quality_4 = PropsSI('Q', 'P', P_condenser, 'H', cycle.states[4]['h'], 'Water')
        
        # Return margin
        return quality_4 - minimum_quality
    
    except:
        return -1e6


# =============================================================================
# STEP 3: SET UP OPTIMIZATION PROBLEM
# =============================================================================

def optimize_rankine_cycle():
    """
    Main optimization function demonstrating scipy.optimize best practices.
    
    This function shows how to:
    1. Define bounds (simple box constraints)
    2. Define nonlinear constraints
    3. Choose initial guess
    4. Select optimization algorithm
    5. Interpret results
    """
    
    print("\n" + "="*80)
    print("RANKINE CYCLE OPTIMIZATION")
    print("="*80)
    
    # -------------------------------------------------------------------------
    # Define bounds for decision variables
    # -------------------------------------------------------------------------
    # Bounds are the simplest type of constraint: lower <= x <= upper
    # Format: [(lower, upper), (lower, upper), (lower, upper)]
    
    # TODO: As you learn more about power plant design, explore how these bounds
    # affect the optimum. What happens with supercritical pressures (>22 MPa)?
    # What if you allow lower condenser pressures (better vacuum)?
    
    bounds = [
        (1e6, 20e6),      # P_boiler: 1-20 MPa
        (400+273.15, 600+273.15),  # T_boiler: 400-600°C converted to K
        (5e3, 50e3)       # P_condenser: 5-50 kPa
    ]
    
    print("\nOptimization Bounds:")
    print(f"  Boiler Pressure:    {bounds[0][0]/1e6:.1f} - {bounds[0][1]/1e6:.1f} MPa")
    print(f"  Boiler Temperature: {bounds[1][0]-273.15:.1f} - {bounds[1][1]-273.15:.1f} °C")
    print(f"  Condenser Pressure: {bounds[2][0]/1e3:.1f} - {bounds[2][1]/1e3:.1f} kPa")
    
    # -------------------------------------------------------------------------
    # Define constraints
    # -------------------------------------------------------------------------
    # Constraints are formatted as dictionaries for scipy.optimize.minimize
    # 'type': 'ineq' means constraint >= 0
    
    constraints = [
        {'type': 'ineq', 'fun': constraint_superheat},
        {'type': 'ineq', 'fun': constraint_quality_turbine_exit}
    ]
    
    # -------------------------------------------------------------------------
    # Initial guess
    # -------------------------------------------------------------------------
    # A good initial guess helps the optimizer converge faster.
    # Start somewhere in the middle of the feasible region.
    
    x0 = np.array([
        10e6,      # P_boiler = 10 MPa (middle of range)
        500+273.15,  # T_boiler = 500°C (middle of range)
        10e3       # P_condenser = 10 kPa (middle of range)
    ])
    
    print(f"\nInitial Guess:")
    print(f"  P_boiler = {x0[0]/1e6:.1f} MPa")
    print(f"  T_boiler = {x0[1]-273.15:.1f} °C")
    print(f"  P_condenser = {x0[2]/1e3:.1f} kPa")
    
    # Evaluate initial efficiency
    initial_efficiency = -objective_function(x0)
    print(f"  Initial Efficiency: {initial_efficiency*100:.2f}%")
    
    # -------------------------------------------------------------------------
    # METHOD 1: Sequential Least Squares Programming (SLSQP)
    # -------------------------------------------------------------------------
    # SLSQP is a gradient-based method that handles constraints well.
    # Pros: Fast, handles constraints, usually finds local optimum
    # Cons: Can get stuck in local optima, sensitive to initial guess
    
    print("\n" + "-"*80)
    print("METHOD 1: SLSQP (Gradient-Based Optimization)")
    print("-"*80)
    
    result_slsqp = minimize(
        objective_function,
        x0,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'disp': True, 'maxiter': 100}
    )
    
    if result_slsqp.success:
        print("\nOptimization successful!")
        print_optimization_results(result_slsqp, "SLSQP")
    else:
        print("\nOptimization failed:", result_slsqp.message)
    
    # -------------------------------------------------------------------------
    # METHOD 2: Differential Evolution (Global Optimization)
    # -------------------------------------------------------------------------
    # Differential Evolution is a genetic algorithm that explores globally.
    # Pros: Finds global optimum, robust, doesn't need initial guess
    # Cons: Slower, requires more function evaluations
    
    print("\n" + "-"*80)
    print("METHOD 2: Differential Evolution (Global Optimization)")
    print("-"*80)
    print("This may take a minute - evaluating many candidate solutions...")
    
    result_de = differential_evolution(
        objective_function,
        bounds,
        constraints=constraints,
        seed=42,  # For reproducibility
        disp=True,
        maxiter=50,  # Adjust based on time available
        workers=1    # Use 1 worker to avoid CoolProp threading issues
    )
    
    if result_de.success:
        print("\nOptimization successful!")
        print_optimization_results(result_de, "Differential Evolution")
    else:
        print("\nOptimization failed:", result_de.message)
    
    # -------------------------------------------------------------------------
    # Compare methods
    # -------------------------------------------------------------------------
    print("\n" + "="*80)
    print("COMPARISON OF OPTIMIZATION METHODS")
    print("="*80)
    print(f"{'Method':<25} {'Efficiency':<15} {'Function Evals':<20}")
    print("-"*80)
    if result_slsqp.success:
        print(f"{'SLSQP':<25} {-result_slsqp.fun*100:<15.3f} {result_slsqp.nfev:<20}")
    if result_de.success:
        print(f"{'Differential Evolution':<25} {-result_de.fun*100:<15.3f} {result_de.nfev:<20}")
    print("="*80)
    
    return result_slsqp, result_de


def print_optimization_results(result, method_name):
    """
    Print detailed optimization results in a readable format.
    
    Parameters:
    -----------
    result : OptimizeResult
        Result object from scipy.optimize
    method_name : str
        Name of optimization method for display
    """
    P_opt, T_opt, P_cond_opt = result.x
    eta_opt = -result.fun  # Remember we minimized negative efficiency
    
    print(f"\nOptimal Parameters ({method_name}):")
    print(f"  Boiler Pressure:    {P_opt/1e6:.3f} MPa")
    print(f"  Boiler Temperature: {T_opt-273.15:.2f} °C")
    print(f"  Condenser Pressure: {P_cond_opt/1e3:.2f} kPa")
    print(f"  Maximum Efficiency: {eta_opt*100:.3f}%")
    print(f"  Function Evaluations: {result.nfev}")
    
    # Create and display optimal cycle
    print(f"\n{method_name} - Optimal Cycle Properties:")
    optimal_cycle = RankineCycle(P_opt, T_opt, P_cond_opt)
    optimal_cycle.print_state_table()
    optimal_cycle.print_performance()


# =============================================================================
# STEP 4: VISUALIZATION OF OPTIMIZATION LANDSCAPE
# =============================================================================

def visualize_optimization_landscape(result_slsqp, result_de):
    """
    Create visualizations showing how efficiency varies with parameters.
    
    This helps you understand:
    1. Is the optimization landscape smooth or rough?
    2. Are there multiple local optima?
    3. How sensitive is efficiency to each parameter?
    4. Which parameters have the strongest effect?
    
    Parameters:
    -----------
    result_slsqp : OptimizeResult
        Result from SLSQP optimization
    result_de : OptimizeResult
        Result from differential evolution
    """
    
    # Use the DE result as the "best" optimum (more likely to be global)
    P_opt, T_opt, P_cond_opt = result_de.x if result_de.success else result_slsqp.x
    
    # -------------------------------------------------------------------------
    # Plot 1: Efficiency vs Boiler Pressure and Temperature
    # -------------------------------------------------------------------------
    print("\nGenerating contour plot: Efficiency vs P_boiler and T_boiler...")
    
    # Create grid of parameters (fix condenser pressure at optimal)
    P_boiler_range = np.linspace(5e6, 20e6, 30)
    T_boiler_range = np.linspace(400+273.15, 600+273.15, 30)
    P_boiler_grid, T_boiler_grid = np.meshgrid(P_boiler_range, T_boiler_range)
    
    # Calculate efficiency at each grid point
    efficiency_grid = np.zeros_like(P_boiler_grid)
    
    for i in range(P_boiler_grid.shape[0]):
        for j in range(P_boiler_grid.shape[1]):
            x_test = [P_boiler_grid[i,j], T_boiler_grid[i,j], P_cond_opt]
            obj_val = objective_function(x_test)
            efficiency_grid[i,j] = -obj_val if obj_val < 1e5 else np.nan
    
    # Create contour plot
    fig1, ax1 = plt.subplots(figsize=(10, 8))
    
    contour = ax1.contourf(P_boiler_grid/1e6, T_boiler_grid-273.15, 
                           efficiency_grid*100, levels=20, cmap='viridis')
    ax1.contour(P_boiler_grid/1e6, T_boiler_grid-273.15, 
                efficiency_grid*100, levels=10, colors='white', 
                linewidths=0.5, alpha=0.5)
    
    # Mark optimal points
    if result_slsqp.success:
        ax1.plot(result_slsqp.x[0]/1e6, result_slsqp.x[1]-273.15, 
                'r*', markersize=20, label='SLSQP Optimum', markeredgecolor='white')
    if result_de.success:
        ax1.plot(result_de.x[0]/1e6, result_de.x[1]-273.15, 
                'y*', markersize=20, label='DE Optimum', markeredgecolor='white')
    
    ax1.set_xlabel('Boiler Pressure [MPa]', fontsize=12)
    ax1.set_ylabel('Boiler Temperature [°C]', fontsize=12)
    ax1.set_title(f'Thermal Efficiency vs Boiler Conditions\n(Condenser P = {P_cond_opt/1e3:.1f} kPa)', 
                  fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    cbar1 = plt.colorbar(contour, ax=ax1)
    cbar1.set_label('Thermal Efficiency [%]', fontsize=11)
    
    plt.tight_layout()
    plt.savefig('optimization_contour_boiler.png', dpi=300, bbox_inches='tight')
    print("  Saved: optimization_contour_boiler.png")
    
    # -------------------------------------------------------------------------
    # Plot 2: Efficiency vs Condenser Pressure and Boiler Pressure
    # -------------------------------------------------------------------------
    print("\nGenerating contour plot: Efficiency vs P_condenser and P_boiler...")
    
    P_boiler_range2 = np.linspace(5e6, 20e6, 30)
    P_cond_range = np.linspace(5e3, 50e3, 30)
    P_boiler_grid2, P_cond_grid = np.meshgrid(P_boiler_range2, P_cond_range)
    
    efficiency_grid2 = np.zeros_like(P_boiler_grid2)
    
    for i in range(P_boiler_grid2.shape[0]):
        for j in range(P_boiler_grid2.shape[1]):
            x_test = [P_boiler_grid2[i,j], T_opt, P_cond_grid[i,j]]
            obj_val = objective_function(x_test)
            efficiency_grid2[i,j] = -obj_val if obj_val < 1e5 else np.nan
    
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    
    contour2 = ax2.contourf(P_boiler_grid2/1e6, P_cond_grid/1e3, 
                            efficiency_grid2*100, levels=20, cmap='plasma')
    ax2.contour(P_boiler_grid2/1e6, P_cond_grid/1e3, 
                efficiency_grid2*100, levels=10, colors='white', 
                linewidths=0.5, alpha=0.5)
    
    if result_slsqp.success:
        ax2.plot(result_slsqp.x[0]/1e6, result_slsqp.x[2]/1e3, 
                'r*', markersize=20, label='SLSQP Optimum', markeredgecolor='white')
    if result_de.success:
        ax2.plot(result_de.x[0]/1e6, result_de.x[2]/1e3, 
                'y*', markersize=20, label='DE Optimum', markeredgecolor='white')
    
    ax2.set_xlabel('Boiler Pressure [MPa]', fontsize=12)
    ax2.set_ylabel('Condenser Pressure [kPa]', fontsize=12)
    ax2.set_title(f'Thermal Efficiency vs Pressure Ratio\n(Boiler T = {T_opt-273.15:.0f}°C)', 
                  fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    cbar2 = plt.colorbar(contour2, ax=ax2)
    cbar2.set_label('Thermal Efficiency [%]', fontsize=11)
    
    plt.tight_layout()
    plt.savefig('optimization_contour_pressure.png', dpi=300, bbox_inches='tight')
    print("  Saved: optimization_contour_pressure.png")
    
    # -------------------------------------------------------------------------
    # Plot 3: 1D sensitivity plots
    # -------------------------------------------------------------------------
    print("\nGenerating sensitivity plots...")
    
    fig3, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Vary boiler pressure
    P_range = np.linspace(5e6, 20e6, 50)
    eta_P = []
    for P in P_range:
        x = [P, T_opt, P_cond_opt]
        eta_P.append(-objective_function(x))
    
    axes[0].plot(P_range/1e6, np.array(eta_P)*100, 'b-', linewidth=2)
    axes[0].axvline(P_opt/1e6, color='r', linestyle='--', label='Optimum')
    axes[0].set_xlabel('Boiler Pressure [MPa]', fontsize=11)
    axes[0].set_ylabel('Thermal Efficiency [%]', fontsize=11)
    axes[0].set_title('Sensitivity to Boiler Pressure', fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # Vary boiler temperature
    T_range = np.linspace(400+273.15, 600+273.15, 50)
    eta_T = []
    for T in T_range:
        x = [P_opt, T, P_cond_opt]
        eta_T.append(-objective_function(x))
    
    axes[1].plot(T_range-273.15, np.array(eta_T)*100, 'g-', linewidth=2)
    axes[1].axvline(T_opt-273.15, color='r', linestyle='--', label='Optimum')
    axes[1].set_xlabel('Boiler Temperature [°C]', fontsize=11)
    axes[1].set_ylabel('Thermal Efficiency [%]', fontsize=11)
    axes[1].set_title('Sensitivity to Boiler Temperature', fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    # Vary condenser pressure
    P_cond_range = np.linspace(5e3, 50e3, 50)
    eta_P_cond = []
    for P_cond in P_cond_range:
        x = [P_opt, T_opt, P_cond]
        eta_P_cond.append(-objective_function(x))
    
    axes[2].plot(P_cond_range/1e3, np.array(eta_P_cond)*100, 'm-', linewidth=2)
    axes[2].axvline(P_cond_opt/1e3, color='r', linestyle='--', label='Optimum')
    axes[2].set_xlabel('Condenser Pressure [kPa]', fontsize=11)
    axes[2].set_ylabel('Thermal Efficiency [%]', fontsize=11)
    axes[2].set_title('Sensitivity to Condenser Pressure', fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig('optimization_sensitivity.png', dpi=300, bbox_inches='tight')
    print("  Saved: optimization_sensitivity.png")
    
    print("\nAll visualization files saved successfully!")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    """
    Run the complete optimization study.
    
    This demonstrates a complete workflow:
    1. Define and solve optimization problem
    2. Compare different algorithms
    3. Visualize results
    4. Interpret findings
    """
    
    # Run optimization
    result_slsqp, result_de = optimize_rankine_cycle()
    
    # Visualize optimization landscape
    visualize_optimization_landscape(result_slsqp, result_de)
    
    # Display plots
    plt.show()
    
    print("\n" + "="*80)
    print("OPTIMIZATION COMPLETE!")
    print("="*80)
    print("\nKey Takeaways:")
    print("1. Gradient-based (SLSQP) is fast but can miss global optimum")
    print("2. Global methods (DE) are slower but more robust")
    print("3. Contour plots reveal optimization landscape structure")
    print("4. Sensitivity plots show which parameters matter most")
    print("\nNext Steps to Explore:")
    print("- Try non-ideal efficiencies (eta_pump < 1, eta_turbine < 1)")
    print("- Add economic objective ($/kWh instead of pure efficiency)")
    print("- Explore supercritical pressures (>22 MPa)")
    print("- Implement multi-objective optimization (efficiency vs cost)")
    print("- Add more realistic constraints (material temperature limits)")
    print("="*80 + "\n")


# =============================================================================
# ADVANCED EXERCISES FOR FURTHER LEARNING
# =============================================================================
"""
Once you're comfortable with this script, try these challenges:

1. MULTI-OBJECTIVE OPTIMIZATION:
   - Maximize efficiency AND minimize cost
   - Use scipy.optimize with weighted objective or Pareto front

2. UNCERTAINTY QUANTIFICATION:
   - Add uncertainty to operating conditions (±5% pressure, ±10 K temp)
   - Use Monte Carlo to evaluate robust optimum

3. COMPONENT SIZING:
   - Add heat exchanger area as design variable
   - Constrain based on heat transfer correlations

4. DYNAMIC OPTIMIZATION:
   - Optimize startup/shutdown trajectories
   - Use time-varying constraints

5. ALTERNATIVE WORKING FLUIDS:
   - Compare optimized water, R134a, and CO2 cycles
   - Which fluid gives best performance for given T_source?

6. EXERGY OPTIMIZATION:
   - Minimize total exergy destruction instead of maximizing efficiency
   - Identify bottleneck components
"""