"""
Rankine Cycle Analysis - Command Line Interface
================================================
Run script for thermodynamic analysis and optimization of steam power cycles.

Usage Examples:
---------------
# Single cycle analysis
python run_rankine_analysis.py --mode single --p-boiler 8e6 --t-boiler 773.15 --p-condenser 10e3

# Optimized cycle
python run_rankine_analysis.py --mode optimize

# Parametric study (sweep boiler pressure)
python run_rankine_analysis.py --mode parametric --sweep p-boiler --p-boiler-range 5e6 20e6

# Compare multiple configurations
python run_rankine_analysis.py --mode compare --configs "8e6,773.15,10e3" "15e6,823.15,5e3"

Author: [Your Name]
Date: February 2026
"""

import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt
from rankine_cycle import RankineCycle
from rankine_optimization import optimize_rankine_cycle, visualize_optimization_landscape


# =============================================================================
# MODE 1: SINGLE CYCLE ANALYSIS
# =============================================================================

def run_single_cycle(args):
    """
    Analyze a single Rankine cycle with specified parameters.
    
    Parameters from args:
    ---------------------
    p_boiler : float [Pa]
    t_boiler : float [K]
    p_condenser : float [Pa]
    eta_pump : float [-] (default 1.0)
    eta_turbine : float [-] (default 1.0)
    show_plots : bool (default True)
    """
    print("\n" + "="*80)
    print("SINGLE CYCLE ANALYSIS")
    print("="*80)
    
    # Create cycle
    cycle = RankineCycle(
        P_boiler=args.p_boiler,
        T_boiler=args.t_boiler,
        P_condenser=args.p_condenser,
        eta_pump=args.eta_pump,
        eta_turbine=args.eta_turbine
    )
    
    # Display results
    cycle.print_state_table()
    cycle.print_performance()
    
    # Generate diagrams
    if args.show_plots:
        fig_ts = cycle.plot_ts_diagram(show_saturation_dome=True)
        fig_pv = cycle.plot_pv_diagram(show_saturation_dome=True)
        plt.show()
    
    print("="*80 + "\n")


# =============================================================================
# MODE 2: OPTIMIZATION
# =============================================================================

def run_optimization(args):
    """
    Run cycle optimization to maximize thermal efficiency.
    
    Uses both SLSQP (gradient-based) and Differential Evolution (global)
    methods as implemented in rankine_optimization.py.
    
    Parameters from args:
    ---------------------
    show_plots : bool (default True)
    """
    print("\n" + "="*80)
    print("OPTIMIZATION MODE")
    print("="*80)
    
    # Run optimization (from rankine_optimization.py)
    result_slsqp, result_de = optimize_rankine_cycle()
    
    # Visualize optimization landscape
    if args.show_plots:
        visualize_optimization_landscape(result_slsqp, result_de)
        plt.show()
    
    print("="*80 + "\n")


# =============================================================================
# MODE 3: PARAMETRIC STUDY
# =============================================================================

def run_parametric_study(args):
    """
    Sweep one parameter and plot efficiency vs. that parameter.
    
    Parameters from args:
    ---------------------
    sweep : str
        Parameter to sweep: 'p-boiler', 't-boiler', or 'p-condenser'
    p_boiler : float [Pa] (baseline if not sweeping)
    t_boiler : float [K] (baseline if not sweeping)
    p_condenser : float [Pa] (baseline if not sweeping)
    p_boiler_range : tuple (min, max) [Pa] (if sweeping p-boiler)
    t_boiler_range : tuple (min, max) [K] (if sweeping t-boiler)
    p_condenser_range : tuple (min, max) [Pa] (if sweeping p-condenser)
    resolution : int (number of points, default 50)
    eta_pump : float [-] (default 1.0)
    eta_turbine : float [-] (default 1.0)
    show_plots : bool (default True)
    """
    print("\n" + "="*80)
    print("PARAMETRIC STUDY")
    print("="*80)
    
    # Determine which parameter to sweep
    sweep_param = args.sweep
    resolution = args.resolution
    
    # Set baseline values
    P_boiler_base = args.p_boiler
    T_boiler_base = args.t_boiler
    P_condenser_base = args.p_condenser
    
    # Generate sweep range
    if sweep_param == 'p-boiler':
        if args.p_boiler_range is None:
            print("Error: Must specify --p-boiler-range when sweeping p-boiler")
            sys.exit(1)
        
        sweep_values = np.linspace(args.p_boiler_range[0], args.p_boiler_range[1], resolution)
        param_name = "Boiler Pressure"
        param_units = "MPa"
        param_scale = 1e6
        
        print(f"\nSweeping: {param_name}")
        print(f"Range: {args.p_boiler_range[0]/1e6:.1f} - {args.p_boiler_range[1]/1e6:.1f} {param_units}")
        print(f"Fixed: T_boiler = {T_boiler_base-273.15:.1f}°C, P_condenser = {P_condenser_base/1e3:.1f} kPa")
        
    elif sweep_param == 't-boiler':
        if args.t_boiler_range is None:
            print("Error: Must specify --t-boiler-range when sweeping t-boiler")
            sys.exit(1)
        
        sweep_values = np.linspace(args.t_boiler_range[0], args.t_boiler_range[1], resolution)
        param_name = "Boiler Temperature"
        param_units = "°C"
        param_scale = 1
        param_offset = 273.15
        
        print(f"\nSweeping: {param_name}")
        print(f"Range: {args.t_boiler_range[0]-273.15:.1f} - {args.t_boiler_range[1]-273.15:.1f} {param_units}")
        print(f"Fixed: P_boiler = {P_boiler_base/1e6:.1f} MPa, P_condenser = {P_condenser_base/1e3:.1f} kPa")
        
    elif sweep_param == 'p-condenser':
        if args.p_condenser_range is None:
            print("Error: Must specify --p-condenser-range when sweeping p-condenser")
            sys.exit(1)
        
        sweep_values = np.linspace(args.p_condenser_range[0], args.p_condenser_range[1], resolution)
        param_name = "Condenser Pressure"
        param_units = "kPa"
        param_scale = 1e3
        
        print(f"\nSweeping: {param_name}")
        print(f"Range: {args.p_condenser_range[0]/1e3:.1f} - {args.p_condenser_range[1]/1e3:.1f} {param_units}")
        print(f"Fixed: P_boiler = {P_boiler_base/1e6:.1f} MPa, T_boiler = {T_boiler_base-273.15:.1f}°C")
    
    else:
        print(f"Error: Invalid sweep parameter '{sweep_param}'")
        print("Valid options: 'p-boiler', 't-boiler', 'p-condenser'")
        sys.exit(1)
    
    print(f"Resolution: {resolution} points\n")
    
    # Perform sweep
    efficiencies = []
    net_work = []
    heat_input = []
    
    print("Calculating cycle performance across parameter range...")
    for value in sweep_values:
        try:
            # Set parameters for this iteration
            if sweep_param == 'p-boiler':
                P_boiler = value
                T_boiler = T_boiler_base
                P_condenser = P_condenser_base
            elif sweep_param == 't-boiler':
                P_boiler = P_boiler_base
                T_boiler = value
                P_condenser = P_condenser_base
            elif sweep_param == 'p-condenser':
                P_boiler = P_boiler_base
                T_boiler = T_boiler_base
                P_condenser = value
            
            # Create cycle and calculate performance
            cycle = RankineCycle(P_boiler, T_boiler, P_condenser, 
                               eta_pump=args.eta_pump, eta_turbine=args.eta_turbine)
            perf = cycle.calculate_performance()
            
            efficiencies.append(perf['eta_thermal'])
            net_work.append(perf['w_net'])
            heat_input.append(perf['q_in'])
            
        except Exception as e:
            # If calculation fails, append NaN
            efficiencies.append(np.nan)
            net_work.append(np.nan)
            heat_input.append(np.nan)
    
    # Convert to arrays
    efficiencies = np.array(efficiencies)
    net_work = np.array(net_work)
    heat_input = np.array(heat_input)
    
    # Find optimal point
    valid_mask = ~np.isnan(efficiencies)
    if np.any(valid_mask):
        max_idx = np.nanargmax(efficiencies)
        max_efficiency = efficiencies[max_idx]
        optimal_value = sweep_values[max_idx]
        
        print(f"\nOptimal Point:")
        if sweep_param == 'p-boiler':
            print(f"  P_boiler = {optimal_value/param_scale:.2f} {param_units}")
        elif sweep_param == 't-boiler':
            print(f"  T_boiler = {optimal_value-param_offset:.2f} {param_units}")
        elif sweep_param == 'p-condenser':
            print(f"  P_condenser = {optimal_value/param_scale:.2f} {param_units}")
        
        print(f"  Max Efficiency = {max_efficiency*100:.2f}%")
    
    # Plot results
    if args.show_plots:
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        
        # Prepare x-axis values
        if sweep_param == 'p-boiler':
            x_plot = sweep_values / param_scale
            x_label = f"{param_name} [{param_units}]"
        elif sweep_param == 't-boiler':
            x_plot = sweep_values - param_offset
            x_label = f"{param_name} [{param_units}]"
        elif sweep_param == 'p-condenser':
            x_plot = sweep_values / param_scale
            x_label = f"{param_name} [{param_units}]"
        
        # Plot 1: Thermal Efficiency
        axes[0].plot(x_plot, efficiencies*100, 'b-', linewidth=2)
        if np.any(valid_mask):
            axes[0].plot(x_plot[max_idx], max_efficiency*100, 'r*', 
                        markersize=15, label=f'Max = {max_efficiency*100:.2f}%')
        axes[0].set_xlabel(x_label, fontsize=11)
        axes[0].set_ylabel('Thermal Efficiency [%]', fontsize=11)
        axes[0].set_title('Thermal Efficiency', fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        
        # Plot 2: Net Work Output
        axes[1].plot(x_plot, net_work/1e3, 'g-', linewidth=2)
        axes[1].set_xlabel(x_label, fontsize=11)
        axes[1].set_ylabel('Net Work [kJ/kg]', fontsize=11)
        axes[1].set_title('Net Work Output', fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        # Plot 3: Heat Input
        axes[2].plot(x_plot, heat_input/1e3, 'm-', linewidth=2)
        axes[2].set_xlabel(x_label, fontsize=11)
        axes[2].set_ylabel('Heat Input [kJ/kg]', fontsize=11)
        axes[2].set_title('Heat Input', fontweight='bold')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    print("="*80 + "\n")


# =============================================================================
# MODE 4: COMPARISON
# =============================================================================

def run_comparison(args):
    """
    Compare multiple cycle configurations side-by-side.
    
    Parameters from args:
    ---------------------
    configs : list of str
        Each string format: "P_boiler,T_boiler,P_condenser"
        Example: "8e6,773.15,10e3"
    eta_pump : float [-] (default 1.0)
    eta_turbine : float [-] (default 1.0)
    show_plots : bool (default True)
    """
    print("\n" + "="*80)
    print("COMPARISON MODE")
    print("="*80)
    
    if not args.configs or len(args.configs) < 2:
        print("Error: Must specify at least 2 configurations to compare")
        print("Usage: --configs \"8e6,773.15,10e3\" \"15e6,823.15,5e3\"")
        sys.exit(1)
    
    # Parse configurations
    cycles = []
    config_labels = []
    
    for i, config_str in enumerate(args.configs):
        try:
            parts = config_str.split(',')
            if len(parts) != 3:
                raise ValueError("Each config must have 3 values: P_boiler,T_boiler,P_condenser")
            
            P_boiler = float(parts[0])
            T_boiler = float(parts[1])
            P_condenser = float(parts[2])
            
            # Create cycle
            cycle = RankineCycle(P_boiler, T_boiler, P_condenser,
                               eta_pump=args.eta_pump, eta_turbine=args.eta_turbine)
            cycles.append(cycle)
            
            # Create label
            label = f"Config {i+1}: {P_boiler/1e6:.1f} MPa, {T_boiler-273.15:.0f}°C, {P_condenser/1e3:.1f} kPa"
            config_labels.append(label)
            
        except Exception as e:
            print(f"Error parsing configuration '{config_str}': {e}")
            sys.exit(1)
    
    print(f"\nComparing {len(cycles)} configurations:\n")
    
    # Print comparison table
    print("="*100)
    print(f"{'Config':<10} {'P_boiler [MPa]':<18} {'T_boiler [°C]':<18} {'P_cond [kPa]':<18} {'η_th [%]':<15} {'W_net [kJ/kg]':<15}")
    print("-"*100)
    
    performances = []
    for i, (cycle, label) in enumerate(zip(cycles, config_labels)):
        perf = cycle.calculate_performance()
        performances.append(perf)
        
        print(f"{i+1:<10} {cycle.P_boiler/1e6:<18.2f} {cycle.T_boiler-273.15:<18.1f} "
              f"{cycle.P_condenser/1e3:<18.2f} {perf['eta_thermal']*100:<15.2f} "
              f"{perf['w_net']/1e3:<15.2f}")
    
    print("="*100)
    
    # Identify best configuration
    efficiencies = [p['eta_thermal'] for p in performances]
    best_idx = np.argmax(efficiencies)
    print(f"\nBest Configuration: Config {best_idx+1} (η = {efficiencies[best_idx]*100:.2f}%)")
    
    # Print detailed state tables for each configuration
    print("\n" + "="*100)
    print("DETAILED STATE PROPERTIES")
    print("="*100)
    
    for i, (cycle, label) in enumerate(zip(cycles, config_labels)):
        print(f"\n{label}")
        print("-"*100)
        cycle.print_state_table()
    
    # Plot comparison
    if args.show_plots:
        # Create figure with subplots
        fig = plt.figure(figsize=(16, 10))
        
        # T-S Diagram
        ax1 = fig.add_subplot(2, 2, 1)
        colors = ['b', 'r', 'g', 'm', 'c', 'y']
        
        for i, (cycle, label) in enumerate(zip(cycles, config_labels)):
            s_vals = [cycle.states[j]['s'] for j in [1, 2, 3, 4, 1]]
            T_vals = [cycle.states[j]['T'] for j in [1, 2, 3, 4, 1]]
            ax1.plot(s_vals, T_vals, f'{colors[i % len(colors)]}o-', 
                    linewidth=2, markersize=6, label=f'Config {i+1}')
        
        ax1.set_xlabel('Specific Entropy, s [J/kg-K]', fontsize=11)
        ax1.set_ylabel('Temperature, T [K]', fontsize=11)
        ax1.set_title('T-S Diagram Comparison', fontweight='bold', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=9)
        
        # P-V Diagram
        ax2 = fig.add_subplot(2, 2, 2)
        
        for i, (cycle, label) in enumerate(zip(cycles, config_labels)):
            v_vals = [cycle.states[j]['v'] for j in [1, 2, 3, 4, 1]]
            P_vals = [cycle.states[j]['P'] for j in [1, 2, 3, 4, 1]]
            ax2.plot(v_vals, P_vals, f'{colors[i % len(colors)]}o-', 
                    linewidth=2, markersize=6, label=f'Config {i+1}')
        
        ax2.set_xlabel('Specific Volume, v [m³/kg]', fontsize=11)
        ax2.set_ylabel('Pressure, P [Pa]', fontsize=11)
        ax2.set_title('P-V Diagram Comparison', fontweight='bold', fontsize=12)
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=9)
        
        # Performance Bar Chart
        ax3 = fig.add_subplot(2, 2, 3)
        
        config_nums = np.arange(1, len(cycles)+1)
        efficiencies_pct = [p['eta_thermal']*100 for p in performances]
        
        bars = ax3.bar(config_nums, efficiencies_pct, 
                      color=[colors[i % len(colors)] for i in range(len(cycles))],
                      alpha=0.7, edgecolor='black', linewidth=1.5)
        
        # Add value labels on bars
        for i, (bar, eff) in enumerate(zip(bars, efficiencies_pct)):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{eff:.2f}%', ha='center', va='bottom', fontweight='bold')
        
        ax3.set_xlabel('Configuration', fontsize=11)
        ax3.set_ylabel('Thermal Efficiency [%]', fontsize=11)
        ax3.set_title('Thermal Efficiency Comparison', fontweight='bold', fontsize=12)
        ax3.set_xticks(config_nums)
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Net Work Bar Chart
        ax4 = fig.add_subplot(2, 2, 4)
        
        net_works = [p['w_net']/1e3 for p in performances]
        
        bars2 = ax4.bar(config_nums, net_works,
                       color=[colors[i % len(colors)] for i in range(len(cycles))],
                       alpha=0.7, edgecolor='black', linewidth=1.5)
        
        for i, (bar, work) in enumerate(zip(bars2, net_works)):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{work:.1f}', ha='center', va='bottom', fontweight='bold')
        
        ax4.set_xlabel('Configuration', fontsize=11)
        ax4.set_ylabel('Net Work [kJ/kg]', fontsize=11)
        ax4.set_title('Net Work Comparison', fontweight='bold', fontsize=12)
        ax4.set_xticks(config_nums)
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.show()
    
    print("="*80 + "\n")


# =============================================================================
# COMMAND LINE ARGUMENT PARSER
# =============================================================================

def parse_arguments():
    """
    Parse command line arguments for Rankine cycle analysis.
    """
    parser = argparse.ArgumentParser(
        description='Rankine Cycle Analysis Tool - Command Line Interface',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
---------
Single cycle analysis:
  python run_rankine_analysis.py --mode single --p-boiler 8e6 --t-boiler 773.15 --p-condenser 10e3

Optimization:
  python run_rankine_analysis.py --mode optimize

Parametric study (sweep boiler pressure):
  python run_rankine_analysis.py --mode parametric --sweep p-boiler --p-boiler-range 5e6 20e6 --t-boiler 773.15 --p-condenser 10e3

Parametric study (sweep boiler temperature):
  python run_rankine_analysis.py --mode parametric --sweep t-boiler --t-boiler-range 673.15 873.15 --p-boiler 10e6 --p-condenser 10e3

Parametric study (sweep condenser pressure):
  python run_rankine_analysis.py --mode parametric --sweep p-condenser --p-condenser-range 5e3 50e3 --p-boiler 10e6 --t-boiler 773.15

Compare configurations:
  python run_rankine_analysis.py --mode compare --configs "8e6,773.15,10e3" "15e6,823.15,5e3" "12e6,800,8e3"
        """
    )
    
    # Main mode selection
    parser.add_argument('--mode', type=str, required=True,
                       choices=['single', 'optimize', 'parametric', 'compare'],
                       help='Analysis mode: single, optimize, parametric, or compare')
    
    # Cycle parameters
    parser.add_argument('--p-boiler', type=float, default=8e6,
                       help='Boiler pressure [Pa] (default: 8e6)')
    parser.add_argument('--t-boiler', type=float, default=773.15,
                       help='Boiler temperature [K] (default: 773.15 K = 500°C)')
    parser.add_argument('--p-condenser', type=float, default=10e3,
                       help='Condenser pressure [Pa] (default: 10e3)')
    
    # Component efficiencies
    parser.add_argument('--eta-pump', type=float, default=1.0,
                       help='Pump isentropic efficiency (default: 1.0)')
    parser.add_argument('--eta-turbine', type=float, default=1.0,
                       help='Turbine isentropic efficiency (default: 1.0)')
    
    # Parametric study options
    parser.add_argument('--sweep', type=str, choices=['p-boiler', 't-boiler', 'p-condenser'],
                       help='Parameter to sweep in parametric mode')
    parser.add_argument('--p-boiler-range', type=float, nargs=2, metavar=('MIN', 'MAX'),
                       help='Boiler pressure range [Pa] for sweep (e.g., 5e6 20e6)')
    parser.add_argument('--t-boiler-range', type=float, nargs=2, metavar=('MIN', 'MAX'),
                       help='Boiler temperature range [K] for sweep (e.g., 673.15 873.15)')
    parser.add_argument('--p-condenser-range', type=float, nargs=2, metavar=('MIN', 'MAX'),
                       help='Condenser pressure range [Pa] for sweep (e.g., 5e3 50e3)')
    parser.add_argument('--resolution', type=int, default=50,
                       help='Number of points in parametric sweep (default: 50)')
    
    # Comparison mode options
    parser.add_argument('--configs', type=str, nargs='+',
                       help='Cycle configurations for comparison (format: "P_boiler,T_boiler,P_condenser")')
    
    # Display options
    parser.add_argument('--no-plots', action='store_true',
                       help='Disable plot display (default: plots shown)')
    
    args = parser.parse_args()
    
    # Set show_plots based on --no-plots flag
    args.show_plots = not args.no_plots
    
    return args


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """
    Main entry point for command line interface.
    """
    # Parse arguments
    args = parse_arguments()
    
    # Route to appropriate analysis mode
    if args.mode == 'single':
        run_single_cycle(args)
    
    elif args.mode == 'optimize':
        run_optimization(args)
    
    elif args.mode == 'parametric':
        run_parametric_study(args)
    
    elif args.mode == 'compare':
        run_comparison(args)
    
    else:
        print(f"Error: Unknown mode '{args.mode}'")
        sys.exit(1)


if __name__ == "__main__":
    main()


# =============================================================================
# FUTURE ENHANCEMENTS - INTERACTIVE TOOLS
# =============================================================================
"""
NEXT LEVEL: Interactive Application Development
================================================

This command-line tool provides a solid foundation. To create more interactive
and user-friendly applications, consider exploring these technologies:

WEB-BASED INTERFACES:
  • Streamlit - fastest way to create shareable web apps from Python scripts
  • Plotly Dash - interactive dashboards with sophisticated callbacks
  • Gradio - ML-focused interface builder, excellent for parameter tuning
  • Flask/FastAPI - full web frameworks for custom applications

DESKTOP GUI:
  • PyQt5/PySide6 - professional desktop applications (steeper learning curve)
  • tkinter - built-in Python GUI, simpler but less modern
  • Dear PyGui - high-performance, GPU-accelerated interfaces

NOTEBOOK INTEGRATION:
  • Jupyter widgets (ipywidgets) - interactive controls in notebooks
  • Voilà - convert Jupyter notebooks to standalone web apps
  • JupyterLab extensions - custom interactive visualizations

DATA VISUALIZATION:
  • Plotly - interactive plots with zoom, pan, hover tooltips
  • Bokeh - web-based interactive visualizations
  • Altair - declarative statistical visualization

REACTIVE FRAMEWORKS:
  • Panel - works with any Python plotting library, very flexible
  • Shiny for Python - R-style reactive programming in Python

API & INTEGRATION:
  • FastAPI - modern API framework for integration with other tools
  • REST API - expose cycle calculations as web service
  • WebSocket - real-time updates for optimization progress

RECOMMENDED STARTING POINT:
  → Streamlit: Minimal code to create shareable web apps
  → Add sliders, dropdowns, file uploads for cycle parameters
  → Deploy to Streamlit Cloud for free public sharing
  → Example: ~50 lines of code to recreate this entire CLI as a web app
"""