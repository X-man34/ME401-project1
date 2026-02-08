"""
Simple Rankine Cycle Analysis Tool
===================================
A human-readable Python script for thermodynamic analysis of basic Rankine cycles.

Purpose:
- Demonstrate object-oriented programming for thermodynamic cycle analysis
- Provide scaffolding for exploring steam power cycles
- Build industry-relevant Python skills for mechanical engineering applications

Author: [Your Name]
Date: February 2026
"""

import numpy as np
import matplotlib.pyplot as plt
from CoolProp.CoolProp import PropsSI


class RankineCycle:
    """
    Represents a basic ideal Rankine cycle for steam power generation.
    
    This class demonstrates ABSTRACTION - we hide the complexity of thermodynamic
    calculations inside methods, so users can simply create a cycle and analyze it
    without worrying about the underlying CoolProp calls.
    
    The basic Rankine cycle has 4 states:
    State 1: Saturated liquid leaving condenser (pump inlet)
    State 2: Compressed liquid leaving pump (boiler inlet)
    State 3: Superheated vapor leaving boiler (turbine inlet)
    State 4: Wet vapor leaving turbine (condenser inlet)
    
    Note: More complex cycles (reheat, regenerative, supercritical) can be explored
    by extending this class or creating similar classes for other cycle configurations.
    """
    
    def __init__(self, P_boiler, T_boiler, P_condenser, eta_pump=1.0, eta_turbine=1.0, fluid='Water'):
        """
        Initialize a Rankine cycle with operating conditions.
        
        Parameters:
        -----------
        P_boiler : float
            Boiler pressure [Pa] (high pressure side)
        T_boiler : float
            Boiler outlet temperature [K] (turbine inlet temperature)
        P_condenser : float
            Condenser pressure [Pa] (low pressure side)
        eta_pump : float
            Pump isentropic efficiency [-] (default 1.0 for ideal cycle)
        eta_turbine : float
            Turbine isentropic efficiency [-] (default 1.0 for ideal cycle)
        fluid : str
            Working fluid name (default 'Water')
            Note: To explore other working fluids (R134a, ammonia, CO2, etc.),
            simply change this parameter. Ensure the fluid is available in CoolProp.
            Different fluids may require different pressure/temperature ranges.
        """
        self.P_boiler = P_boiler
        self.T_boiler = T_boiler
        self.P_condenser = P_condenser
        self.eta_pump = eta_pump
        self.eta_turbine = eta_turbine
        self.fluid = fluid
        
        # Initialize state properties (will be calculated)
        self.states = {}
        
        # Calculate all state points
        self._calculate_states()
        
    def _calculate_states(self):
        """
        Calculate thermodynamic properties at all four states.
        
        This is a PRIVATE method (indicated by the underscore _) because users
        don't need to call it directly - it's called automatically during initialization.
        This demonstrates ENCAPSULATION - hiding internal implementation details.
        """
        
        # State 1: Saturated liquid at condenser pressure
        # We specify pressure and quality (x=0 means saturated liquid)
        self.states[1] = {
            'P': self.P_condenser,
            'T': PropsSI('T', 'P', self.P_condenser, 'Q', 0, self.fluid),
            'h': PropsSI('H', 'P', self.P_condenser, 'Q', 0, self.fluid),
            's': PropsSI('S', 'P', self.P_condenser, 'Q', 0, self.fluid),
            'v': PropsSI('V', 'P', self.P_condenser, 'Q', 0, self.fluid)
        }
        
        # State 2: Compressed liquid leaving pump (ideal: isentropic compression)
        # For ideal pump: s2 = s1
        s2_ideal = self.states[1]['s']
        h2_ideal = PropsSI('H', 'P', self.P_boiler, 'S', s2_ideal, self.fluid)
        
        # Actual pump work accounting for efficiency
        h2 = self.states[1]['h'] + (h2_ideal - self.states[1]['h']) / self.eta_pump
        
        self.states[2] = {
            'P': self.P_boiler,
            'T': PropsSI('T', 'P', self.P_boiler, 'H', h2, self.fluid),
            'h': h2,
            's': PropsSI('S', 'P', self.P_boiler, 'H', h2, self.fluid),
            'v': PropsSI('V', 'P', self.P_boiler, 'H', h2, self.fluid)
        }
        
        # State 3: Superheated vapor leaving boiler
        # We specify pressure and temperature
        self.states[3] = {
            'P': self.P_boiler,
            'T': self.T_boiler,
            'h': PropsSI('H', 'P', self.P_boiler, 'T', self.T_boiler, self.fluid),
            's': PropsSI('S', 'P', self.P_boiler, 'T', self.T_boiler, self.fluid),
            'v': PropsSI('V', 'P', self.P_boiler, 'T', self.T_boiler, self.fluid)
        }
        
        # State 4: Wet vapor leaving turbine (ideal: isentropic expansion)
        # For ideal turbine: s4 = s3
        s4_ideal = self.states[3]['s']
        h4_ideal = PropsSI('H', 'P', self.P_condenser, 'S', s4_ideal, self.fluid)
        
        # Actual turbine work accounting for efficiency
        h4 = self.states[3]['h'] - self.eta_turbine * (self.states[3]['h'] - h4_ideal)
        
        self.states[4] = {
            'P': self.P_condenser,
            'T': PropsSI('T', 'P', self.P_condenser, 'H', h4, self.fluid),
            'h': h4,
            's': PropsSI('S', 'P', self.P_condenser, 'H', h4, self.fluid),
            'v': PropsSI('V', 'P', self.P_condenser, 'H', h4, self.fluid)
        }
    
    def calculate_performance(self):
        """
        Calculate cycle performance metrics.
        
        Returns:
        --------
        dict : Performance metrics including work, heat transfer, and efficiency
        
        Units: All specific quantities in J/kg, efficiency is dimensionless
        """
        # Specific work and heat for each component (per kg of working fluid)
        w_pump = self.states[2]['h'] - self.states[1]['h']  # Work INTO pump
        q_in = self.states[3]['h'] - self.states[2]['h']    # Heat INTO boiler
        w_turbine = self.states[3]['h'] - self.states[4]['h']  # Work OUT of turbine
        q_out = self.states[4]['h'] - self.states[1]['h']   # Heat OUT of condenser
        
        # Net work and thermal efficiency
        w_net = w_turbine - w_pump
        eta_thermal = w_net / q_in
        
        # Back work ratio (fraction of turbine work used by pump)
        bwr = w_pump / w_turbine
        
        return {
            'w_pump': w_pump,
            'q_in': q_in,
            'w_turbine': w_turbine,
            'q_out': q_out,
            'w_net': w_net,
            'eta_thermal': eta_thermal,
            'bwr': bwr
        }
    
    def plot_ts_diagram(self, show_saturation_dome=True):
        """
        Plot Temperature-Entropy (T-S) diagram for the cycle.
        
        Parameters:
        -----------
        show_saturation_dome : bool
            If True, plots the saturation dome for reference (default: True)
        
        Note: All units in SI (temperature in K, entropy in J/kg-K).
        TODO: Add function to convert axes to imperial units (°F, BTU/lb-°R) if needed.
        """
        fig, ax = plt.subplots(figsize=(10, 7))
        
        # Plot saturation dome if requested
        if show_saturation_dome:
            self._plot_saturation_dome_ts(ax)
        
        # Extract state properties for plotting
        s_vals = [self.states[i]['s'] for i in [1, 2, 3, 4, 1]]  # Close the cycle
        T_vals = [self.states[i]['T'] for i in [1, 2, 3, 4, 1]]
        
        # Plot the cycle
        ax.plot(s_vals, T_vals, 'ro-', linewidth=2, markersize=8, label='Rankine Cycle')
        
        # Annotate state points
        for i in [1, 2, 3, 4]:
            ax.annotate(f'  State {i}', 
                       xy=(self.states[i]['s'], self.states[i]['T']),
                       fontsize=10, fontweight='bold')
        
        ax.set_xlabel('Specific Entropy, s [J/kg-K]', fontsize=12)
        ax.set_ylabel('Temperature, T [K]', fontsize=12)
        ax.set_title('Temperature-Entropy Diagram', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        
        plt.tight_layout()
        return fig
    
    def plot_pv_diagram(self, show_saturation_dome=True):
        """
        Plot Pressure-Volume (P-V) diagram for the cycle.
        
        Parameters:
        -----------
        show_saturation_dome : bool
            If True, plots the saturation dome for reference (default: True)
        
        Note: All units in SI (pressure in Pa, specific volume in m³/kg).
        TODO: Add function to convert axes to imperial units (psi, ft³/lb) if needed.
        """
        fig, ax = plt.subplots(figsize=(10, 7))
        
        # Plot saturation dome if requested
        if show_saturation_dome:
            self._plot_saturation_dome_pv(ax)
        
        # Extract state properties for plotting
        v_vals = [self.states[i]['v'] for i in [1, 2, 3, 4, 1]]  # Close the cycle
        P_vals = [self.states[i]['P'] for i in [1, 2, 3, 4, 1]]
        
        # Plot the cycle
        ax.plot(v_vals, P_vals, 'bo-', linewidth=2, markersize=8, label='Rankine Cycle')
        
        # Annotate state points
        for i in [1, 2, 3, 4]:
            ax.annotate(f'  State {i}', 
                       xy=(self.states[i]['v'], self.states[i]['P']),
                       fontsize=10, fontweight='bold')
        
        ax.set_xlabel('Specific Volume, v [m³/kg]', fontsize=12)
        ax.set_ylabel('Pressure, P [Pa]', fontsize=12)
        ax.set_title('Pressure-Volume Diagram', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        ax.set_yscale('log')  # Log scale helps visualize wide pressure range
        
        plt.tight_layout()
        return fig
    
    def _plot_saturation_dome_ts(self, ax):
        """
        Helper method to plot saturation dome on T-S diagram.
        
        The saturation dome shows the boundary between liquid and vapor phases,
        providing important context for where our cycle operates.
        """
        # Get critical point properties
        T_crit = PropsSI('Tcrit', self.fluid)
        P_crit = PropsSI('Pcrit', self.fluid)
        
        # Temperature range for saturation dome (from triple point to critical point)
        T_min = PropsSI('Ttriple', self.fluid)
        T_range = np.linspace(T_min, T_crit, 100)
        
        # Calculate saturated liquid and vapor properties
        s_liquid = []
        s_vapor = []
        
        for T in T_range:
            try:
                s_liquid.append(PropsSI('S', 'T', T, 'Q', 0, self.fluid))
                s_vapor.append(PropsSI('S', 'T', T, 'Q', 1, self.fluid))
            except:
                # Skip points that cause errors (near critical point)
                continue
        
        # Plot saturation dome
        ax.plot(s_liquid, T_range[:len(s_liquid)], 'k--', linewidth=1, alpha=0.5, label='Saturation Dome')
        ax.plot(s_vapor, T_range[:len(s_vapor)], 'k--', linewidth=1, alpha=0.5)
    
    def _plot_saturation_dome_pv(self, ax):
        """
        Helper method to plot saturation dome on P-V diagram.
        """
        # Get critical point properties
        T_crit = PropsSI('Tcrit', self.fluid)
        P_crit = PropsSI('Pcrit', self.fluid)
        
        # Temperature range for saturation dome
        T_min = PropsSI('Ttriple', self.fluid)
        T_range = np.linspace(T_min, T_crit, 100)
        
        # Calculate saturated liquid and vapor properties
        v_liquid = []
        v_vapor = []
        P_sat = []
        
        for T in T_range:
            try:
                P = PropsSI('P', 'T', T, 'Q', 0, self.fluid)
                v_liquid.append(PropsSI('V', 'T', T, 'Q', 0, self.fluid))
                v_vapor.append(PropsSI('V', 'T', T, 'Q', 1, self.fluid))
                P_sat.append(P)
            except:
                continue
        
        # Plot saturation dome
        ax.plot(v_liquid, P_sat, 'k--', linewidth=1, alpha=0.5, label='Saturation Dome')
        ax.plot(v_vapor, P_sat, 'k--', linewidth=1, alpha=0.5)
    
    def print_state_table(self):
        """
        Print a formatted table of thermodynamic properties at each state.
        
        This helps visualize and verify the cycle calculations.
        Units: P [Pa], T [K], h [J/kg], s [J/kg-K], v [m³/kg]
        """
        print("\n" + "="*80)
        print(f"Rankine Cycle State Properties ({self.fluid})")
        print("="*80)
        print(f"{'State':<10} {'P [MPa]':<15} {'T [K]':<15} {'h [kJ/kg]':<15} {'s [J/kg-K]':<15} {'v [m³/kg]':<15}")
        print("-"*80)
        
        for i in [1, 2, 3, 4]:
            print(f"{i:<10} {self.states[i]['P']/1e6:<15.3f} {self.states[i]['T']:<15.2f} "
                  f"{self.states[i]['h']/1e3:<15.2f} {self.states[i]['s']:<15.2f} "
                  f"{self.states[i]['v']:<15.6f}")
        
        print("="*80 + "\n")
    
    def print_performance(self):
        """
        Print cycle performance metrics in a readable format.
        """
        perf = self.calculate_performance()
        
        print("\n" + "="*80)
        print("Rankine Cycle Performance")
        print("="*80)
        print(f"Pump Work:          {perf['w_pump']/1e3:>10.2f} kJ/kg")
        print(f"Turbine Work:       {perf['w_turbine']/1e3:>10.2f} kJ/kg")
        print(f"Net Work:           {perf['w_net']/1e3:>10.2f} kJ/kg")
        print(f"Heat Input:         {perf['q_in']/1e3:>10.2f} kJ/kg")
        print(f"Heat Rejected:      {perf['q_out']/1e3:>10.2f} kJ/kg")
        print(f"Thermal Efficiency: {perf['eta_thermal']*100:>10.2f} %")
        print(f"Back Work Ratio:    {perf['bwr']*100:>10.2f} %")
        print("="*80 + "\n")


# =============================================================================
# EXAMPLE USAGE (This can be moved to a separate run script)
# =============================================================================

if __name__ == "__main__":
    """
    Example of how to use the RankineCycle class.
    
    This section demonstrates the VALUE of abstraction:
    - We don't need to know HOW CoolProp calculates properties
    - We don't need to manually call PropsSI for each state
    - We just specify operating conditions and let the class handle the rest!
    
    This same approach can be used in a larger optimization or parametric study.
    """
    
    # Define cycle operating conditions
    P_boiler = 8e6        # 8 MPa (typical steam power plant high pressure)
    T_boiler = 773.15     # 500°C = 773.15 K (superheat temperature)
    P_condenser = 10e3    # 10 kPa (typical condenser vacuum pressure)
    
    # Create a Rankine cycle object
    # See how clean and simple this is? That's the power of abstraction!
    cycle = RankineCycle(P_boiler, T_boiler, P_condenser)
    
    # Print state properties
    cycle.print_state_table()
    
    # Print performance metrics
    cycle.print_performance()
    
    # Generate T-S diagram
    fig_ts = cycle.plot_ts_diagram(show_saturation_dome=True)
    plt.savefig('rankine_ts_diagram.png', dpi=300, bbox_inches='tight')
    print("T-S diagram saved as 'rankine_ts_diagram.png'")
    
    # Generate P-V diagram
    fig_pv = cycle.plot_pv_diagram(show_saturation_dome=True)
    plt.savefig('rankine_pv_diagram.png', dpi=300, bbox_inches='tight')
    print("P-V diagram saved as 'rankine_pv_diagram.png'")
    
    # Show plots
    plt.show()


# =============================================================================
# RECOMMENDATIONS FOR FUTURE EXPANSION
# =============================================================================
"""
As you develop your skills, consider these expansions (in order of complexity):

1. PARAMETRIC STUDIES: Loop over different pressures/temperatures, plot efficiency vs. pressure
2. NON-IDEAL CYCLES: Implement actual pump/turbine efficiencies (eta < 1.0)
3. ALTERNATIVE FLUIDS: Try R134a, ammonia, or CO2 - compare working fluids
4. COMPLEX CYCLES: Add reheat (split turbine), regeneration (feedwater heaters)
5. EXERGY ANALYSIS: Calculate irreversibilities in each component
6. OPTIMIZATION: Use scipy.optimize to find optimal operating conditions
7. UNIT CONVERSION: Add methods to display results in imperial units
8. COST ANALYSIS: Integrate economic models for component sizing/costing
9. GUI INTERFACE: Create simple tkinter or streamlit interface for exploration
10. VALIDATION: Compare results against published steam tables or textbook examples
"""