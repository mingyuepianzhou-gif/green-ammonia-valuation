import numpy as np
import pandas as pd

class GreenAmmoniaRealOptions:
    """
    Green Ammonia Investment Valuation Model using Least Squares Monte Carlo (LSM)
    """
    def __init__(self, p_amm_base=3700, day_amm_base=2800):
        # 1. Macro & Financial Parameters
        self.rf = 0.06                # Risk-free rate
        self.tax = 0.15               # Corporate tax rate
        self.n_steps = 30             # Investment decision window (years)
        self.n_repl = 5000            # Monte Carlo simulation paths
        self.dt = 1                   # Time step (year)
        self.lt = 20                  # Project life time
        
        # 2. Volatility & Drift (Stochastic Processes)
        self.vol_amm = 0.20           # Ammonia price annual volatility
        self.mu_amm = 0.0             # Ammonia price drift
        self.vol_c = 0.25             # Carbon price annual volatility
        self.mu_c = 0.03              # Carbon price drift
        self.mu_inv = -0.08           # CAPEX reduction rate (Technological progress)
        
        # 3. Base Scenarios
        self.p_amm_base = p_amm_base
        self.p_c_base = 91.8
        self.day_amm = day_amm_base
        
        # 4. Operational Parameters (Simplified for engine demonstration)
        self.days_per_year = 326
        self.q_amm_annual = self.days_per_year * self.day_amm
        self.fixed_cost_annual = 50000000  # Placeholder for complex fixed cost calculations
        
    def _simulate_price_paths(self):
        """Simulate Geometric Brownian Motion (GBM) for Ammonia and Carbon prices."""
        np.random.seed(42)
        total_steps = self.n_steps + self.lt
        
        p_amm_paths = np.zeros((self.n_repl, total_steps))
        p_c_paths = np.zeros((self.n_repl, total_steps))
        
        p_amm_paths[:, 0] = self.p_amm_base
        p_c_paths[:, 0] = self.p_c_base
        
        for t in range(1, total_steps):
            z1 = np.random.standard_normal(self.n_repl)
            z2 = np.random.standard_normal(self.n_repl)
            
            p_amm_paths[:, t] = p_amm_paths[:, t-1] * np.exp(
                (self.mu_amm - 0.5 * self.vol_amm**2) * self.dt + self.vol_amm * np.sqrt(self.dt) * z1
            )
            p_c_paths[:, t] = p_c_paths[:, t-1] * np.exp(
                (self.mu_c - 0.5 * self.vol_c**2) * self.dt + self.vol_c * np.sqrt(self.dt) * z2
            )
            
        return p_amm_paths, p_c_paths

    def execute_lsm_valuation(self, initial_investment):
        """
        Execute the Least Squares Monte Carlo (LSM) algorithm to find optimal investment timing.
        """
        p_amm_paths, p_c_paths = self._simulate_price_paths()
        
        # Calculate dynamic cash flows
        # CF = (Revenue - Fixed Cost) * (1 - Tax)
        revenue_paths = self.q_amm_annual * p_amm_paths
        cf_dynamic = (revenue_paths - self.fixed_cost_annual) * (1 - self.tax)
        
        # Calculate dynamic CAPEX (declining due to tech progress)
        capex_dynamic = initial_investment * (1 + self.mu_inv) ** np.arange(self.n_steps)
        
        # Step 1: Calculate Present Value of operating cash flows for each potential investment year
        pv_matrix = np.zeros((self.n_repl, self.n_steps))
        for j in range(self.n_steps):
            for t in range(self.lt):
                # Discount cash flows back to the decision year 'j'
                pv_matrix[:, j] += cf_dynamic[:, j+t] * np.exp(-self.rf * t)
                
        # Step 2: Calculate intrinsic value (Free Cash Flow of immediate exercise)
        intrinsic_value = pv_matrix - capex_dynamic
        option_value = np.maximum(intrinsic_value, 0)
        
        # Step 3: Backward Induction using Polynomial Regression (LSM Core)
        decision_matrix = np.zeros((self.n_repl, self.n_steps))
        
        # At the final step, we invest if intrinsic value > 0
        decision_matrix[option_value[:, -1] > 0, -1] = 1
        
        for j in range(self.n_steps - 2, -1, -1):
            # Only consider paths where it's currently in-the-money
            itm_idx = np.where(option_value[:, j] > 0)[0]
            
            if len(itm_idx) > 0:
                x = pv_matrix[itm_idx, j]
                # Discount the next period's option value back one step
                y = option_value[itm_idx, j+1] * np.exp(-self.rf * self.dt)
                
                # Cross-sectional regression (Degree 2 polynomial)
                coef = np.polyfit(x, y, 2)
                expected_continuation_value = np.polyval(coef, x)
                
                # Decision rule: Invest if immediate payoff > expected continuation value
                invest_idx = itm_idx[option_value[itm_idx, j] >= expected_continuation_value]
                wait_idx = itm_idx[option_value[itm_idx, j] < expected_continuation_value]
                
                decision_matrix[invest_idx, j] = 1
                decision_matrix[wait_idx, j] = 0
                
                # Update option value matrix for the backward induction
                option_value[invest_idx, j] = intrinsic_value[invest_idx, j]
                option_value[wait_idx, j] = option_value[wait_idx, j+1] * np.exp(-self.rf * self.dt)

        # Extract Optimal Investment Timing Distribution
        optimal_years = []
        for m in range(self.n_repl):
            invest_years = np.where(decision_matrix[m, :] == 1)[0]
            if len(invest_years) > 0:
                optimal_years.append(invest_years[0] + 2026) # Add base year
                
        return pd.Series(optimal_years).value_counts(normalize=True).sort_index()

# --- Example Usage ---
if __name__ == "__main__":
    print("Initializing Green Ammonia Valuation Engine...")
    model = GreenAmmoniaRealOptions(p_amm_base=3700, day_amm_base=2800)
    
    print("Running Monte Carlo and LSM regression (5000 paths)...")
    # Assuming an initial CAPEX of 1.5 Billion CNY for demonstration
    opt_timing_prob = model.execute_lsm_valuation(initial_investment=1.5e9)
    
    print("\nOptimal Investment Timing Probability Distribution:")
    print(opt_timing_prob)