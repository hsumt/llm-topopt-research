def validate(metrics: dict, rho_final: np.ndarray,
             volfrac_target: float, n_cells: int) -> dict:
    
    # Check #1 that the compliance is monotonically decreasing after iter 5. 
    # Check that vol fraction is within the target range
    # No checkerboards
    # Density field is all 0s and 1s
    monkey = metrics
    return monkey