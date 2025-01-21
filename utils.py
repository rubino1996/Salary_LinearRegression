import logging

def compute_cost(x, y, w, b): 
    """
    Computes the cost function for linear regression.
    """
    m = x.shape[0]
    cost_sum = sum((w * x[i] + b - y[i]) ** 2 for i in range(m))
    return (1 / (2 * m)) * cost_sum

def compute_gradient(x, y, w, b): 
    """
    Computes the gradient for linear regression.
    """
    m = x.shape[0]
    dj_dw = sum((w * x[i] + b - y[i]) * x[i] for i in range(m)) / m
    dj_db = sum(w * x[i] + b - y[i] for i in range(m)) / m
    return dj_dw, dj_db

def update_parameters(w, b, dj_dw, dj_db, alpha):
    """
    Updates parameters using gradient descent step.
    """
    w -= alpha * dj_dw
    b -= alpha * dj_db
    return w, b

def gradient_descent(x, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters, tolerance=1e-6):
    """
    Performs gradient descent to optimize parameters w and b.
    
    Args:
        x (np.ndarray): Input features.
        y (np.ndarray): Target values.
        w_in (float): Initial value of parameter w.
        b_in (float): Initial value of parameter b.
        cost_function (function): Function to compute cost.
        gradient_function (function): Function to compute gradient.
        alpha (float): Learning rate.
        num_iters (int): Number of iterations.
        tolerance (float): Convergence tolerance for cost change.

    Returns:
        w (float): Optimized parameter w.
        b (float): Optimized parameter b.
        J_history (list): List of cost values at each iteration.
    """
    J_history = []
    w, b = w_in, b_in

    for i in range(num_iters):
        # Compute gradient
        dj_dw, dj_db = gradient_function(x, y, w, b)

        # Update parameters
        w, b = update_parameters(w, b, dj_dw, dj_db, alpha)

        # Compute cost and store it
        cost = cost_function(x, y, w, b)
        J_history.append(cost)

        # Log progress
        if i % (num_iters // 10) == 0 or i == num_iters - 1:
            logging.info(f"Iteration {i}: Cost = {cost:.6f}, w = {w:.6f}, b = {b:.6f}")

        # Check for convergence
        if i > 0 and abs(J_history[-2] - J_history[-1]) < tolerance:
            logging.info(f"Converged at iteration {i}")
            break

    return w, b, J_history
