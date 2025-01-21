import numpy as np
import matplotlib.pyplot as plt
from utils import compute_cost, compute_gradient, gradient_descent
import logging


def main():
    # Load the dataset
    file_path = "salary_dataset.csv"  # Path to the dataset
    try:
        data = np.loadtxt(file_path, delimiter=",", skiprows=1)  # Skip header if present
        x_train = data[:, 1]  # Second column as features
        y_train = data[:, -1]  # Last column as target
    except FileNotFoundError:
        logging.error(f"Dataset not found at {file_path}")
        return

    logging.info(f"Loaded dataset with {x_train.shape[0]} samples.")

    # Visualize the data
    plt.scatter(x_train, y_train, marker='x', c='r')
    plt.title("Salary vs Years of Experience")
    plt.ylabel("Salary")
    plt.xlabel("Years of Experience")
    plt.show()

    # Initial parameters
    initial_w, initial_b = 0, 0
    iterations = 1000
    alpha = 0.01

    # Perform Gradient Descent
    logging.info("Starting Gradient Descent...")
    w, b, J_history = gradient_descent(
        x_train, y_train, initial_w, initial_b, compute_cost, compute_gradient, alpha, iterations
    )

    logging.info(f"Optimized parameters: w = {w:.4f}, b = {b:.4f}")

    # Plot cost function history
    plt.plot(J_history)
    plt.title("Cost Function History")
    plt.ylabel("Cost")
    plt.xlabel("Iteration")
    plt.show()

    # Predict and plot the linear fit
    predictions = w * x_train + b
    plt.scatter(x_train, y_train, marker='x', c='r', label="Actual Data")
    plt.plot(x_train, predictions, c='b', label="Prediction")
    plt.title("Salary vs Years of Experience")
    plt.ylabel("Salary")
    plt.xlabel("Years of Experience")
    plt.legend()
    plt.show()

    # Output final results
    logging.info(f"Model Training Complete. Final Cost: {J_history[-1]:.4f}")
    print(f"Optimized w: {w}, b: {b}")
    print("Training Complete.")

if __name__ == "__main__":
    main()
