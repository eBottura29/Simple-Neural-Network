# Had to learn matplotlib just for this project
# You're welcome... (wasn't even that difficult)

import matplotlib.pyplot as plt
import csv
import numpy as np

# Read costs from CSV file
batches = []
costs = []

with open("training_data/points.csv", "r") as f:
    reader = csv.reader(f)
    next(reader)  # Skip header
    for row in reader:
        batches.append(int(row[0]))
        costs.append(float(row[1]))

# Fit a higher-order polynomial to the cost data
degree = 10  # Degree of the polynomial
trend_coeffs = np.polyfit(batches, costs, degree)
trend_line = np.poly1d(trend_coeffs)

# Generate trend line values
trend_costs = trend_line(batches)

plt.plot(batches, costs, label="Cost", color="blue")
plt.plot(batches, trend_costs, label="Trend Line", color="red", linestyle="--")
plt.xlabel("Batch")
plt.ylabel("Average Cost")
plt.title("Cost Evolution During Training with Trend Line")
plt.legend()
plt.grid()
plt.show()
