from network import *
import random


def train(network, amount_batches, per_batch, path, cost_path, learning_rate=0.01):
    batch_iteration = 1
    costs = []  # List to store average costs for each batch

    while batch_iteration <= amount_batches:
        average_cost = 0
        iteration = 1

        while iteration <= per_batch:
            # Generate random inputs and compute the expected outputs
            inputs = [random.uniform(-200, 200), random.uniform(-200, 200), random.uniform(-100, 100), random.uniform(-100, 100)]
            answer = inputs[1] > inputs[0] * inputs[2] + inputs[3]
            expected_outputs = [1 if answer else 0, 0 if answer else 1]

            # Forward pass
            outputs = network.compute(inputs)

            # Calculate the cost
            cost = sum([(outputs[0] - expected_outputs[0]) ** 2, (outputs[1] - expected_outputs[1]) ** 2])
            average_cost += cost

            # Calculate output layer deltas (error * derivative of activation function)
            deltas = []
            for i, neuron in enumerate(network.layers[-1].neurons):
                error = outputs[i] - expected_outputs[i]
                delta = error * outputs[i] * (1 - outputs[i])  # Derivative of sigmoid: y * (1 - y)
                deltas.append(delta)

            # Update weights and biases for output layer
            for i, neuron in enumerate(network.layers[-1].neurons):
                for j in range(len(neuron.weights)):
                    neuron.weights[j] -= learning_rate * deltas[i] * network.layers[-2].outputs[j]
                neuron.bias -= learning_rate * deltas[i]

            # Backpropagate to hidden layers
            for l in range(len(network.layers) - 2, 0, -1):  # Start from second-to-last layer
                layer = network.layers[l]
                next_layer = network.layers[l + 1]

                new_deltas = []
                for i, neuron in enumerate(layer.neurons):
                    error = sum([next_layer.neurons[k].weights[i] * deltas[k] for k in range(len(next_layer.neurons))])
                    delta = error * neuron.value * (1 - neuron.value)  # Derivative of sigmoid
                    new_deltas.append(delta)

                    # Update weights and biases for this layer
                    for j in range(len(neuron.weights)):
                        neuron.weights[j] -= learning_rate * delta * network.layers[l - 1].outputs[j]
                    neuron.bias -= learning_rate * delta

                deltas = new_deltas  # Update deltas for the next (earlier) layer

            iteration += 1

        # Calculate and display average cost for the batch
        average_cost /= per_batch
        costs.append(average_cost)  # Save the cost for later analysis
        print(f"Batch {batch_iteration}/{amount_batches}, Average Cost: {average_cost}")

        batch_iteration += 1

    # Save training data to a file
    with open(path, "w") as f:
        for l in network.layers:
            for n in l.neurons:
                f.write(f"{n.weights}:{n.bias}\n")

    # Save costs to a file for graphing
    with open(cost_path, "w") as f:
        f.write("Batch,Cost\n")  # Add a header for CSV
        for i, cost in enumerate(costs, start=1):
            f.write(f"{i},{cost}\n")


def manual_test(network, path):
    # Load weights and biases from file
    with open(path, "r") as f:
        data = f.readlines()

    weights_biases = [line.strip() for line in data]
    idx = 0

    # Ensure the number of weights and biases matches the network structure
    for l in network.layers:
        for n in l.neurons:
            if idx >= len(weights_biases):
                raise ValueError("Mismatch between file data and network structure.")

            weights_bias = weights_biases[idx]
            weights_str, bias_str = weights_bias.split(":")
            n.weights = [float(w) for w in weights_str.strip("[]").split(",")]
            n.bias = float(bias_str)
            idx += 1

    # User input for the line and point
    x1 = float(input("Enter X1 (point x-coordinate): "))
    x2 = float(input("Enter X2 (point y-coordinate): "))
    slope = float(input("Enter the slope of the line: "))
    intercept = float(input("Enter the y-intercept of the line: "))

    inputs = [x1, x2, slope, intercept]
    outputs = network.compute(inputs)
    print(f"Network outputs: {outputs}")

    if outputs[0] > outputs[1]:
        print("The point is ABOVE the line.")
    else:
        print("The point is BELOW the line.")


def automated_test(network, path, size_of_test):
    # Load weights and biases from file
    with open(path, "r") as f:
        data = f.readlines()

    weights_biases = [line.strip() for line in data]
    idx = 0

    # Ensure the number of weights and biases matches the network structure
    for l in network.layers:
        for n in l.neurons:
            if idx >= len(weights_biases):
                raise ValueError("Mismatch between file data and network structure.")

            weights_bias = weights_biases[idx]
            weights_str, bias_str = weights_bias.split(":")
            n.weights = [float(w) for w in weights_str.strip("[]").split(",")]
            n.bias = float(bias_str)
            idx += 1

    amount = 0
    correct = 0

    while amount < size_of_test:
        inputs = [random.uniform(-200, 200), random.uniform(-200, 200), random.uniform(-100, 100), random.uniform(-100, 100)]
        answer = inputs[1] > inputs[0] * inputs[2] + inputs[3]

        outputs = network.compute(inputs)
        network_answer = outputs[0] > outputs[1]

        if answer == network_answer:
            correct += 1
            amount += 1
            print(f"Iteration: {amount}, Network was Right")
        else:
            amount += 1
            print(f"Iteration: {amount}, Network was Wrong")

    percentage = correct / amount * 100

    print(f"The network got the test {percentage:.0f}% right.")


if __name__ == "__main__":
    network = Network([4, 64, 64, 64, 2])
    train(network, 1000, 100, "training_data/points.td", "training_data/points.csv")
    automated_test(network, "training_data/points.td", 25000)
