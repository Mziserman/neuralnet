from network import Network

patterns = [
    [[0, 0], [0]],
    [[0, 1], [1]],
    [[1, 0], [1]],
    [[1, 1], [0]]
]

print("Creating neural network with shape [2, 4, 1]")
print("2 inputs, 4 hidden neurons, 1 output")
print()

n = Network([2, 4, 1])

print("Training for 5000 iterations...")
print()

n.train(patterns, iterations=5000, N=0.5, M=0.1)

print()
print("Testing predictions:")
print("-" * 40)
for p in patterns:
    inputs = p[0]
    target = p[1]
    n.feed_forward(inputs)
    output = n.layers[-1].neurons[0].output
    print(f"Input: {inputs} -> Target: {target[0]}, Output: {output:.4f}")
