# plot the architecture of the neural network, showing the neurons 
# as circles and the connections as arrows. use graphviz and dot.

import graphviz

n_hidden = [6, 2]
# assume 2 input features and 1 output

dot = graphviz.Digraph(comment='Neural Network Architecture')
dot.attr(rankdir='LR')  # Left to right layout
dot.attr('node', width='0.5', height='0.5', fixedsize='true')  # Fixed size for all nodes

# Add nodes for input layer
with dot.subgraph() as s:
    s.attr(rank='same')
    for i in range(2):  # 2 input features
        s.node(f'input_{i}', f'x{i+1}', shape='circle')

# Add nodes for hidden layers
for layer_idx, n_neurons in enumerate(n_hidden):
    with dot.subgraph() as s:
        s.attr(rank='same')
        for neuron_idx in range(n_neurons):
            node_id = f'layer{layer_idx}_neuron{neuron_idx}'
            s.node(node_id, f'a{layer_idx}{neuron_idx}', shape='circle')
            
            # Connect to previous layer
            if layer_idx == 0:  # First hidden layer connects to input
                for input_idx in range(2):
                    dot.edge(f'input_{input_idx}', node_id)
            else:  # Other hidden layers connect to previous layer
                prev_layer_size = n_hidden[layer_idx-1]
                for prev_neuron_idx in range(prev_layer_size):
                    prev_node_id = f'layer{layer_idx-1}_neuron{prev_neuron_idx}'
                    dot.edge(prev_node_id, node_id)

# Add output node
with dot.subgraph() as s:
    s.attr(rank='same')
    s.node('output', 'y', shape='circle')

# Connect last hidden layer to output
last_layer_size = n_hidden[-1]
for neuron_idx in range(last_layer_size):
    node_id = f'layer{len(n_hidden)-1}_neuron{neuron_idx}'
    dot.edge(node_id, 'output')

# Save the visualization
dot.render('arhitektura', format='svg', cleanup=True)
print("Neural network visualization saved as 'arhitektura.svg'")

