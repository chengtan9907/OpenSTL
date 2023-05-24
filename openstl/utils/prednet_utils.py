import numpy as np
import torch


def get_initial_states(input_shape, row_axis, col_axis, num_layers,
                       R_stack_sizes, stack_sizes, channel_axis,
                       device):
    # input_shape.shape: (batch_size, timeSteps, Channel, Height, Width)
    init_height = input_shape[row_axis]
    init_width = input_shape[col_axis]

    base_initial_state = np.zeros(input_shape)
    non_channel_axis = -1
    for _ in range(2):
        base_initial_state = np.sum(base_initial_state, axis=non_channel_axis)
    base_initial_state = np.sum(base_initial_state, axis=1)  # (batch_size, Channel)

    initial_states = []
    states_to_pass = ['R', 'C', 'E']  # R is `representation`, C is Cell state in ConvLSTM, E is `error`.
    num_layer_to_pass = {stp: num_layers for stp in states_to_pass}
    states_to_pass.append('Ahat')  # pass prediction in states so can use as actual for t+1 when extrapolating
    num_layer_to_pass['Ahat'] = 1

    for stp in states_to_pass:
        for l in range(num_layer_to_pass[stp]):
            downsample_factor = 2 ** l
            row = init_height // downsample_factor
            col = init_width // downsample_factor
            if stp in ['R', 'C']:
                stack_size = R_stack_sizes[l]
            elif stp == 'E':
                stack_size = stack_sizes[l] * 2
            elif stp == 'Ahat':
                stack_size = stack_sizes[l]

            output_size = stack_size * row * col  # flattened size
            reducer = np.zeros((input_shape[channel_axis], output_size))  # (Channel, output_size)
            initial_state = np.dot(base_initial_state, reducer)  # (batch_size, output_size)

            output_shape = (-1, stack_size, row, col)
            initial_state = torch.from_numpy(np.reshape(initial_state, output_shape)).float().to(
                device).requires_grad_()  # requires_grad=True
            initial_states += [initial_state]

    initial_states += [
        torch.zeros(1, dtype=torch.int).to(device)]  # the last state will correspond to the current timestep
    return initial_states