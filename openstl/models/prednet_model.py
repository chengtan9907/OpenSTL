import torch
from torch import nn
from torch.nn import functional as F

from openstl.modules import PredNetConvLSTMCell


class PredNet_Model(nn.Module):
    r"""PredNet Model

    Implementation of `Deep Predictive Coding Networks for Video Prediction
    and Unsupervised Learning <https://arxiv.org/abs/1605.08104>`_.

    """

    def __init__(self, configs, output_mode='error', **kwargs):
        super(PredNet_Model, self).__init__()
        self.configs = configs
        self.in_shape = configs.in_shape
        _, _, H, W = configs.in_shape

        self.a_channels = getattr(configs, "A_channels", (3, 48, 96, 192))
        self.r_channels = getattr(configs, "R_channels", (3, 48, 96, 192))
        self.n_layers = len(self.r_channels)
        self.r_channels += (0, )  # for convenience
        self.output_mode = output_mode
        self.gating_mode = getattr(configs, "gating_mode", 'mul')
        self.extrap_start_time = getattr(configs, "extrap_start_time", None)
        self.peephole = getattr(configs, "peephole", False)
        self.lstm_tied_bias = getattr(configs, "lstm_tied_bias", False)
        self.p_max = getattr(configs, "p_max", 1.0)

        # Input validity checks
        default_output_modes = ['prediction', 'error', 'pred+err']
        layer_output_modes = [unit + str(l) for l in range(self.n_layers) for unit in ['R', 'E', 'A', 'Ahat']]
        default_gating_modes = ['mul', 'sub']
        assert output_mode in default_output_modes + layer_output_modes, 'Invalid output_mode: ' + str(output_mode)
        assert self.gating_mode in default_gating_modes, 'Invalid gating_mode: ' + str(self.gating_mode)
        
        if self.output_mode in layer_output_modes:
            self.output_layer_type = self.output_mode[:-1]
            self.output_layer_num = int(self.output_mode[-1])
        else:
            self.output_layer_type = None
            self.output_layer_num = None

        # h, w = self.input_size

        for i in range(self.n_layers):
            # A_channels multiplied by 2 because E_l concactenates pred-target and target-pred
            # Hidden states don't have same size due to upsampling
            # How does this handle i = L-1 (final layer) | appends a zero

            if self.gating_mode == 'mul':	
                cell = PredNetConvLSTMCell((H, W), 2 * self.a_channels[i] + self.r_channels[i+1], self.r_channels[i],
                    (3, 3), gating_mode='mul', peephole=self.peephole, tied_bias=self.lstm_tied_bias)
            elif self.gating_mode == 'sub':
                cell = PredNetConvLSTMCell((H, W), 2 * self.a_channels[i] + self.r_channels[i+1], self.r_channels[i],
                    (3, 3), gating_mode='sub', peephole=self.peephole, tied_bias=self.lstm_tied_bias)

            setattr(self, 'cell{}'.format(i), cell)
            H = H // 2
            W = W // 2

        for i in range(self.n_layers):
            # Calculate predictions A_hat
            conv = nn.Sequential(nn.Conv2d(self.r_channels[i], self.a_channels[i], 3, padding=1), nn.ReLU())
            setattr(self, 'conv{}'.format(i), conv)

        self.upsample = nn.Upsample(scale_factor=2)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        for l in range(self.n_layers - 1):
            # Propagate error as next layer's target (line 16 of Lotter algo)
            # In channels = 2 * A_channels[l] because of pos/neg error concat
            # NOTE: Operation belongs to curr layer l and produces next layer  state l+1

            update_A = nn.Sequential(nn.Conv2d(2* self.a_channels[l], self.a_channels[l+1], (3, 3), padding=1), self.maxpool)
            setattr(self, 'update_A{}'.format(l), update_A)

        self.criterion = nn.MSELoss()

    def set_output_mode(self, output_mode):
        self.output_mode = output_mode

        # Input validity checks
        default_output_modes = ['prediction', 'error', 'pred+err']
        layer_output_modes = [unit + str(l) for l in range(self.n_layers) for unit in ['R', 'E', 'A', 'Ahat']]
        assert output_mode in default_output_modes + layer_output_modes, 'Invalid output_mode: ' + str(output_mode)
        
        if self.output_mode in layer_output_modes:
            self.output_layer_type = self.output_mode[:-1]
            self.output_layer_num = int(self.output_mode[-1])
        else:
            self.output_layer_type = None
            self.output_layer_num = None

    def step(self, a, states):
        batch_size = a.size(0)
        R_layers = states[:self.n_layers]
        C_layers = states[self.n_layers:2*self.n_layers]
        E_layers = states[2*self.n_layers:3*self.n_layers]

        if self.extrap_start_time is not None:
            t = states[-1]
            if t >= self.extrap_start_time: # if past self.extra_start_time use previous prediction as input
                a = states[-2]

        # Update representation units
        for l in reversed(range(self.n_layers)):
            cell = getattr(self, 'cell{}'.format(l))
            r_tm1 = R_layers[l]
            c_tm1 = C_layers[l]
            e_tm1 = E_layers[l]
            if l == self.n_layers - 1:
                r, c = cell(e_tm1, (r_tm1, c_tm1))
            else:
                tmp = torch.cat((e_tm1, self.upsample(R_layers[l+1])), 1)
                r, c = cell(tmp, (r_tm1, c_tm1))
            R_layers[l] = r
            C_layers[l] = c

        # Perform error forward pass
        for l in range(self.n_layers):
            conv = getattr(self, 'conv{}'.format(l))
            a_hat = conv(R_layers[l])
            if l == 0:
                a_hat= torch.min(a_hat, torch.tensor(self.p_max).to(self.configs.device)) # alternative SatLU (Lotter)
                frame_prediction = a_hat
            pos = F.relu(a_hat - a)
            neg = F.relu(a - a_hat)
            e = torch.cat([pos, neg],1)
            E_layers[l] = e
            
            # Handling layer-specific outputs
            if self.output_layer_num == l:
                if self.output_layer_type == 'A':
                    output = a
                elif self.output_layer_type == 'Ahat':
                    output = a_hat
                elif self.output_layer_type == 'R':
                    output = R_layers[l]
                elif self.output_layer_type == 'E':
                    output = E_layers[l]

            if l < self.n_layers - 1: # updating A for next layer
                update_A = getattr(self, 'update_A{}'.format(l))
                a = update_A(e)

        if self.output_layer_type is None:
            if self.output_mode == 'prediction':
                output = frame_prediction
            else:
                # Batch flatten (return 2D matrix) then mean over units
                # Finally, concatenate layers (batch, n_layers)
                mean_E_layers = torch.cat([torch.mean(e.view(batch_size, -1), axis=1, keepdim=True) for e in E_layers], axis=1)
                if self.output_mode == 'error':
                    output = mean_E_layers
                else:
                    output = torch.cat([frame_prediction.view(batch_size, -1), mean_E_layers], axis=1)

        states = R_layers + C_layers + E_layers
        if self.extrap_start_time is not None:
            states += [frame_prediction, t+1]
        return output, states

    def forward(self, input_tensor, **kwargs):

        R_layers = [None] * self.n_layers
        C_layers = [None] * self.n_layers
        E_layers = [None] * self.n_layers

        _, _, h, w = self.in_shape
        batch_size = input_tensor.size(0)

        # Initialize states
        for l in range(self.n_layers):
            R_layers[l] = torch.zeros(batch_size, self.r_channels[l], h, w, requires_grad=True).to(self.configs.device)
            C_layers[l] = torch.zeros(batch_size, self.r_channels[l], h, w, requires_grad=True).to(self.configs.device)
            E_layers[l] = torch.zeros(batch_size, 2*self.a_channels[l], h, w, requires_grad=True).to(self.configs.device)
            # Size of hidden state halves from each layer to the next
            h = h//2
            w = w//2

        states = R_layers + C_layers + E_layers
        # Initialize previous_prediction
        if self.extrap_start_time is not None:
            frame_prediction = torch.zeros_like(input_tensor[:,0], dtype=torch.float32).to(self.configs.device)
            states += [frame_prediction, -1] # [a, t]
            
        num_time_steps = input_tensor.size(1)
        total_output = [] # contains output sequence
        for t in range(num_time_steps):
            a = input_tensor[:,t].type(torch.FloatTensor).to(self.configs.device)
            output, states = self.step(a, states)
            total_output.append(output)

        ax = len(output.shape)
        # print(output.shape)
        total_output = [out.view(out.shape + (1,)) for out in total_output]
        total_output = torch.cat(total_output, axis=ax) # (batch, ..., nt)

        return total_output
