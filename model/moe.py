r"""
Naive gate
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

r"""
Base gate with standard interface
"""
import torch.nn as nn


class BaseGate(nn.Module):
    def __init__(self, num_expert, world_size):
        super().__init__()
        self.world_size = world_size
        self.num_expert = num_expert
        self.tot_expert = world_size * num_expert
        self.loss = None

    def forward(self, x):
        raise NotImplementedError('Base gate cannot be directly used for fwd')

    def set_loss(self, loss):
        self.loss = loss

    def get_loss(self, clear=True):
        loss = self.loss
        if clear:
            self.loss = None
        return loss

    @property
    def has_loss(self):
        return self.loss is not None


class NaiveGate(BaseGate):
    r"""
    A naive gate implementation that defines the standard behavior of the gate
    which determines which experts the tokens are going to.
    Both the indicies and the score, or confidence, are output to the parent
    module.
    The load-balance strategies are also designed to be implemented within the
    `Gate` module.
    """

    def __init__(self, d_model, num_expert, world_size, top_k=1):
        super().__init__(num_expert, world_size)
        self.gate = nn.Linear(d_model, self.tot_expert)
        self.top_k = top_k
        self.observe = False
        self.d_model = d_model

    def forward(self, inp):
        r"""
        The naive implementation simply calculates the top-k of a linear layer's
        output.
        """
        gate = self.gate(inp)

        '''直接构造topk = 1的gate'''
        # gate = (gate[: ,:, 0] > gate[:, :, 1]).unsqueeze(-1).type(torch.float32)
        # gates = [gate, 1-gate]
        # shift_ratio = 0

        '''正常构造gate，fix ratio'''
        gate_top_k_val, gate_top_k_idx = torch.sort(gate, dim=-1, descending=True)
        top_k_logits = gate_top_k_val[:, : self.top_k]
        top_k_indices = gate_top_k_idx[:, : self.top_k]
            
        if self.top_k == 1:
            top_k_gates = torch.ones_like(top_k_indices, device=top_k_indices.device,dtype=torch.float32)
        else:
            top_k_gates = F.softmax(top_k_logits, dim=-1)
        zeros = torch.zeros_like(gate, requires_grad=True).type(torch.float32)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)
        return gates

class SparseDispatcher():
    """Helper for implementing a mixture of experts.
    The purpose of this class is to create input minibatches for the
    experts and to combine the results of the experts to form a unified
    output tensor.
    There are two functions:
    dispatch - take an input Tensor and create input Tensors for each expert.
    combine - take output Tensors from each expert and form a combined output
      Tensor.  Outputs from different experts for the same batch element are
      summed together, weighted by the provided "gates".
    The class is initialized with a "gates" Tensor, which specifies which
    batch elements go to which experts, and the weights to use when combining
    the outputs.  Batch element b is sent to expert e iff gates[b, e] != 0.
    The inputs and outputs are all two-dimensional [batch, depth].
    Caller is responsible for collapsing additional dimensions prior to
    calling this class and reshaping the output to the original shape.
    See common_layers.reshape_like().
    Example use:
    gates: a float32 `Tensor` with shape `[batch_size, num_experts]`
    inputs: a float32 `Tensor` with shape `[batch_size, input_size]`
    experts: a list of length `num_experts` containing sub-networks.
    dispatcher = SparseDispatcher(num_experts, gates)
    expert_inputs = dispatcher.dispatch(inputs)
    expert_outputs = [experts[i](expert_inputs[i]) for i in range(num_experts)]
    outputs = dispatcher.combine(expert_outputs)
    The preceding code sets the output for a particular example b to:
    output[b] = Sum_i(gates[b, i] * experts[i](inputs[b]))
    This class takes advantage of sparsity in the gate matrix by including in the
    `Tensor`s for expert i only the batch elements for which `gates[b, i] > 0`.
    """

    def __init__(self, num_experts, gates, top_k=2):
        """Create a SparseDispatcher."""

        self._gates = gates
        self._num_experts = num_experts

        if num_experts == 2 and top_k == 1: 
            ''' Codes here are able to have same result as dynamic computeGraph only if num_experts == 2 and top_k == 1'''
            ''' Use code below, we can get static computeGraph in TVM '''
            index = torch.arange(gates.shape[0],dtype=torch.long,device=gates.device)
            sorted_experts, index_sorted_experts = gates[:,1].type(torch.long).sort()

            expert_index = sorted_experts.unsqueeze(1)
            self._batch_index = index[index_sorted_experts]

            sorted_experts = torch.stack((index, sorted_experts), dim=1)
            index_sorted_experts = torch.stack((index, index_sorted_experts), dim=1)

        else:
            ''' A more versatile computing scheme '''
            indices = torch.where(gates != 0)
            nonzero_indices = torch.stack(indices, dim=1)
            sorted_experts, index_sorted_experts = nonzero_indices.sort(0)
            # get according batch index for each expert
            self._batch_index = sorted_experts[index_sorted_experts[:, 1],0]
            # drop indices
            _, expert_index = sorted_experts.split(1, dim=1) # get second column

        # calculate num samples that each expert gets
        self._part_sizes = (self._gates > 0).sum(0).tolist()
        # expand gates to match with self._batch_index
        gates_exp = self._gates[self._batch_index]
        self._nonzero_gates = torch.gather(gates_exp, 1, expert_index)

    def dispatch(self, inp):
        """Create one input Tensor for each expert.
        The `Tensor` for a expert `i` contains the slices of `inp` corresponding
        to the batch elements `b` where `gates[b, i] > 0`.
        Args:
          inp: a `Tensor` of shape "[batch_size, <extra_input_dims>]`
        Returns:
          a list of `num_experts` `Tensor`s with shapes
            `[expert_batch_size_i, <extra_input_dims>]`.
        """

        # assigns samples to experts whose gate is nonzero

        # expand according to batch index so we can just split by _part_sizes
        # print(inp[self._batch_index].shape)
        inp_exp = inp[self._batch_index].flatten(1)
        return torch.split(inp_exp, self._part_sizes, dim=0)

    def combine(self, expert_out, multiply_by_gates=True):
        """Sum together the expert output, weighted by the gates.
        The slice corresponding to a particular batch element `b` is computed
        as the sum over all experts `i` of the expert output, weighted by the
        corresponding gate values.  If `multiply_by_gates` is set to False, the
        gate values are ignored.
        Args:
          expert_out: a list of `num_experts` `Tensor`s, each with shape
            `[expert_batch_size_i, <extra_output_dims>]`.
          multiply_by_gates: a boolean
        Returns:
          a `Tensor` with shape `[batch_size, <extra_output_dims>]`.
        """
        # apply exp to expert outputs, so we are not longer in log space
        stitched = torch.cat(expert_out, 0).exp().type(torch.float32)

        if multiply_by_gates:
            stitched = stitched.mul(self._nonzero_gates)
        zeros = torch.zeros(self._gates.size(0), expert_out[-1].size(1), requires_grad=True, device=stitched.device)
        # combine samples that have been processed by the same k experts

        index = self._batch_index.unsqueeze(-1).expand_as(stitched)
        # combined = zeros.scatter_add(0, self._batch_index.unsqueeze(-1).expand_as(stitched), stitched.float())
        combined = zeros.scatter(0, index, stitched)
        
        # add eps to all zero values in order to avoid nans when going back to log space

        combined += np.finfo(float).eps
        # back to log space
        return combined.log()


class MLP(nn.Module):
    def __init__(self, ln, sigmoid_layer=-1):
        super(MLP, self).__init__()
        layers = nn.ModuleList()
        for i in range(0, ln.size - 1):
            n = ln[i]
            m = ln[i + 1]

            # construct fully connected operator
            LL = nn.Linear(int(n), int(m), bias=True)

            # initialize the weights
            # with torch.no_grad():
            # custom Xavier input, output or two-sided fill
            mean = 0.0  # std_dev = np.sqrt(variance)
            std_dev = np.sqrt(2 / (m + n))  # np.sqrt(1 / m) # np.sqrt(1 / n)
            W = np.random.normal(mean, std_dev, size=(m, n)).astype(np.float32)
            std_dev = np.sqrt(1 / m)  # np.sqrt(2 / (m + 1))
            bt = np.random.normal(mean, std_dev, size=m).astype(np.float32)
            # approach 1
            LL.weight.data = torch.tensor(W, requires_grad=True)
            LL.bias.data = torch.tensor(bt, requires_grad=True)
            # approach 2
            # LL.weight.data.copy_(torch.tensor(W))
            # LL.bias.data.copy_(torch.tensor(bt))
            # approach 3
            # LL.weight = Parameter(torch.tensor(W),requires_grad=True)
            # LL.bias = Parameter(torch.tensor(bt),requires_grad=True)
            layers.append(LL)

            # construct sigmoid or relu operator
            if i == sigmoid_layer:
                layers.append(nn.Sigmoid())
            else:
                layers.append(nn.ReLU())

        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class MOE(nn.Module):
    r"""
    A complete MoE MLP module in a Transformer block.
    * `activation` is the activation function to be used in MLP in each expert.
    * `d_hidden` is the dimension of the MLP layer.
    """

    def __init__(
        self,
        ln,
        num_expert=2,
        world_size=1,
        top_k=1,
        gate=NaiveGate,
        sigmoid_layer=-1,
    ):
        super(MOE, self).__init__()
        self.num_expert = num_expert
        self.d_model = ln[0]
        self.gate = gate(self.d_model, num_expert, world_size, top_k)
        self.top_k = top_k

        self.experts = nn.ModuleList()
        for _ in range(0,num_expert):
            self.experts.append(MLP(ln,sigmoid_layer))
        
        

    def forward(self, inp: torch.Tensor):
        r"""
        Args:
        x: tensor shape [batch_size, input_size]
        train: a boolean scalar.
        loss_coef: a scalar - multiplier on load-balancing losses
        Returns:
        y: a tensor with shape [batch_size, output_size].
        extra_training_loss: a scalar.  This should be added into the overall
        training loss of the model.  The backpropagation of this loss
        encourages all experts to be approximately equally used across a batch.
        """

        gates = self.gate(inp)

        dispatcher = SparseDispatcher(self.num_expert, gates, self.top_k)
        expert_inputs = dispatcher.dispatch(inp)
        expert_outputs = [self.experts[i](expert_inputs[i]) for i in range(self.num_expert)]
        y = dispatcher.combine(expert_outputs)

        return y