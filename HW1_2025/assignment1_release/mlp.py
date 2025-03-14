from typing import List, Tuple

import torch
from torch import nn


class Linear(nn.Module):
    r"""Applies a linear transformation to the incoming data: :math:`y = xA^T + b`
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
    Shape:
        - Input: :math:`(*, H_{in})` where :math:`*` means any number of
          dimensions including none and :math:`H_{in} = \text{in\_features}`.
        - Output: :math:`(*, H_{out})` where all but the last dimension
          are the same shape as the input and :math:`H_{out} = \text{out\_features}`.

        >>> m = nn.Linear(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
    """

    def __init__(self, in_features: int, out_features: int) -> None:
        super(Linear, self).__init__()

        # follow pytorch init for these functions.
        val = torch.sqrt(torch.tensor(1 / in_features))
        self.weight = nn.Parameter(
            torch.rand(out_features, in_features) * (val * 2) - val
        )
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, input):
        """
        :param input: [bsz, in_features]
        :return result [bsz, out_features]
        """
        output = input @ self.weight.T

        bias = self.bias.expand_as(output)
        return output + bias


class MLP(torch.nn.Module):
    # 20 points
    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int],
        num_classes: int,
        activation: str = "relu",
    ):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        assert len(hidden_sizes) > 1, "You should at least have one hidden layer"
        self.num_classes = num_classes
        self.activation = activation
        assert activation in ["tanh", "relu", "sigmoid"], "Invalid choice of activation"
        self.hidden_layers, self.output_layer = self._build_layers(
            input_size, hidden_sizes, num_classes
        )

        # Initialization
        self._initialize_linear_layer(self.output_layer)
        for layer in self.hidden_layers:
            self._initialize_linear_layer(layer)

    def _build_layers(
        self, input_size: int, hidden_sizes: List[int], num_classes: int
    ) -> Tuple[nn.ModuleList, nn.Module]:
        """
        Build the layers for MLP. Be wary of corner cases.
        :param input_size: An int
        :param hidden_sizes: A list of ints. E.g., for [32, 32] means two hidden layers with 32 each.
        :param num_classes: An int
        :Return:
            hidden_layers: nn.ModuleList. Within the list, each item has type nn.Module
            output_layer: nn.Module
        """

        sizes = [input_size] + hidden_sizes

        hidden_layers = nn.ModuleList([])
        for in_size, out_size in zip(sizes[:-1], sizes[1:]):
            module = Linear(in_size, out_size)
            hidden_layers.append(module)

        output_layer = Linear(hidden_sizes[-1], num_classes)

        return hidden_layers, output_layer

    def activation_fn(self, activation, inputs: torch.Tensor) -> torch.Tensor:
        """process the inputs through different non-linearity function according to activation name"""

        if activation == "tanh":
            # tanh can be very unstable at float32 due to exp, so convert to float64
            term1 = torch.exp(inputs.to(dtype=torch.float64))
            term2 = torch.exp(-inputs.to(dtype=torch.float64))

            outputs = ((term1 - term2) / (term1 + term2)).to(dtype=inputs.dtype)
        elif activation == "relu":
            # relu output -- max(0, x)
            outputs = torch.maximum(inputs, torch.zeros_like(inputs))
        elif activation == "sigmoid":
            # sigmoid is super numerically unstable naively so we convert to float64 so then things stay in bounds
            outputs = 1.0 / (1.0 + torch.exp(-inputs.to(dtype=torch.float64)))
            outputs = outputs.to(dtype=inputs.dtype)

        return outputs

    def _initialize_linear_layer(self, module: nn.Linear) -> None:
        """For bias set to zeros. For weights set to glorot normal"""

        # first set bias to zeros
        module.bias.data.zero_()

        # compute standard deviation of glorot output (gain = 1 for linear layers)
        fan_out, fan_in = module.weight.size()
        std = torch.sqrt(torch.tensor(2.0 / (fan_out + fan_in)))

        # set weight to N(0, std^2)
        module.weight.data.normal_(mean=0.0, std=std**2)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Forward images and compute logits.
        1. The images are first fattened to vectors.
        2. Forward the result to each layer in the self.hidden_layer with activation_fn
        3. Finally forward the result to the output_layer.

        :param images: [batch, channels, width, height]
        :return logits: [batch, num_classes]
        """

        # first flatten the images.
        outputs = torch.flatten(images, start_dim=1, end_dim=-1)

        # now pass through each of the hidden modules.
        for hidden_layer in self.hidden_layers:
            outputs = hidden_layer(outputs)
            outputs = self.activation_fn(self.activation, outputs)

        # now pass through the output layer.
        outputs = self.output_layer(outputs)

        return outputs
