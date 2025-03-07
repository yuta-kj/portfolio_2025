from imports import (
  nn,
  torch
)

class CustomNeuralNetClsf(nn.Module):
  def __init__(self, hidden_dim_lst, activation, input_dim=10, output_dim=1):
    super(CustomNeuralNetClsf, self).__init__()
    self.activation = activation()
    self.layers_dim_lst = [input_dim] + hidden_dim_lst + [output_dim]
    self.layers = nn.ModuleList()
    self.batchnorms = nn.ModuleList()

    # 隠れ層とバッチ正規化層の作成
    for i in range(len(self.layers_dim_lst) - 1):
      inp_dim = self.layers_dim_lst[i]
      out_dim = self.layers_dim_lst[i + 1]
      self.layers.append(nn.Linear(inp_dim, out_dim))
      if i < len(self.layers_dim_lst) - 2:  # 最後の出力層にはバッチ正規化を適用しない
        self.batchnorms.append(nn.BatchNorm1d(out_dim))

  def forward(self, x):
    # print(self.layers_dim_lst)
    # import pdb; pdb.set_trace()
    for i, layer in enumerate(self.layers[:-1]):
      x = layer(x)
      x = self.batchnorms[i](x)
      x = self.activation(x)
    x = self.layers[-1](x)
    x = torch.sigmoid(x)
    return x