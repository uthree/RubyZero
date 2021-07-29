# RubyZero::NN::Conv1d

1次元畳み込み層。

## shapes
batch_size, width は任意の実数

input shape: `[batch_size, width, input_channel]`

output shape: `[batch_size, width-kernel+padding, output_channel]`

## 特異メソッド `self.new(input_channel, output_channel, kernel, padding:0, stride:1)` -> `RubyZero::NN::Conv1D`
初期化を行う。

### パラメータ `input_channel` : `Integer`
入力チャネル数

### パラメータ `output_channel` : `Integer`
出力チャネル数

### パラメータ `kernel` : `Integer`
カーネルサイズ。

### オプション `paddng` : `Integer`
パディングの数。
デフォルトは`0`

### オプション `stride` : `Integer`
ストライドの数。
デフォルトは`1`