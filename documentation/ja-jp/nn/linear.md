# RubyZero::NN::Linear

全結合層。
数学的にはアフィン変換、`bias:false`で線形変換に相当する処理。

## shapes
input shape: `[nil, input_units]`
output shape: `[nil, output_units]`

## 特異メソッド `self.new(input_units, output_units, bias:true)` `-> RubyZero::NN::Linear`

### パラメータ input_units : Integer
入力ベクトルの次元数。

### パラメータ output_units : Integer
出力ベクトルの次元数。

### パラメータ bias : [TrueClass|FalseClass]
バイアス項の有無

