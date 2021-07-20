# RubyZero::NN::Linear

全結合層。
数学的にはアフィン変換、`bias:false`で線形変換に相当する処理。

## shapes
input shape: `[nil, input_units]`
output shape: `[nil, output_units]`

## 使用例
```ruby
linear = RubyZero::NN::Linear(10, 5)
tensor = RubyZero::FloatTensor.zeros(5, 10)
linear.train
output = linear.call(tensor) # forward pass
output.backward # backward pass
p output
```

## 特異メソッド `self.new(input_units, output_units, bias:true)` `-> RubyZero::NN::Linear`

### パラメータ input_units : Integer
入力ベクトルの次元数。

### パラメータ output_units : Integer
出力ベクトルの次元数。

### パラメータ bias : [TrueClass|FalseClass]
バイアス項の有無

