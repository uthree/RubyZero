# RubyZero::NN::Module
ニューラルネットワークの基本パーツ。

基本的にクラス継承をして使う。

`initialize` メソッド中に `@variable` でインスタンス変数を与えた場合、それが自動的にパラメータとなる。
また、`initialize` 内の最終行に`super`を呼び出さなければならない。

`forward` メソッドをオーバーライドすることで、順伝播処理を記述する。
順伝播処理を呼び出すときは、`forward`ではなく`call(*args)`または`.(*args)`を使用しなければならない。

## 使用例
```ruby
class TwoLP < RubyZero::NN::Module
    def initialize(input_units, mid_units, output_units)
        @l1 = RubyZero::NN::Linear(input_units, mid_units)
        @f2 = RubyZero::NN::ReLU
        @l2 = RubyZero::NN::Linear(mid_units, output_units)
        super # DONT FORGET CALL super
    end

    def forward(x)
        x = @l1.call(x)
        x = @f1.call(x)
        x = @l2.call(x)
        return x
    end
end

x = RubyZero::FloatTensor.zeros(5, 2)
model = TwoLP.new(2, 10, 1)
model.call(x)
```

## メソッド Module#call(*args) -> RubyZero::Tensor
モジュールの順伝播

## メソッド Module#parameters -> RubyZero::NN::Parameters
パラメータを取得

## メソッド Module#train -> nil
学習モードへ切り替え。
通常、学習開始前にこのメソッドを呼び出す。　

## メソッド Module#eval -> nil


