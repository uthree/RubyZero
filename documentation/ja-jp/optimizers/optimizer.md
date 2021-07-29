# RubyZero::Optimizers::Optimizer

最適化手法の親クラス。
各種Optimizerは、このクラスを継承して作られる。

## 使用例

```ruby
module RubyZero::Optimizers
    class SGD < Optimizer
        def initialize(learning_rate:0.01)
            @lr = learning_rate
            super
        end
        def update_parameter(tensor)
            tensor -= tensor.grad * @lr
            return tensor
        end
    end
end
```

## 特異メソッド `self.new(*args, **kwargs, &block)` -> Optimizer
初期化

## メソッド `<<(nn_module)`
最適化するモジュールを追加します。
### パラメータ nn_module : RubyZero::NN::Module
追加するモジュール。

## メソッド `step()`
パラメータを更新します。
更新は`update_parameter(tensor)`メソッドによって行われます。

## メソッド `init_gradients()`
追加されたモジュールの学習可能なパラメータの勾配をゼロにします。

### エイリアス
 - init_grad
 - zero_grad

## プライベート メソッド `update_parameter(tensor)` -> Tensor
入力されたテンソルに対して、勾配を更新したテンソルを返すメソッドです
### パラメータ tensor : RubyZero::Tensor
勾配を更新する対象のテンソル。

## プライベート メソッド `on_add_module(nn_module)`
モジュールが追加された際に自動的に呼び出される関数。
パラメータ毎に対応する数値(例えば, Momentumの場合は `@velocity`)などを初期化したりする処理を書くのに使用します。