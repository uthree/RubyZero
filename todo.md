# TODO
- 書き直し Loss Functions
- 実装 ブロードキャスト
- 実装 整数・浮動小数点数のモンキーパッチ
- スカラー倍
- Square, Log, Sin, Cos, Tan, Sinh, Cosh, Tanh, etc...
- スライス関数
- Optimizerの改良
    - 複数のTensorを同時にoptimizeする処理を書く必要があるので、一つのテンソルに対する処理だけで完結するように, OptimizerUnitクラスを実装して、それをOptimizerがまとめるという形にする。

- ライブラリ名変更 RubyZeroだとバージョンを併記した場合RubyZero2.0みたいな表記になってややこしいので、わかりやすい名前にする
  - 候補
  - Candle (Torchが松明だからそれに近いものとして。)