require_relative '../lib/rubyzero'


include RubyZero
model = L.mlp(2,5,1) # 多層パーセプトロンモデルを初期化(入力2次元, 隠れ層5次元, 出力層1次元)
data = RubyZero::Data::Presets::Xor.new() # XOR演算のデータセット
trainer = Utils::Trainer.new(model) # 学習を勝手にしてくれるやつ
trainer.train(data, data, num_epochs:100) # 学習を自動的にするメソッド
p model # ついでにモデルの中身を表示