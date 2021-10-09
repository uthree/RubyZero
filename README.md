# Rubyzero

A Simple deep learning library for Ruby.

## Example
```ruby
require 'rubyzero'
include RubyZero
model = L.mlp(2,5,1) # 多層パーセプトロンモデルを初期化(入力2次元, 隠れ層5次元, 出力層1次元)
data = RubyZero::Data::Presets::Xor.new() # XOR演算のデータセット
trainer = Utils::Trainer.new(model) # 学習を勝手にしてくれるやつ
trainer.train(data, data, num_epochs:100) # 学習を自動的にするメソッド
p model # ついでにモデルの中身を表示
```

## Installation

Add this line to your application's Gemfile:

```ruby
gem 'rubyzero'
```

And then execute:

    $ bundle install

Or install it yourself as:

    $ gem install rubyzero

## Usage

TODO: Write usage instructions here

## Development

After checking out the repo, run `bin/setup` to install dependencies. Then, run `rake ` to run the tests. You can also run `bin/console` for an interactive prompt that will allow you to experiment.

To install this gem onto your local machine, run `bundle exec rake install`. To release a new version, update the version number in `version.rb`, and then run `bundle exec rake release`, which will create a git tag for the version, push git commits and the created tag, and push the `.gem` file to [rubygems.org](https://rubygems.org).

## Contributing

Bug reports and pull requests are welcome on GitHub at https://github.com/[USERNAME]/rubyzero.
