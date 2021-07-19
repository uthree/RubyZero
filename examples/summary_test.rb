require "../lib/ruby_zero"

include RubyZero

seq1 = NN::Sequential.new
seq1 << NN::Linear.new(10, 10)
seq1 << NN::ReLU
seq1 << NN::Linear.new(10, 10)
seq1 << NN::ReLU
seq1 << NN::Linear.new(10, 10)
seq1 << NN::ReLU

seq2 = NN::Sequential.new
seq2 << NN::Linear.new(10, 10)
seq2 << NN::ReLU
seq2 << NN::Linear.new(10, 10)
seq2 << NN::ReLU
seq2 << NN::Linear.new(10, 10)
seq2 << NN::ReLU

seq3 = NN::Sequential.new
seq3 << seq1
seq3 << seq2
seq3 << seq2
p seq3

# ------------------------------------------------------------
# summary of RubyZero::NN::Sequential
# ------------------------------------------------------------
# RubyZero::NN::ModuleList : 660
#  RubyZero::NN::Sequential : 330
#   RubyZero::NN::ModuleList : 330
#    RubyZero::NN::Linear : 110
#    RubyZero::NN::Linear : 110
#    RubyZero::NN::Linear : 110
#  RubyZero::NN::Sequential : 330
#   RubyZero::NN::ModuleList : 330
#    RubyZero::NN::Linear : 110
#    RubyZero::NN::Linear : 110
#    RubyZero::NN::Linear : 110
# ------------------------------------------------------------
# parameters: 660
# ------------------------------------------------------------