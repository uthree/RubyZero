module RubyZero::Utils
    class Trainer
        def initialize(model, loss_function: nil, optimizer: nil)
            @model = model
            @optimizer = optimizer || RubyZero::NN::Optimizers::Momentum.new(@model.parameters())
            @loss_function = loss_function || RubyZero::NN::Losses::MSE.new()
        end

        def train(train_data, test_data, num_epochs:1, batch_size:1, shuffle:true, show_graph:false, show_graph_finish:true)
            train_loader = RubyZero::Data::DataLoader.new(train_data, batch_size:batch_size, shuffle:shuffle)
            test_loader = RubyZero::Data::DataLoader.new(test_data, batch_size:batch_size, shuffle:shuffle)

            losses_train = []
            losses_test = []
            num_epochs.times do |epoch|
                losses_train_b = []
                losses_test_b = []
                train_loader.each do |input, target|
                    @optimizer.zero_grad
                    loss = @loss_function.call(@model.call(input), target)
                    loss.backward()
                    @optimizer.step()
                    losses_train_b << loss.data[0]
                end
                test_loader.each do |input, target|
                    loss = @loss_function.call(@model.call(input), target)
                    losses_test_b << loss.data[0]
                end
                avg_loss_train_b = losses_train_b.reduce(:+) / losses_train_b.size
                avg_loss_test_b = losses_test_b.reduce(:+) / losses_test_b.size
                losses_train << avg_loss_train_b
                losses_test << avg_loss_test_b
                if show_graph or (show_graph_finish and epoch == num_epochs-1)
                    clear_console()
                    plot = UnicodePlot.lineplot((0..epoch).to_a, losses_train, name:"train loss")
                    UnicodePlot.lineplot!(plot, (0..epoch).to_a, losses_test, name:"test loss")
                    plot.render()
                    puts "train loss:#{avg_loss_train_b}\n test loss:#{avg_loss_test_b}"
                end
            end
            return losses_train[-1]
        end
        
        private
        def clear_console()
            puts "\e[H\e[2J"
        end
    end
end