require "numo/narray" unless defined?(Numo)

Dir[File.expand_path('../core/', __FILE__) << '/*.rb'].each do |file|
    require file
end

Dir[File.expand_path('../functions/', __FILE__) << '/*.rb'].each do |file|
    require file
end

Dir[File.expand_path('../nn/', __FILE__) << '/*.rb'].each do |file|
    require file
end

Dir[File.expand_path('../losses/', __FILE__) << '/*.rb'].each do |file|
    require file
end

Dir[File.expand_path('../optimizers/', __FILE__) << '/*.rb'].each do |file|
    require file
end

Dir[File.expand_path('../data/', __FILE__) << '/*.rb'].each do |file|
    require file
end

Dir[File.expand_path('../utils/', __FILE__) << '/*.rb'].each do |file|
    require file
end

Dir[File.expand_path('../metrics/', __FILE__) << '/*.rb'].each do |file|
    require file
end