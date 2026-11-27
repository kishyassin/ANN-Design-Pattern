package me.kichyassine.mekichyasssineanndp.model;

import java.util.List;

public abstract class AbstractLayer implements Layer {
    private final List<Neuron> neurons;

    protected AbstractLayer(List<Neuron> neurons) {
        this.neurons = List.copyOf(neurons);
        if (this.neurons.isEmpty()) {
            throw new IllegalArgumentException("A layer must contain at least one neuron.");
        }
    }

    public final List<Neuron> neurons() {
        return neurons;
    }

    @Override
    public double[] forward(double[] input) {
        return neurons.stream().mapToDouble(Neuron::compute).toArray();
    }
}
