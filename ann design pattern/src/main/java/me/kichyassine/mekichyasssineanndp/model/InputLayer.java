package me.kichyassine.mekichyasssineanndp.model;

import java.util.List;

public final class InputLayer extends AbstractLayer {
    public InputLayer(List<Neuron> neurons) {
        super(neurons);
    }

    @Override
    public double[] forward(double[] input) {
        if (input.length != neurons().size()) {
            throw new IllegalArgumentException("Incorrect dimension for input vector.");
        }
        for (int i = 0; i < input.length; i++) {
            neurons().get(i).acceptInput(input[i]);
        }
        return input.clone();
    }
}
