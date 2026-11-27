package me.kichyassine.mekichyasssineanndp.model;

import me.kichyassine.mekichyasssineanndp.activation.Softmax;

import java.util.List;
import java.util.Objects;

public final class OutputLayer extends AbstractLayer {
    private final Softmax softmax;

    public OutputLayer(List<Neuron> neurons) {
        super(neurons);
        softmax = null;
    }

    public OutputLayer(List<Neuron> neurons, Softmax softmax) {
        super(neurons);
        this.softmax = Objects.requireNonNull(softmax);
    }

    @Override
    public double[] forward(double[] input) {
        double[] scores = super.forward(input);
        return softmax == null ? scores : softmax.apply(scores);
    }
}
