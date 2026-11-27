package me.kichyassine.mekichyasssineanndp.model;



import me.kichyassine.mekichyasssineanndp.activation.ActivationFunction;
import me.kichyassine.mekichyasssineanndp.activation.Identity;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

public final class Neuron {
    private double bias;
    private final List<Connection> incoming = new ArrayList<>();
    private double output;
    private ActivationFunction activation;

    public Neuron(double bias) {
        this(bias, new Identity());
    }

    public Neuron(double bias, ActivationFunction activation) {
        this.bias = bias;
        setActivation(activation);
    }

    public void setActivation(ActivationFunction activation) {
        this.activation = Objects.requireNonNull(activation);
    }

    public void connectFrom(Neuron source, double weight) {
        incoming.add(new Connection(source, this, weight));
    }

    List<Connection> incoming() {
        return List.copyOf(incoming);
    }

    double bias() {
        return bias;
    }

    void setBias(double bias) {
        this.bias = bias;
    }

    void acceptInput(double value) {
        output = value;
    }

    public double compute() {
        double sum = bias;
        for (Connection connection : incoming) {
            sum += connection.source().output * connection.weight();
        }
        output = activation.apply(sum);
        return output;
    }
}
