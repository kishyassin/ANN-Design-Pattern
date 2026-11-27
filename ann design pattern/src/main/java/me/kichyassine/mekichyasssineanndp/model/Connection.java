package me.kichyassine.mekichyasssineanndp.model;

import java.util.Objects;

public record Connection(Neuron source, Neuron destination, double weight) {
    public Connection {
        Objects.requireNonNull(source);
        Objects.requireNonNull(destination);
        if (!Double.isFinite(weight) || weight < 0 || weight > 1) {
            throw new IllegalArgumentException("Weight must belong to [0, 1].");
        }
    }
}
