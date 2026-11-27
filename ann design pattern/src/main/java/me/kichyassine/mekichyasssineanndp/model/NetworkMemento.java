package me.kichyassine.mekichyasssineanndp.model;

import java.time.Instant;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

public final class NetworkMemento {
    private final NetworkState state;
    private final List<double[]> layerBiases;
    private final Instant createdAt;

    NetworkMemento(NetworkState state, List<double[]> layerBiases) {
        this.state = Objects.requireNonNull(state);
        this.layerBiases = layerBiases.stream()
                .map(double[]::clone)
                .toList();
        this.createdAt = Instant.now();
    }

    public NetworkState state() {
        return state;
    }

    public Instant createdAt() {
        return createdAt;
    }

    List<double[]> layerBiases() {
        return layerBiases.stream()
                .map(biases -> Arrays.copyOf(biases, biases.length))
                .toList();
    }
}
