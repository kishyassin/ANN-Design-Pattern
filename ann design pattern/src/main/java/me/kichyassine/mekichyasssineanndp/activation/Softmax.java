package me.kichyassine.mekichyasssineanndp.activation;

import java.util.Arrays;

public final class Softmax {
    public double[] apply(double[] scores) {
        double max = Arrays.stream(scores).max()
                .orElseThrow(() -> new IllegalArgumentException("Softmax requires at least one score."));
        double[] probabilities = Arrays.stream(scores)
                .map(score -> Math.exp(score - max))
                .toArray();
        double sum = Arrays.stream(probabilities).sum();
        for (int i = 0; i < probabilities.length; i++) {
            probabilities[i] /= sum;
        }
        return probabilities;
    }
}
