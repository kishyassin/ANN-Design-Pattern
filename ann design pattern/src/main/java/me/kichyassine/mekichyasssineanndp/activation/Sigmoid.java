package me.kichyassine.mekichyasssineanndp.activation;

public final class Sigmoid implements ActivationFunction {
    @Override
    public double apply(double value) {
        return 1.0 / (1.0 + Math.exp(-value));
    }
}
