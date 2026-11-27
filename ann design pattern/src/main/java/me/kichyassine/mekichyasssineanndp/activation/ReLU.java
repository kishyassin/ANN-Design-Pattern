package me.kichyassine.mekichyasssineanndp.activation;

public final class ReLU implements ActivationFunction {
    @Override
    public double apply(double value) {
        return Math.max(0.0, value);
    }
}
