package me.kichyassine.mekichyasssineanndp.activation;

import java.util.Locale;
import java.util.Objects;

public final class ActivationFactory {
    private ActivationFactory() {
    }

    public static ActivationFunction create(ActivationType type) {
        Objects.requireNonNull(type, "Activation type must not be null.");
        return switch (type) {
            case IDENTITY -> new Identity();
            case SIGMOID -> new Sigmoid();
            case RELU -> new ReLU();
        };
    }

    public static ActivationFunction create(String name) {
        Objects.requireNonNull(name, "Activation name must not be null.");
        try {
            return create(ActivationType.valueOf(name.trim().toUpperCase(Locale.ROOT)));
        } catch (IllegalArgumentException invalid) {
            throw new IllegalArgumentException("Unknown activation function: " + name, invalid);
        }
    }
}
