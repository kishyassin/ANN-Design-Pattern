package me.kichyassine.mekichyasssineanndp.activation;

import java.util.Objects;

public final class ActivationAdapter implements ActivationFunction {
    private final LegacyActivation legacy;

    public ActivationAdapter(LegacyActivation legacy) {
        this.legacy = Objects.requireNonNull(legacy);
    }

    @Override
    public double apply(double value) {
        return legacy.evaluer(value);
    }
}
