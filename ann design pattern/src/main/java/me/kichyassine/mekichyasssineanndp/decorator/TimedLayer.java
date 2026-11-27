package me.kichyassine.mekichyasssineanndp.decorator;

import me.kichyassine.mekichyasssineanndp.model.Layer;

import java.util.Objects;
import java.util.function.LongConsumer;

public final class TimedLayer extends LayerDecorator {
    private final LongConsumer recordDuration;

    public TimedLayer(Layer layer, LongConsumer recordDuration) {
        super(layer);
        this.recordDuration = Objects.requireNonNull(recordDuration);
    }

    @Override
    public double[] forward(double[] input) {
        long start = System.nanoTime();
        double[] result = super.forward(input);
        recordDuration.accept(System.nanoTime() - start);
        return result;
    }
}
