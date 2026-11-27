package me.kichyassine.mekichyasssineanndp.decorator;

import me.kichyassine.mekichyasssineanndp.model.Layer;

import java.util.Objects;

public abstract class LayerDecorator implements Layer {
    private final Layer layer;

    protected LayerDecorator(Layer layer) {
        this.layer = Objects.requireNonNull(layer);
    }

    @Override
    public double[] forward(double[] input) {
        return layer.forward(input);
    }
}
