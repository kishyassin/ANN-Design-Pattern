package me.kichyassine.mekichyasssineanndp.builder;

import me.kichyassine.mekichyasssineanndp.activation.ActivationFactory;
import me.kichyassine.mekichyasssineanndp.activation.ActivationFunction;
import me.kichyassine.mekichyasssineanndp.activation.ActivationType;
import me.kichyassine.mekichyasssineanndp.activation.Softmax;
import me.kichyassine.mekichyasssineanndp.model.HiddenLayer;
import me.kichyassine.mekichyasssineanndp.model.InputLayer;
import me.kichyassine.mekichyasssineanndp.model.NeuralNetwork;
import me.kichyassine.mekichyasssineanndp.model.Neuron;
import me.kichyassine.mekichyasssineanndp.model.OutputLayer;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.ThreadLocalRandom;
import java.util.function.DoubleSupplier;

public final class NeuralNetworkBuilder {
    private int inputSize;
    private final List<LayerSpec> hiddenSpecs = new ArrayList<>();
    private LayerSpec outputSpec;
    private Softmax outputSoftmax;
    private DoubleSupplier weightInitializer = () -> ThreadLocalRandom.current().nextDouble();
    private DoubleSupplier biasInitializer = () -> 0.0;

    public NeuralNetworkBuilder inputLayer(int size) {
        validateSize(size, "Input layer");
        this.inputSize = size;
        return this;
    }

    public NeuralNetworkBuilder hiddenLayer(int size, ActivationFunction activation) {
        validateSize(size, "Hidden layer");
        hiddenSpecs.add(new LayerSpec(size, Objects.requireNonNull(activation)));
        return this;
    }

    public NeuralNetworkBuilder hiddenLayer(int size, ActivationType activationType) {
        return hiddenLayer(size, ActivationFactory.create(activationType));
    }

    public NeuralNetworkBuilder outputLayer(int size, ActivationFunction activation) {
        validateSize(size, "Output layer");
        this.outputSpec = new LayerSpec(size, Objects.requireNonNull(activation));
        this.outputSoftmax = null;
        return this;
    }

    public NeuralNetworkBuilder outputLayer(int size, ActivationType activationType) {
        return outputLayer(size, ActivationFactory.create(activationType));
    }

    public NeuralNetworkBuilder outputLayer(int size, ActivationFunction activation, Softmax softmax) {
        validateSize(size, "Output layer");
        this.outputSpec = new LayerSpec(size, Objects.requireNonNull(activation));
        this.outputSoftmax = Objects.requireNonNull(softmax);
        return this;
    }

    public NeuralNetworkBuilder withWeightInitializer(DoubleSupplier weightInitializer) {
        this.weightInitializer = Objects.requireNonNull(weightInitializer);
        return this;
    }

    public NeuralNetworkBuilder withBiasInitializer(DoubleSupplier biasInitializer) {
        this.biasInitializer = Objects.requireNonNull(biasInitializer);
        return this;
    }

    public NeuralNetwork build() {
        if (inputSize <= 0) {
            throw new IllegalStateException("An input layer must be configured before building the network.");
        }
        if (outputSpec == null) {
            throw new IllegalStateException("An output layer must be configured before building the network.");
        }

        List<Neuron> inputNeurons = createNeurons(inputSize, ActivationFactory.create(ActivationType.IDENTITY), false);
        InputLayer inputLayer = new InputLayer(inputNeurons);

        List<Neuron> previous = inputNeurons;
        List<HiddenLayer> hiddenLayers = new ArrayList<>();
        for (LayerSpec spec : hiddenSpecs) {
            List<Neuron> neurons = createNeurons(spec.size(), spec.activation(), true);
            connectFully(previous, neurons);
            hiddenLayers.add(new HiddenLayer(neurons));
            previous = neurons;
        }

        List<Neuron> outputNeurons = createNeurons(outputSpec.size(), outputSpec.activation(), true);
        connectFully(previous, outputNeurons);
        OutputLayer outputLayer = outputSoftmax == null
                ? new OutputLayer(outputNeurons)
                : new OutputLayer(outputNeurons, outputSoftmax);

        return new NeuralNetwork(inputLayer, hiddenLayers, outputLayer);
    }

    private List<Neuron> createNeurons(int size, ActivationFunction activation, boolean initializeBias) {
        List<Neuron> neurons = new ArrayList<>(size);
        for (int i = 0; i < size; i++) {
            double bias = initializeBias ? biasInitializer.getAsDouble() : 0.0;
            neurons.add(new Neuron(bias, activation));
        }
        return neurons;
    }

    private void connectFully(List<Neuron> sources, List<Neuron> destinations) {
        for (Neuron destination : destinations) {
            for (Neuron source : sources) {
                destination.connectFrom(source, weightInitializer.getAsDouble());
            }
        }
    }

    private static void validateSize(int size, String layerName) {
        if (size <= 0) {
            throw new IllegalArgumentException(layerName + " must contain at least one neuron.");
        }
    }

    private record LayerSpec(int size, ActivationFunction activation) {
    }
}
