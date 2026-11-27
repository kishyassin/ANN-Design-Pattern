package me.kichyassine.mekichyasssineanndp.model;

import me.kichyassine.mekichyasssineanndp.decorator.LayerDecorator;

import java.util.ArrayList;
import java.util.List;
import java.util.HashSet;
import java.util.Objects;
import java.util.Set;
import java.util.function.Function;

public final class NeuralNetwork {
    private final List<AbstractLayer> layers;

    private final List<NetworkObserver> observers = new ArrayList<>();
    private NetworkState state = NetworkState.CONSTRUCTION;

    private final List<Layer> forwardLayers;


    public NeuralNetwork(InputLayer input, List<HiddenLayer> hidden, OutputLayer output) {
        List<AbstractLayer> ordered = new ArrayList<>();
        ordered.add(input);
        ordered.addAll(hidden);
        ordered.add(output);
        layers = List.copyOf(ordered);
        forwardLayers = new ArrayList<>(layers);
    }

    public void decorateLayer(int index, Function<Layer, ? extends LayerDecorator> decorator) {
        Layer current = forwardLayers.get(index);
        forwardLayers.set(index, Objects.requireNonNull(decorator.apply(current)));
    }

    public NetworkState state() {
        return state;
    }

    public void subscribe(NetworkObserver observer) {
        observers.add(Objects.requireNonNull(observer));
    }

    public void fit() {
        state.fit(this);
    }

    public void stop() {
        state.stop(this);
    }

    public void train(double[][] inputs, double[][] targets) {
        state.train(this, inputs, targets);
    }

    public double[] predict(double[] input) {
        return state.predict(this, input);
    }

    public NetworkMemento saveCheckpoint() {
        List<double[]> snapshot = layers.stream()
                .map(layer -> layer.neurons().stream().mapToDouble(Neuron::bias).toArray())
                .toList();
        return new NetworkMemento(state, snapshot);
    }

    public void restoreCheckpoint(NetworkMemento memento) {
        Objects.requireNonNull(memento, "Checkpoint memento must not be null.");
        List<double[]> savedBiases = memento.layerBiases();
        if (savedBiases.size() != layers.size()) {
            throw new IllegalArgumentException("Incompatible checkpoint: layer count mismatch.");
        }
        for (int i = 0; i < layers.size(); i++) {
            List<Neuron> neurons = layers.get(i).neurons();
            if (savedBiases.get(i).length != neurons.size()) {
                throw new IllegalArgumentException("Incompatible checkpoint: neuron count mismatch at layer " + i + ".");
            }
        }
        for (int i = 0; i < layers.size(); i++) {
            List<Neuron> neurons = layers.get(i).neurons();
            double[] biases = savedBiases.get(i);
            for (int j = 0; j < neurons.size(); j++) {
                neurons.get(j).setBias(biases[j]);
            }
        }
        if (state != memento.state()) {
            transitionTo(memento.state());
        }
    }

    void transitionTo(NetworkState current) {
        NetworkState previous = state;
        state = current;
        for (NetworkObserver observer : List.copyOf(observers)) {
            try {
                observer.onStateChanged(previous, current);
            } catch (RuntimeException failure) {
                System.getLogger(NeuralNetwork.class.getName()).log(
                        System.Logger.Level.WARNING, "Network notification failed.", failure);
            }
        }
    }

    void validateTopology() {
        Set<Neuron> seen = new HashSet<>();
        List<Neuron> preceding = List.of();
        for (AbstractLayer layer : layers) {
            List<Neuron> sources = preceding;
            for (Neuron neuron : layer.neurons()) {
                if (!seen.add(neuron) || neuron.incoming().stream()
                        .anyMatch(connection -> !sources.contains(connection.source()))) {
                    throw new IllegalStateException("Invalid topology: reused neuron or out-of-layer connection.");
                }
            }
            preceding = layer.neurons();
        }
    }

    void runTraining(double[][] inputs, double[][] targets) {
        if (inputs.length == 0 || inputs.length != targets.length) {
            throw new IllegalArgumentException("A target is required for each training input.");
        }
        List<Neuron> outputs = layers.get(layers.size() - 1).neurons();
        for (int i = 0; i < inputs.length; i++) {
            if (inputs[i].length != layers.get(0).neurons().size()
                    || targets[i].length != outputs.size()) {
                throw new IllegalArgumentException("Incorrect training dataset dimensions.");
            }
        }
        double[] initialBiases = outputs.stream().mapToDouble(Neuron::bias).toArray();
        transitionTo(NetworkState.TRAINING);
        try {
            double best = loss(inputs, targets);
            double step = 0.5;
            for (int epoch = 0; epoch < 24; epoch++) {
                boolean improved = false;
                for (Neuron neuron : outputs) {
                    double initial = neuron.bias();
                    neuron.setBias(initial + step);
                    double plus = loss(inputs, targets);
                    neuron.setBias(initial - step);
                    double minus = loss(inputs, targets);
                    if (plus < best && plus <= minus) {
                        neuron.setBias(initial + step);
                        best = plus;
                        improved = true;
                    } else if (minus < best) {
                        best = minus;
                        improved = true;
                    } else {
                        neuron.setBias(initial);
                    }
                }
                if (!improved) {
                    step /= 2;
                }
            }
        } catch (RuntimeException failure) {
            for (int i = 0; i < outputs.size(); i++) {
                outputs.get(i).setBias(initialBiases[i]);
            }
            transitionTo(NetworkState.ARRET);
            throw failure;
        }
        transitionTo(NetworkState.TRAINED);
    }

    private double loss(double[][] inputs, double[][] targets) {
        double squaredError = 0;
        for (int i = 0; i < inputs.length; i++) {
            double[] predictions = forward(inputs[i]);
            for (int j = 0; j < predictions.length; j++) {
                double error = predictions[j] - targets[i][j];
                squaredError += error * error;
            }
        }
        if (!Double.isFinite(squaredError)) {
            throw new IllegalArgumentException("Training error must be finite.");
        }
        return squaredError / inputs.length;
    }

    double[] forward(double[] input) {
        double[] values = input;
        for (Layer layer : forwardLayers) {
            values = layer.forward(values);
        }
        return values;
    }
}
