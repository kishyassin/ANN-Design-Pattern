package me.kichyassine.mekichyasssineanndp;

import me.kichyassine.mekichyasssineanndp.activation.ActivationAdapter;
import me.kichyassine.mekichyasssineanndp.activation.Sigmoid;
import me.kichyassine.mekichyasssineanndp.activation.Softmax;
import me.kichyassine.mekichyasssineanndp.aspect.NetworkAccess;
import me.kichyassine.mekichyasssineanndp.decorator.TimedLayer;
import me.kichyassine.mekichyasssineanndp.model.*;
import org.springframework.boot.autoconfigure.SpringBootApplication;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;

@SpringBootApplication
public class AnnDesignPatternApplication {

    public static void main(String[] args) {
        NetworkAccess.runAs(NetworkAccess.Role.OPERATOR, AnnDesignPatternApplication::demonstrateNetwork);
    }

    private static void demonstrateNetwork() {
        Neuron first = new Neuron(0);
        Neuron second = new Neuron(0);
        Neuron hidden = new Neuron(0.5);
        Neuron output = new Neuron(0.125);

        hidden.connectFrom(first, 0.5);
        hidden.connectFrom(second, 0.25);
        output.connectFrom(hidden, 0.5);

        NeuralNetwork network = new NeuralNetwork(
                new InputLayer(List.of(first, second)),
                List.of(new HiddenLayer(List.of(hidden))),
                new OutputLayer(List.of(output)));

        double[] input = {2, 4};
        Board dashboard = new Board();
        network.subscribe(dashboard);
        System.out.println("Initial state : " + network.state());
        try {
            network.predict(input);
            throw new AssertionError("Prediction was accepted before training.");
        } catch (IllegalStateException expected) {
            System.out.println("Prediction rejected before training : " + expected.getMessage());
        }
        network.fit();
        network.stop();
        network.train(new double[][]{input, {4, 4}}, new double[][]{{2}, {2.5}});
        double[] result = network.predict(input);
        System.out.println("Input : " + Arrays.toString(input));
        System.out.println("Theoretical initial output : [1.375]");
        System.out.println("Target : [2.0]");
        System.out.println("Prediction after training : " + Arrays.toString(result));
        if (network.state() != NetworkState.TRAINED || dashboard.transitions() != 4
                || result.length != 1 || Math.abs(result[0] - 2.0) > 1e-6) {
            throw new IllegalStateException("Incorrect lifecycle or training.");
        }
        demonstrateActivations();
        demonstrateDecorator(network, input, result);
        demonstrateSecurity(network, input);
    }

    private static void demonstrateSecurity(NeuralNetwork network, double[] input) {
        System.out.println("\nAccess control");
        NetworkAccess.runAs(NetworkAccess.Role.VIEWER, () -> {
            try {
                network.predict(input);
                throw new AssertionError("A VIEWER must not be able to predict.");
            } catch (SecurityException expected) {
                System.out.println("VIEWER : " + expected.getMessage());
            }
        });
        System.out.println("OPERATOR : authorized prediction " + Arrays.toString(network.predict(input)));
    }

    private static void demonstrateDecorator(NeuralNetwork network, double[] input, double[] expected) {
        System.out.println("\nTiming decorator");
        List<Long> durations = new ArrayList<>();
        network.decorateLayer(1, layer -> new TimedLayer(layer, durations::add));
        double[] withTiming = network.predict(input);
        if (durations.size() != 1 || !Arrays.equals(expected, withTiming)) {
            throw new IllegalStateException("The decorator must time a single propagation and preserve the output.");
        }
        System.out.printf("Hidden layer : %d ns%n", durations.get(0));
        System.out.printf("Prediction without / with decorator : %s / %s%n",
                Arrays.toString(expected), Arrays.toString(withTiming));
    }

    private static void demonstrateActivations() {
        Neuron neuron = new Neuron(2.0);
        double identity = neuron.compute();
        neuron.setActivation(new Sigmoid());
        double sigmoid = neuron.compute();
        neuron.setActivation(new ActivationAdapter(value -> value * value));
        double adapted = neuron.compute();

        OutputLayer output = new OutputLayer(List.of(new Neuron(1001), new Neuron(1000)), new Softmax());
        double[] probabilities = output.forward(new double[0]);
        double sum = Arrays.stream(probabilities).sum();

        System.out.println("\nActivations on score 2.0");
        System.out.printf(Locale.ROOT, "Identity = %.1f | Sigmoid = %.6f | Adapter = %.1f%n",
                identity, sigmoid, adapted);
        System.out.printf(Locale.ROOT, "Softmax([1001, 1000]) = [%.6f, %.6f] | sum = %.6f%n",
                probabilities[0], probabilities[1], sum);

        if (identity != 2.0 || Math.abs(sigmoid - 0.880797) > 1e-6 || adapted != 4.0
                || Math.abs(probabilities[0] - 0.731059) > 1e-6 || Math.abs(sum - 1.0) > 1e-12) {
            throw new IllegalStateException("Unexpected result for activations.");
        }
    }

}
