package me.kichyassine.mekichyasssineanndp;

import me.kichyassine.mekichyasssineanndp.activation.ActivationFactory;
import me.kichyassine.mekichyasssineanndp.activation.ActivationFunction;
import me.kichyassine.mekichyasssineanndp.activation.ActivationType;
import me.kichyassine.mekichyasssineanndp.activation.Identity;
import me.kichyassine.mekichyasssineanndp.activation.ReLU;
import me.kichyassine.mekichyasssineanndp.activation.Sigmoid;
import me.kichyassine.mekichyasssineanndp.activation.Softmax;
import me.kichyassine.mekichyasssineanndp.aspect.NetworkAccess;
import me.kichyassine.mekichyasssineanndp.builder.NeuralNetworkBuilder;
import me.kichyassine.mekichyasssineanndp.memento.NetworkHistory;
import me.kichyassine.mekichyasssineanndp.model.NetworkMemento;
import me.kichyassine.mekichyasssineanndp.model.NetworkState;
import me.kichyassine.mekichyasssineanndp.model.NeuralNetwork;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.NoSuchElementException;

import static org.junit.jupiter.api.Assertions.*;

class DesignPatternImprovementsTest {

    @Test
    void activationFactoryCreatesExpectedStrategies() {
        ActivationFunction identity = ActivationFactory.create(ActivationType.IDENTITY);
        ActivationFunction sigmoid = ActivationFactory.create(ActivationType.SIGMOID);
        ActivationFunction relu = ActivationFactory.create("relu");

        assertInstanceOf(Identity.class, identity);
        assertInstanceOf(Sigmoid.class, sigmoid);
        assertInstanceOf(ReLU.class, relu);

        assertEquals(3.5, identity.apply(3.5), 1e-9);
        assertEquals(0.5, sigmoid.apply(0.0), 1e-9);
        assertEquals(0.0, relu.apply(-4.2), 1e-9);
        assertEquals(4.2, relu.apply(4.2), 1e-9);

        assertThrows(IllegalArgumentException.class, () -> ActivationFactory.create("unknown"));
    }

    @Test
    void builderConstructsValidFullyConnectedFig11Network() {
        NetworkAccess.runAs(NetworkAccess.Role.OPERATOR, () -> {
            NeuralNetwork network = new NeuralNetworkBuilder()
                    .inputLayer(3)
                    .hiddenLayer(5, ActivationType.SIGMOID)
                    .outputLayer(2, ActivationFactory.create(ActivationType.IDENTITY), new Softmax())
                    .withWeightInitializer(() -> 0.4)
                    .withBiasInitializer(() -> 0.1)
                    .build();

            network.fit();
            network.stop();
            network.train(
                    new double[][]{{1.0, 2.0, 3.0}, {3.0, 2.0, 1.0}},
                    new double[][]{{0.7, 0.3}, {0.3, 0.7}});

            double[] probabilities = network.predict(new double[]{1.0, 2.0, 3.0});
            assertEquals(2, probabilities.length);
            assertEquals(1.0, probabilities[0] + probabilities[1], 1e-9);
            assertEquals(NetworkState.TRAINED, network.state());
        });
    }

    @Test
    void builderValidatesRequiredLayersAndPositiveSizes() {
        assertThrows(IllegalArgumentException.class, () -> new NeuralNetworkBuilder().inputLayer(0));
        assertThrows(IllegalStateException.class, () -> new NeuralNetworkBuilder().outputLayer(1, ActivationType.IDENTITY).build());
        assertThrows(IllegalStateException.class, () -> new NeuralNetworkBuilder().inputLayer(2).build());
    }

    @Test
    void mementoSavesAndRestoresBiasesAndState() {
        NetworkAccess.runAs(NetworkAccess.Role.OPERATOR, () -> {
            NeuralNetwork network = new NeuralNetworkBuilder()
                    .inputLayer(2)
                    .hiddenLayer(1, ActivationType.IDENTITY)
                    .outputLayer(1, ActivationType.IDENTITY)
                    .withWeightInitializer(() -> 0.5)
                    .withBiasInitializer(() -> 0.0)
                    .build();

            List<NetworkState> observedStates = new ArrayList<>();
            network.subscribe((previous, current) -> observedStates.add(current));

            network.fit();
            network.stop();

            NetworkHistory history = new NetworkHistory();
            assertTrue(history.isEmpty());
            assertThrows(NoSuchElementException.class, history::pop);

            // Train toward target 5.0 and save a checkpoint
            network.train(new double[][]{{2.0, 2.0}}, new double[][]{{5.0}});
            double[] trainedOutput = network.predict(new double[]{2.0, 2.0});
            assertEquals(5.0, trainedOutput[0], 1e-6);

            NetworkMemento trainedCheckpoint = network.saveCheckpoint();
            history.push(trainedCheckpoint);
            assertEquals(1, history.size());
            assertEquals(NetworkState.TRAINED, history.peek().state());
            assertNotNull(trainedCheckpoint.createdAt());

            // Roll back to an ARRET checkpoint, retrain to target 1.0, then restore the 5.0 checkpoint
            NetworkMemento arretCheckpoint = new NeuralNetworkBuilder()
                    .inputLayer(2)
                    .hiddenLayer(1, ActivationType.IDENTITY)
                    .outputLayer(1, ActivationType.IDENTITY)
                    .withWeightInitializer(() -> 0.5)
                    .build()
                    .saveCheckpoint();
            assertEquals(NetworkState.CONSTRUCTION, arretCheckpoint.state());

            network.restoreCheckpoint(arretCheckpoint);
            assertEquals(NetworkState.CONSTRUCTION, network.state());
            assertEquals(NetworkState.CONSTRUCTION, observedStates.get(observedStates.size() - 1));

            network.fit();
            network.stop();
            network.train(new double[][]{{2.0, 2.0}}, new double[][]{{1.0}});
            assertEquals(1.0, network.predict(new double[]{2.0, 2.0})[0], 1e-6);

            network.restoreCheckpoint(history.pop());
            assertTrue(history.isEmpty());
            assertEquals(NetworkState.TRAINED, network.state());
            assertEquals(5.0, network.predict(new double[]{2.0, 2.0})[0], 1e-6);
        });
    }

    @Test
    void mementoRejectsIncompatibleTopologyAndEnforcesSecurity() {
        NeuralNetwork small = new NeuralNetworkBuilder()
                .inputLayer(2)
                .outputLayer(1, ActivationType.IDENTITY)
                .withWeightInitializer(() -> 0.5)
                .build();
        NeuralNetwork larger = new NeuralNetworkBuilder()
                .inputLayer(3)
                .outputLayer(1, ActivationType.IDENTITY)
                .withWeightInitializer(() -> 0.5)
                .build();

        NetworkMemento smallCheckpoint = small.saveCheckpoint();

        NetworkAccess.runAs(NetworkAccess.Role.OPERATOR, () ->
                assertThrows(IllegalArgumentException.class, () -> larger.restoreCheckpoint(smallCheckpoint)));

        NetworkAccess.runAs(NetworkAccess.Role.VIEWER, () ->
                assertThrows(SecurityException.class, () -> small.restoreCheckpoint(smallCheckpoint)));
    }
}
