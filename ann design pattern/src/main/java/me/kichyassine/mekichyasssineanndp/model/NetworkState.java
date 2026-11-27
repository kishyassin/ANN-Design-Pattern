package me.kichyassine.mekichyasssineanndp.model;

public enum NetworkState {
    CONSTRUCTION {
        @Override
        void fit(NeuralNetwork network) {
            network.validateTopology();
            network.transitionTo(EXECUTION);
        }
    },
    EXECUTION {
        @Override
        void stop(NeuralNetwork network) {
            network.transitionTo(ARRET);
        }
    },
    ARRET {
        @Override
        void train(NeuralNetwork network, double[][] inputs, double[][] targets) {
            network.runTraining(inputs, targets);
        }
    },
    TRAINING,
    TRAINED {
        @Override
        double[] predict(NeuralNetwork network, double[] input) {
            return network.forward(input);
        }
    };

    void fit(NeuralNetwork network) {
        throw unavailable("fit");
    }

    void stop(NeuralNetwork network) {
        throw unavailable("stop");
    }

    void train(NeuralNetwork network, double[][] inputs, double[][] targets) {
        throw unavailable("train");
    }

    double[] predict(NeuralNetwork network, double[] input) {
        throw unavailable("predict");
    }

    private IllegalStateException unavailable(String operation) {
        return new IllegalStateException(operation + "() unavailable in state " + this);
    }
}
