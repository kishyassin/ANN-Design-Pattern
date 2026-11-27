package me.kichyassine.mekichyasssineanndp.memento;

import me.kichyassine.mekichyasssineanndp.model.NetworkMemento;

import java.util.ArrayDeque;
import java.util.Deque;
import java.util.NoSuchElementException;
import java.util.Objects;

public final class NetworkHistory {
    private final Deque<NetworkMemento> checkpoints = new ArrayDeque<>();

    public void push(NetworkMemento memento) {
        checkpoints.push(Objects.requireNonNull(memento, "Checkpoint memento must not be null."));
    }

    public NetworkMemento pop() {
        if (checkpoints.isEmpty()) {
            throw new NoSuchElementException("No checkpoints available in history.");
        }
        return checkpoints.pop();
    }

    public NetworkMemento peek() {
        if (checkpoints.isEmpty()) {
            throw new NoSuchElementException("No checkpoints available in history.");
        }
        return checkpoints.peek();
    }

    public int size() {
        return checkpoints.size();
    }

    public boolean isEmpty() {
        return checkpoints.isEmpty();
    }
}
