package me.kichyassine.mekichyasssineanndp.model;

@FunctionalInterface
public interface NetworkObserver {
    void onStateChanged(NetworkState previous, NetworkState current);
}
