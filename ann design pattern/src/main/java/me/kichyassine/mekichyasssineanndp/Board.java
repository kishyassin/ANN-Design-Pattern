package me.kichyassine.mekichyasssineanndp;

import me.kichyassine.mekichyasssineanndp.model.NetworkObserver;
import me.kichyassine.mekichyasssineanndp.model.NetworkState;


final class Board implements NetworkObserver {
    private int transitions;

    @Override
    public void onStateChanged(NetworkState previous, NetworkState current) {
        transitions++;
        System.out.printf("Dashboard [%d] : %s -> %s%n", transitions, previous, current);
    }

    int transitions() {
        return transitions;
    }
}
