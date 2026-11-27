package me.kichyassine.mekichyasssineanndp.aspect;

import java.util.Objects;

public final class NetworkAccess {
    public enum Role { OPERATOR, VIEWER }

    private static final ThreadLocal<Role> CURRENT_ROLE = new ThreadLocal<>();

    private NetworkAccess() {
    }

    public static void runAs(Role role, Runnable action) {
        Role previous = CURRENT_ROLE.get();
        CURRENT_ROLE.set(Objects.requireNonNull(role));
        try {
            action.run();
        } finally {
            if (previous == null) {
                CURRENT_ROLE.remove();
            } else {
                CURRENT_ROLE.set(previous);
            }
        }
    }

    static Role currentRole() {
        return CURRENT_ROLE.get();
    }
}
