package me.kichyassine.mekichyasssineanndp.aspect;

import org.aspectj.lang.JoinPoint;
import org.aspectj.lang.annotation.Aspect;
import org.aspectj.lang.annotation.Before;

@Aspect
public class SecurityAspect {
    @Before("execution(* me.kichyassine.mekichyasssineanndp.model.NeuralNetwork.fit(..))"
            + " || execution(* me.kichyassine.mekichyasssineanndp.model.NeuralNetwork.stop(..))"
            + " || execution(* me.kichyassine.mekichyasssineanndp.model.NeuralNetwork.train(..))"
            + " || execution(* me.kichyassine.mekichyasssineanndp.model.NeuralNetwork.predict(..))"
            + " || execution(* me.kichyassine.mekichyasssineanndp.model.NeuralNetwork.decorateLayer(..))")
    public void requireOperator(JoinPoint point) {
        if (NetworkAccess.currentRole() != NetworkAccess.Role.OPERATOR) {
            throw new SecurityException("Access denied to " + point.getSignature().getName()
                    + "() : OPERATOR role required.");
        }
    }
}
