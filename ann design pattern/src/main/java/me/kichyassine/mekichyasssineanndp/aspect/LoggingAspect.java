package me.kichyassine.mekichyasssineanndp.aspect;

import org.aspectj.lang.JoinPoint;
import org.aspectj.lang.annotation.Aspect;
import org.aspectj.lang.annotation.Before;
import org.aspectj.lang.annotation.DeclarePrecedence;

@Aspect
@DeclarePrecedence("me.kichyassine.mekichyasssineanndp.aspect.LoggingAspect, me.kichyassine.mekichyasssineanndp.aspect.SecurityAspect")
public class LoggingAspect {
    @Before("execution(* me.kichyassine.mekichyasssineanndp..*(..)) && !within(me.kichyassine.mekichyasssineanndp.aspect..*)")
    public void trace(JoinPoint point) {
        if (Boolean.getBoolean("network.trace")) {
            System.out.println("[TRACE] " + point.getSignature().toShortString());
        }
    }
}
