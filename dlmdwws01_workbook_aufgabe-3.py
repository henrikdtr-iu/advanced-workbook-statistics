import sympy as sp

# ==============================================
# Aufgabenstellung 3: Transformierte Zufallsvariablen
# ==============================================

# ----------------------------------------------
# Daten und symbolische Parameter
# ----------------------------------------------

# Symbolisch Variablen definieren
a = sp.symbols("a", real=True)
b, theta = sp.symbols("b theta", positive=True)
t = sp.symbols("t", nonnegative=True)

# ----------------------------------------------
# Funktionen
# ----------------------------------------------

# Berechnung der Randdichte f_T(b)
def berechne_f_T_t():

   # Gemeinsame Dichte nach der Transformation
    f_g_ab = (
        1 / (28800 * theta**14)
        * ((a + b) / 2)**6
        * ((b - a) / 2)**6
        * sp.exp(-b / theta)
    )

    # Integration ueber a im Bereich [-b, b]
    f_T_b = sp.integrate(f_g_ab, (a, -b, b))

    # Variable b durch t ersetzen
    f_T_t = f_T_b.subs(b, t)

    return f_T_t

# Berechnung des Maximum-Likelihood-Schaetzers fuer Theta
def berechne_mls_theta():

    # Im Experiment ermittelte Gesamtbandbreiten
    t_werte = [10, 5, 121, 126, 94]

    # Log-Likelihood aufstellen
    log_likelihood = 0

    for t_i in t_werte:
        log_likelihood += sp.log(
            (t_i**13 * sp.exp(-t_i / theta))
            / (172972800 * theta**14)
        )

   # Ableitung der Log-Likelihood-Funktion nach Theta
    ableitung = sp.diff(log_likelihood, theta)

    # Maximum-Likelihood-Schaetzer durch Nullsetzen der Ableitung bestimmen
    theta_mls = sp.solve(ableitung, theta)[0]

    return theta_mls

# Berechnung des Erwartungswerts der Gesamtbandbreite T
def berechne_erwartungswert_T(f_T_t, theta_mls):

    # Erwartungswert E[T] = Integral von t * f_T(t) ueber [0, unendlich)
    erwartungswert_T = sp.integrate(t * f_T_t, (t, 0, sp.oo))

    # Modellparameter theta_MLS in den Erwartungswert einsetzen
    erwartungswert_T_theta = erwartungswert_T.subs(theta, theta_mls)

    return erwartungswert_T, erwartungswert_T_theta
    
# ----------------------------------------------
# Hauptprogramm
# ----------------------------------------------

def main():

    # Randdichte berechnen und ausgeben
    f_T_t = berechne_f_T_t()
    print("f_T(t) =", f_T_t)

    # Maximum-Likelihood-Schaetzer berechnen und ausgeben
    theta_mls = berechne_mls_theta()
    print("theta_MLS =", sp.N(theta_mls), "TB")

    # Erwartungswert berechnen und ausgeben
    erwartungswert_T, erwartungswert_T_theta = berechne_erwartungswert_T(f_T_t, theta_mls)
    print("E[T] =", erwartungswert_T)
    print("E[T]_Dach =", erwartungswert_T_theta, "=", sp.N(erwartungswert_T_theta), "TB")

if __name__ == "__main__":
    main()