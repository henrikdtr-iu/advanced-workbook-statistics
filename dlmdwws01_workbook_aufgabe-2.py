import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

# ==============================================
# Aufgabenstellung 2: Einfache Wahrscheinlichkeiten und Visualisierungen
# ==============================================

# ----------------------------------------------
# Daten und symbolische Parameter
# ----------------------------------------------

# Symbolische Variable
y = sp.symbols("y")

# Gegebene Parameter (Periodische Dezimalzahlen exakt als Brueche)
xi_5 = sp.Rational(82, 99) # 0,828282... = 82/99
xi_6 = 8
xi_7 = sp.Rational(17, 99) # 0,171717... = 17/99
xi_8 = 7

# Kumulative Verteilungsfunktion
F_Y = 1 - (xi_5 * sp.exp(-xi_6 * y**2) + xi_7 * sp.exp(-xi_8 * y**2))

# Einheitliche Groesse fuer alle Abbildungen
figsize = (7, 4)

# Farben festlegen
farbe_graph = "#4C72B0"          # Blau
farbe_mittelwert = "#2E8B57"     # Gruen
farbe_sigma_dunkel = "#3CB371"   # Mittleres Gruen
farbe_sigma_hell = "#A9DFBF"     # Helles Gruen
farbe_q1 = "#D6C4F0"             # Helles Violett
farbe_q2 = "#A98AD6"             # Mittleres Violett
farbe_q3 = "#6F4E9C"             # Dunkles Violett

# ----------------------------------------------
# Funktionen
# ----------------------------------------------

# Berechnung der Wahrscheinlichkeit Intervall [2, 4]
def berechne_wahrscheinlichkeit_2_4(f_Y):

    wahrscheinlichkeit = sp.integrate(f_Y, (y, 2, 4))

    return wahrscheinlichkeit

# Berechnung des Mittelwerts und der Varianz
def berechne_mittelwert_varianz(f_Y):

    # Mittelwert E(Y)
    mittelwert = sp.integrate(y * f_Y, (y, 0, sp.oo))

    # Varianz Var(Y)
    varianz = sp.integrate((y - mittelwert)**2 * f_Y, (y, 0, sp.oo))

    return mittelwert, varianz

# Berechnung der Quartile
def berechne_quartile():

    # SymPy-Ausdruck in numerisch auswertbare Funktion umwandeln
    F_Y_num = sp.lambdify(y, F_Y, "numpy")

    # Zu berechnende Quantile festlegen
    q1 = 0.25
    q2 = 0.5
    q3 = 0.75

    # Numerische Loesung der Gleichung F_Y(x_q) = q durch Nullstellensuche von F_Y(x_q) - q (brentq)
    # Betrachtetes Intervall: [0, 4] (Stunden)
    Q1 = brentq(lambda x_q: F_Y_num(x_q) - q1, 0, 4)
    Q2 = brentq(lambda x_q: F_Y_num(x_q) - q2, 0, 4)
    Q3 = brentq(lambda x_q: F_Y_num(x_q) - q3, 0, 4)

    return Q1, Q2, Q3

# Darstellung der Wahrscheinlichkeitsdichtefunktion (x-Achse in Stunden)
def plot_dichtefunktion(f_Y, mittelwert, varianz, Q1, Q2, Q3):

    # Standardabweichung fuer die grafische Darstellung berechnen
    standardabweichung = sp.sqrt(varianz)

    # SymPy-Ausdruck in numerisch auswertbare Funktion umwandeln
    f_Y_num = sp.lambdify(y, f_Y, "numpy")

    # Wertebereich fuer die x-Achse festlegen
    y_werte = np.linspace(0, 4, 400)

    # Funktionswerte der Dichtefunktion berechnen
    f_werte = f_Y_num(y_werte)

    # Plot erstellen
    plt.figure(figsize=figsize)

    # Wahrscheinlichkeitsdichtefunktion darstellen
    plt.plot(y_werte, f_werte, color=farbe_graph, label=r"$f_Y(y)$")

    # Mittelwert einzeichnen
    plt.axvline(
        float(mittelwert),
        color=farbe_mittelwert,
        linestyle="-",
        label=rf"$E(Y) = {float(mittelwert):.3f}\,h$"
    )

    # Untere Grenze des Standardabweichungsbereichs einzeichnen
    plt.axvline(
        float(mittelwert - standardabweichung),
        color=farbe_sigma_hell,
        linestyle="-",
        label=rf"$E(Y)-\sigma_Y = {float(mittelwert - standardabweichung):.3f}\,h$"
    )

    # Obere Grenze des Standardabweichungsbereichs einzeichnen
    plt.axvline(
        float(mittelwert + standardabweichung),
        color=farbe_sigma_dunkel,
        linestyle="-",
        label=rf"$E(Y)+\sigma_Y = {float(mittelwert + standardabweichung):.3f}\,h$"
    )

    # Quartile einzeichnen
    plt.axvline(float(Q1), color=farbe_q1, linestyle=":", label=rf"$Q_1 = {Q1:.3f}\,h$")
    plt.axvline(float(Q2), color=farbe_q2, linestyle=":", label=rf"$Q_2 = {Q2:.3f}\,h$")
    plt.axvline(float(Q3), color=farbe_q3, linestyle=":", label=rf"$Q_3 = {Q3:.3f}\,h$")

    # Plot-Eigenschaften
    plt.xlabel("Wartezeit in Stunden")
    plt.ylabel(r"$f_Y(y)$")
    plt.xlim(0, 4)
    plt.grid(True)
    plt.legend()

    # Plot ausgeben
    plt.show()

# Darstellung als minutengenaues Histogramm
def plot_histogramm(mittelwert, varianz, Q1, Q2, Q3):

    # Standardabweichung fuer die grafische Darstellung berechnen
    standardabweichung = sp.sqrt(varianz)

    # SymPy-Ausdruck in numerisch auswertbare Funktion umwandeln
    F_Y_num = sp.lambdify(y, F_Y, "numpy")

    # Zeitraum von 0 bis 120 Minuten festlegen
    minuten = np.arange(0, 120)

    # Liste fuer die Wahrscheinlichkeiten der einzelnen Minutenintervalle
    wahrscheinlichkeiten_min = []

    # Wahrscheinlichkeit fuer jedes einzelne Minutenintervall berechnen
    for minute in minuten:

        # Grenzen des Minutenintervalls in Stunden umrechnen
        untere_grenze = minute / 60
        obere_grenze = (minute + 1) / 60

        # P(a <= Y < b) = F_Y(b) - F_Y(a)
        p = F_Y_num(obere_grenze) - F_Y_num(untere_grenze)

        wahrscheinlichkeiten_min.append(p)

    # Plot erstellen
    plt.figure(figsize=figsize)

    # Minutengenaue Wahrscheinlichkeiten darstellen
    plt.bar(
        minuten,
        wahrscheinlichkeiten_min,
        width=1,
        color=farbe_graph,
        alpha=0.6,
        label="Wahrscheinlichkeiten je Minutenintervall"
    )

    # Mittelwert einzeichnen
    plt.axvline(
        float(mittelwert * 60),
        color=farbe_mittelwert,
        linestyle="-",
        linewidth=2,
        label=rf"$E(Y) = {float(mittelwert * 60):.2f}\,min$"
    )

    # Untere Standardabweichungsgrenze
    plt.axvline(
        float((mittelwert - standardabweichung) * 60),
        color=farbe_sigma_hell,
        linestyle="-",
        linewidth=2,
        label=rf"$E(Y)-\sigma_Y = {float((mittelwert - standardabweichung) * 60):.2f}\,min$"
    )

    # Obere Standardabweichungsgrenze
    plt.axvline(
        float((mittelwert + standardabweichung) * 60),
        color=farbe_sigma_dunkel,
        linestyle="-",
        linewidth=2,
        label=rf"$E(Y)+\sigma_Y = {float((mittelwert + standardabweichung) * 60):.2f}\,min$"
    )

    # Quartile einzeichnen
    plt.axvline(float(Q1 * 60), color=farbe_q1, linestyle="-", label=rf"$Q_1 = {Q1 * 60:.2f}\,min$")
    plt.axvline(float(Q2 * 60), color=farbe_q2, linestyle="-", label=rf"$Q_2 = {Q2 * 60:.2f}\,min$")
    plt.axvline(float(Q3 * 60), color=farbe_q3, linestyle="-", label=rf"$Q_3 = {Q3 * 60:.2f}\,min$")

    # Plot-Eigenschaften
    plt.xlabel("Wartezeit in Minuten")
    plt.ylabel("Wahrscheinlichkeit")
    plt.xlim(0, 120)
    plt.grid(axis="y", alpha=0.3)
    plt.legend()

    # Plot ausgeben
    plt.show()

# ----------------------------------------------
# Hauptprogramm
# ----------------------------------------------

def main():

    # Wahrscheinlichkeitsdichtefunktion berechnen und ausgeben
    f_Y = sp.diff(F_Y, y)
    print("f_Y(y) =", f_Y)

    # Wahrscheinlichkeit einer Wartezeit zwischen 2 und 4 Stunden berechnen und ausgeben
    wahrscheinlichkeit = berechne_wahrscheinlichkeit_2_4(f_Y)
    print("P(2 <= Y < 4) =", sp.N(wahrscheinlichkeit))

    # Mittelwert und Varianz berechnen und ausgeben
    mittelwert, varianz = berechne_mittelwert_varianz(f_Y)
    print("E(Y) =", sp.N(mittelwert), "Stunden =", sp.N(mittelwert * 60), "Minuten")
    print("Var(Y) =", sp.N(varianz), "Stunden^2 =", sp.N(varianz * 60**2), "Minuten^2")

    # Quartile berechnen und ausgeben
    Q1, Q2, Q3 = berechne_quartile()
    print("Q1 =", Q1, "Stunden =", Q1 * 60, "Minuten")
    print("Q2 =", Q2, "Stunden =", Q2 * 60, "Minuten")
    print("Q3 =", Q3, "Stunden =", Q3 * 60, "Minuten")

    # Wahrscheinlichkeitsdichtefunktion grafisch darstellen
    plot_dichtefunktion(f_Y, mittelwert, varianz, Q1, Q2, Q3)

    # Minutengenaues Histogramm darstellen
    plot_histogramm(mittelwert, varianz, Q1, Q2, Q3)

if __name__ == "__main__":
    main()