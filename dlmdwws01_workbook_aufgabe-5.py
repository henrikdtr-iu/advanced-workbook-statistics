import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

# ==============================================
# Aufgabenstellung 5: Regularisierte Regression
# ==============================================

# ----------------------------------------------
# Daten und symbolische Parameter
# ----------------------------------------------

# Gegebene Stuetzstellen des Graphen (xi_16)
stuetzstellen = np.array([
    [-17, -318247958961304.4],
    [20, 2204630655229896],
    [-1, -3.82],
    [-4, -30734989.16],
    [18, 711037006682266.2],
    [2, 32705.82],
    [13, 19041721141110.8],
    [-2, -6301.63],
    [-5, -391200362.04],
    [19, 1269997809934756.2],
    [6, 4120037495.73],
    [-13, -17315420230850.92],
    [-15, -82560387565709.7],
    [7, 22964271536.4],
    [17, 367223699069000],
    [11, 3180636470304.55],
    [-7, -17293218149.04],
    [-14, -38077802567641.65],
    [-10, -895197495233.78],
    [-11, -2526891348423.06],
    [4, 52153881.15],
    [0, -6.28],
    [12, 7808900471648.74],
    [10, 1087963626010.75],
    [-18, -587642528831354.8],
    [-3, -1062613.25]
], dtype=float)

# Symbolische Modellparameter a0 bis a12
a = sp.symbols('a0:13')

# Symbolischer Ridge-Parameter
ridge_lambda = sp.symbols('lambda')

# Einheitliche Groesse fuer alle Abbildungen
figsize = (7, 4)

# Bereich der Detaildarstellung
x_min_detail = -7.5
x_max_detail = 7.5

# ----------------------------------------------
# Funktionen
# ----------------------------------------------

# Berechnung der polynomialen Modellfunktion
def berechne_modellwert(x_i):

    y_dach = 0

    for k in range(len(a)):
        y_dach += a[k] * x_i**k

    return y_dach

# Berechnung der OLS-Kostenfunktion
def berechne_kostenfunktion():

    kostenfunktion = 0

    # Summe ueber alle Stuetzstellen
    for i in range(len(stuetzstellen)):

        # Dezimalwerte fuer eine exakte symbolische Berechnung umwandeln
        x_i = sp.Rational(str(stuetzstellen[i][0]))
        y_i = sp.Rational(str(stuetzstellen[i][1]))

        # Vorhergesagter Modellwert
        y_dach_i = berechne_modellwert(x_i)

        # Quadriertes Residuum zur Kostenfunktion addieren
        kostenfunktion += (y_i - y_dach_i)**2

    return sp.expand(kostenfunktion)

# Berechnung der partiellen Ableitungen der Kostenfunktion
def berechne_ableitungen(kostenfunktion):

    # Array fuer die partiellen Ableitungen
    dC_daj = []

    # Partielle Ableitungen nach a0 bis a12 berechnen
    for j in range(len(a)):
        ableitung_j = sp.diff(kostenfunktion, a[j])
        dC_daj.append(ableitung_j)

    return dC_daj

# Berechnung des Schaetzers
def berechne_schaetzer(dC_daj):

    gleichungen = []

    # Partielle Ableitungen gleich null setzen
    for ableitung in dC_daj:
        gleichungen.append(sp.Eq(ableitung, 0))

    # Gleichungssystem nach a0 bis a12 loesen
    a_dach = sp.solve(gleichungen, a)

    return a_dach

# Berechnung des Ridge-Strafterms
def berechne_ridge_strafterm(ridge_lambda):

    strafterm = 0

    # a0 wird nicht regularisiert
    for j in range(1, len(a)):
        strafterm += ridge_lambda * a[j]**2

    return strafterm

# Berechnung der Ridge-Kostenfunktion
def berechne_ridge_kostenfunktion(cost_C, ridge_lambda):

    ridge_strafterm = berechne_ridge_strafterm(ridge_lambda)

    cost_C_ridge = cost_C + ridge_strafterm

    return sp.expand(cost_C_ridge)

# Berechnung der Modellwerte anhand der geschaetzten Parameter
def berechne_modellwerte(x_werte, schaetzer):

    y_werte = 0

    for k in range(len(a)):
        y_werte += float(schaetzer[a[k]]) * x_werte**k

    return y_werte

# OLS- und Ridge-Modellfunktionen darstellen
def plot_modelle(x_plot, a_dach_ols, ridge_schaetzer):

    # OLS-Modellwerte berechnen und darstellen
    y_ols = berechne_modellwerte(x_plot, a_dach_ols)
    plt.plot(x_plot, y_ols, linewidth=2, label="OLS")

    # Ridge-Modellwerte berechnen und darstellen
    for lambda_wert, a_dach_ridge in ridge_schaetzer.items():

        y_ridge = berechne_modellwerte(x_plot, a_dach_ridge)

        plt.plot(
            x_plot,
            y_ridge,
            linewidth=2,
            label=rf"Ridge $\lambda={lambda_wert}$"
        )

# Gesamtdarstellung der Stuetzstellen und Naeherungsfunktionen
def plot_gesamtansicht(a_dach_ols, ridge_schaetzer):

    # Gesamten x-Bereich der Stuetzstellen verwenden
    x_min = np.min(stuetzstellen[:, 0])
    x_max = np.max(stuetzstellen[:, 0])

    x_plot = np.linspace(x_min, x_max, 500)
    
    # Neue Abbildung erstellen
    plt.figure(figsize=figsize)

    # Alle gegebenen Stuetzstellen darstellen
    plt.plot(
        stuetzstellen[:, 0],
        stuetzstellen[:, 1],
        linestyle="None",
        marker="x",
        markersize=5,
        label="Stuetzstellen"
    )

    # OLS- und Ridge-Modellfunktionen darstellen
    plot_modelle(x_plot, a_dach_ols, ridge_schaetzer)

    # Grenzen der Detaildarstellung markieren
    plt.axvline(
        x=x_min_detail,
        color="red",
        linewidth=2,
        label="Grenzen der Detaildarstellung"
    )

    plt.axvline(
        x=x_max_detail,
        color="red",
        linewidth=2
    )

    # Achsen und Darstellung
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.grid()
    plt.legend()
    plt.show()

# Detaildarstellung der Stuetzstellen und Naeherungsfunktionen
def plot_detailansicht(a_dach_ols, ridge_schaetzer):

    # x-Werte fuer den Detailbereich
    x_plot = np.linspace(x_min_detail, x_max_detail, 500)

    # Nur Stuetzstellen im dargestellten Bereich auswaehlen
    stuetzstellen_plot = stuetzstellen[
        (stuetzstellen[:, 0] >= x_min_detail)
        & (stuetzstellen[:, 0] <= x_max_detail)
    ]

    # Neue Abbildung erstellen
    plt.figure(figsize=figsize)

    # Stuetzstellen darstellen
    plt.plot(
        stuetzstellen_plot[:, 0],
        stuetzstellen_plot[:, 1],
        linestyle="None",
        marker="x",
        markersize=7,
        label="Stuetzstellen"
    )

    # OLS- und Ridge-Modellfunktionen darstellen
    plot_modelle(x_plot, a_dach_ols, ridge_schaetzer)

    # Achsen und Darstellung
    plt.xlabel("x")
    plt.ylabel("f(x)")

    # Darstellungsbereich begrenzen
    plt.xlim(x_min_detail, x_max_detail)

    plt.grid()
    plt.legend()
    plt.show()

# ----------------------------------------------
# Hauptprogramm
# ----------------------------------------------

def main():

    # OLS-Kostenfunktion berechnen
    cost_C = berechne_kostenfunktion()

    # Partielle Ableitungen berechnen
    dC_daj = berechne_ableitungen(cost_C)

    # OLS-Schaetzer berechnen
    a_dach_ols = berechne_schaetzer(dC_daj)

    # OLS-Schaetzer im Terminal ausgeben
    print("\nOLS-Schaetzer:")

    for j in range(len(a)):
        print(f"a_{j}_dach = {float(a_dach_ols[a[j]]):.3f}")

    # Verschiedene Lambda-Werte fuer Strafterm festlegen
    lambda_werte = [1, 10, 100, 1000]

    # Dictionary zur Speicherung der Ridge-Schaetzer anlegen
    ridge_schaetzer = {}

    # Ridge-Schaetzer fuer alle Lambda-Werte berechnen
    for lambda_wert in lambda_werte:

        # Ridge-Kostenfunktion berechnen
        cost_C_ridge = berechne_ridge_kostenfunktion(cost_C, lambda_wert)

        # Partielle Ableitungen berechnen
        dC_ridge_daj = berechne_ableitungen(cost_C_ridge)

        # Ridge-Schaetzer berechnen
        a_dach_ridge = berechne_schaetzer(dC_ridge_daj)

        # Ridge-Schaetzer im Dictionary speichern
        ridge_schaetzer[lambda_wert] = a_dach_ridge

        # Ridge-Schaetzer im Terminal ausgeben
        print(f"\nRidge-Schaetzer fuer Lambda = {lambda_wert}:")

        for j in range(len(a)):
            print(f"a_{j}_dach_ridge = {float(a_dach_ridge[a[j]]):.3f}")

    # Grafische Darstellung der Stuetzstellen
    plot_gesamtansicht(a_dach_ols, ridge_schaetzer)

    # Grafische Darstellung der OLS- und Ridge-Modellfunktionen
    plot_detailansicht(a_dach_ols, ridge_schaetzer)

if __name__ == "__main__":
    main()