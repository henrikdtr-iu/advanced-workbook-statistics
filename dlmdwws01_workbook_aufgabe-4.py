import numpy as np
from scipy.stats import t

# ==============================================
# Aufgabenstellung 4: Hypothesentest
# ==============================================

# ----------------------------------------------
# Daten und Parameter
# ----------------------------------------------

# Langfristiger Mittelwert der Hammergewichte in Gramm
xi_11 = 807

# Stichprobe der Hammergewichte in Gramm
xi_14 = np.array([783, 801, 782, 861, 834, 773, 894, 770, 776, 834])

# ----------------------------------------------
# Funktionen
# ----------------------------------------------

# Berechnung von Mittelwert und Standardabweichung der Stichprobe
def berechne_stichprobenwerte():

    # Stichprobenmittelwert
    mittelwert = np.mean(xi_14)

    # Anzahl der Beobachtungen
    n = len(xi_14)

    # Summe der quadrierten Abweichungen vom Mittelwert
    summe_quadratische_abweichungen = np.sum((xi_14 - mittelwert)**2)

    # Stichprobenvarianz mit Division durch n - 1
    varianz = summe_quadratische_abweichungen / (n - 1)

    # Stichprobenstandardabweichung
    standardabweichung = np.sqrt(varianz)

    return mittelwert, standardabweichung

# Berechnung der beobachteten t-Teststatistik
def berechne_t_beobachtet(mittelwert, standardabweichung):

    # Stichprobengroesse
    n = len(xi_14)

    # Beobachtete t-Teststatistik
    t_beobachtet = (
        (mittelwert - xi_11)
        / (standardabweichung / np.sqrt(n))
    )

    return t_beobachtet

# Berechnung des kritischen t-Werts
def berechne_t_cutoff():

    # Signifikanzniveau
    alpha = 0.05

    # Stichprobengroesse und Freiheitsgrade
    n = len(xi_14)
    freiheitsgrade = n - 1

    # Kritischer Wert t_cutoff fuer den unteren Ablehnungsbereich
    t_cutoff = t.ppf(alpha, freiheitsgrade)

    return t_cutoff

# ----------------------------------------------
# Hauptprogramm
# ----------------------------------------------

def main():

    # Stichprobenmittelwert und Stichprobenstandardabweichung berechnen und ausgeben
    mittelwert, standardabweichung = berechne_stichprobenwerte()
    print("Stichprobenmittelwert =", mittelwert, "g")
    print("Stichprobenstandardabweichung =", standardabweichung, "g")

    # Beobachtete t-Teststatistik berechnen und ausgeben
    t_beobachtet = berechne_t_beobachtet(mittelwert, standardabweichung)
    print("Beobachtete t-Teststatistik =", t_beobachtet)

    # Kritischen t-Wert berechnen und ausgeben
    t_cutoff = berechne_t_cutoff()
    print("Kritischer t-Wert =", t_cutoff)

if __name__ == "__main__":
    main()