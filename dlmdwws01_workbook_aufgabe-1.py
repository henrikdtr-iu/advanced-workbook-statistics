import matplotlib.pyplot as plt

# ==============================================
# Aufgabenstellung 1: Einfache Wahrscheinlichkeiten und Visualisierungen
# ==============================================

# ----------------------------------------------
# Parameter
# ----------------------------------------------

p_dafuer = 0.38
p_dagegen = 1 - p_dafuer

ausgaenge = ["dafuer", "dagegen"]
wahrscheinlichkeiten = [p_dafuer, p_dagegen]

# Einheitliche Groesse fuer alle Abbildungen
figsize = (7, 4)

# Farben festlegen
farben = ["#9ACD9A", "#E6A8A8"]

# ----------------------------------------------
# Funktionen
# ----------------------------------------------

# Grafische Darstellung der Wahrscheinlichkeiten
def plot_wahrscheinlichkeiten():

    plt.figure(figsize=figsize)

    plt.bar(ausgaenge, wahrscheinlichkeiten, color=farben)

    plt.xlabel("Abstimmung")
    plt.ylabel("Wahrscheinlichkeit")
    plt.ylim(0, 1)

    # Prozentwerte oberhalb der Balken anzeigen
    for i, p in enumerate(wahrscheinlichkeiten):
        plt.text(i, p + 0.02, f"{p:.0%}", ha="center")

    plt.show()

# ----------------------------------------------
# Hauptprogramm
# ----------------------------------------------

def main():

    plot_wahrscheinlichkeiten()

if __name__ == "__main__":
    main()