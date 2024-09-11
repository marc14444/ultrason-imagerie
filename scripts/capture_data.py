import serial
import time
import csv
import os

# Paramètres pour le port série
SERIAL_PORT = 'COM4'  # Assure-toi que cela correspond à ton port série correct
BAUD_RATE = 9600      # Assure-toi que cela correspond à la configuration de ton capteur
DATA_FILE = os.path.join('../data', 'simulated_data.csv')  # Chemin vers le fichier de données

def read_from_sensor():
    # Connexion au port série
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    time.sleep(2)  # Donne du temps au port série pour se stabiliser

    # Créer le répertoire si nécessaire
    if not os.path.exists('data'):
        os.makedirs('data')

    with open(DATA_FILE, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['x', 'y', 'distance'])  # En-têtes de colonnes

        print("Début de la lecture des données du capteur...")

        try:
            while True:
                if ser.in_waiting > 0:
                    # Lire la ligne venant du capteur
                    line = ser.readline().decode('utf-8').rstrip()
                    print(f"Data from sensor (raw): {line}")

                    # Suppose que le capteur envoie des données sous forme 'x,y,distance'
                    data = line.split(',')
                    if len(data) == 3:
                        try:
                            x, y, distance = map(float, data)
                            writer.writerow([int(x), int(y), distance])
                            print(f"Distance enregistrée : x={int(x)}, y={int(y)}, distance={distance}")
                        except ValueError:
                            print(f"Impossible de convertir les données : {line}")
                    else:
                        print(f"Données mal formatées : {line}")

                # Attendre un moment avant de refaire une lecture
                time.sleep(0.1)  # Ajuster ce délai selon la fréquence de détection souhaitée

        except KeyboardInterrupt:
            print("Arrêt de la lecture des données du capteur.")
        finally:
            ser.close()

if __name__ == '__main__':
    read_from_sensor()
