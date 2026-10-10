#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>

// Si votre classe YieldCurve est dans yield_curve.cpp, assurez-vous qu'elle est déclarée ici ou via un #include "yield_curve.hpp"
// Sinon, vous pouvez placer la classe YieldCurve directement ici.

int main() {
    std::string filename = "daily-treasury-rates.csv";
    std::ifstream file(filename);
    
    if (!file.is_open()) {
        std::cerr << "Erreur : Impossible d'ouvrir le fichier " << filename << std::endl;
        return 1;
    }
    
    std::string line;
    std::string last_line;
    
    // On lit toutes les lignes pour garder uniquement la dernière (la plus récente)
    std::getline(file, line); // Ignore l'en-tête
    while (std::getline(file, line)) {
        if (!line.empty()) {
            last_line = line;
        }
    }
    file.close();

    // Découpage de la dernière ligne par rapport aux virgules
    std::stringstream ss(last_line);
    std::string cell;
    std::vector<std::string> row_values;

    while (std::getline(ss, cell, ',')) {
        row_values.push_back(cell);
    }

    if (row_values.empty()) {
        std::cerr << "Erreur : La ligne lue est vide." << std::endl;
        return 1;
    }

    // Affichage de la date (première colonne)
    std::cout << "Date des taux charges : " << row_values[0] << std::endl;

    // Définition des maturités standard des taux US Treasury (en années)
    // Ajustez ce vecteur selon le nombre de colonnes exactes de votre CSV FRED
    std::vector<double> maturities = {1.0/12.0, 2.0/12.0, 3.0/12.0, 6.0/12.0, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0};
    std::vector<double> rates;

    // On convertit les strings du CSV en double (en partant de la colonne 1)
    for (size_t i = 1; i < row_values.size() && (i - 1) < maturities.size(); ++i) {
        // Si vos taux dans le CSV sont en pourcentage (ex: 4.5), on divise par 100. 
        // S'ils sont déjà en décimales (ex: 0.045), retirez le / 100.0
        double r = std::stod(row_values[i]) / 100.0; 
        rates.push_back(r);
    }

    // Création de l'objet YieldCurve avec les données du marché
    // (Assurez-vous que votre classe YieldCurve accepte ce constructeur)
    // YieldCurve market_curve(maturities, rates);

    // --- EXPORT DE LA COURBE INTERPOLÉE ---
    std::ofstream outFile("interpolated_curve.csv");
    outFile << "Maturity,Rate\n";

    // Exemple de boucle d'export (décommentez market_curve.getRate(t) une fois l'objet créé)
    for (double t = 0.5; t <= 30.0; t += 0.5) {
        // double rate = market_curve.getRate(t);
        // outFile << t << "," << rate * 100 << "\n"; // en pourcentage pour le graphique
    }
    outFile.close();

    std::cout << "Fichier 'interpolated_curve.csv' généré avec succès !" << std::endl;
    
    return 0;
}