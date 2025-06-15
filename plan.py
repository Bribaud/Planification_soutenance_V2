import streamlit as st
from datetime import datetime, timedelta
import pandas as pd
import plotly.express as px
import networkx as nx
from itertools import combinations
import numpy as np
from collections import defaultdict
import random
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


st.set_page_config(page_title="Planification Soutenances", layout="wide")
st.title("Planification Optimisée des Soutenances de Stage")

# Configuration des étapes
etapes = [
    "etudiants", "salles", "duree_soutenance", "co_jury",
    "dates", "disponibilites", "disponibilites_selection", "generation"
]

etapes_labels = {
    "etudiants": "Étape 1 : Étudiants",
    "salles": "Étape 2 : Salles",
    "duree_soutenance": "Étape 3 : Durée",
    "co_jury": "Étape 4 : Co-jurys",
    "dates": "Étape 5 : Dates",
    "disponibilites": "Étape 6 : Créneaux",
    "disponibilites_selection": "Étape 7 : Disponibilités",
    "generation": "Étape 8 : Planning"
}


def afficher_navigation():
    st.sidebar.markdown("### 🧭 Navigation")
    etape_selectionnee = st.sidebar.selectbox(
        "Aller à une autre étape :",
        options=etapes,
        format_func=lambda x: etapes_labels.get(x, x),
        index=etapes.index(st.session_state.etape) if st.session_state.etape in etapes else 0
    )
    if etape_selectionnee != st.session_state.etape:
        st.session_state.etape = etape_selectionnee
        st.rerun()


# Initialisation des variables de session
if "etape" not in st.session_state:
    st.session_state.etape = "etudiants"
if "etudiants" not in st.session_state:
    st.session_state.etudiants = []
if "co_jurys" not in st.session_state:
    st.session_state.co_jurys = []
if "dates_soutenance" not in st.session_state:
    st.session_state.dates_soutenance = []
if "disponibilites" not in st.session_state:
    st.session_state.disponibilites = {}
if "planning_final" not in st.session_state:
    st.session_state.planning_final = []
if "nb_salles" not in st.session_state:
    st.session_state.nb_salles = 2
if "duree_soutenance" not in st.session_state:
    st.session_state.duree_soutenance = 50


@dataclass
class Individu:
    """Représente un individu (solution) dans l'algorithme génétique"""
    genes: List[int]  # Pour chaque étudiant, l'index du créneau assigné (-1 si non assigné)
    fitness: float = 0.0
    soutenances_planifiees: int = 0
    conflits: int = 0


class AlgorithmeGenetique:
    def __init__(self, planificateur, taille_population=80, nb_generations=800,
                 taux_mutation=0.12, taux_croisement=0.8):
        self.planificateur = planificateur
        self.taille_population = taille_population
        self.nb_generations = nb_generations
        self.taux_mutation = taux_mutation
        self.taux_croisement = taux_croisement

        # Pré-calculer les créneaux valides pour chaque étudiant
        self.creneaux = planificateur.generer_creneaux_uniques()
        self.creneaux_valides_par_etudiant = self._precalculer_creneaux_valides()
        self.nb_etudiants = len(planificateur.etudiants)

        # Statistiques
        self.historique_fitness = []
        self.meilleure_solution = None

    def _precalculer_creneaux_valides(self):
        """Pré-calcule les créneaux valides pour chaque étudiant"""
        creneaux_valides = {}

        for idx_etu, etudiant in enumerate(self.planificateur.etudiants):
            tuteur = etudiant["Tuteur"]
            creneaux_possibles = []

            for idx_creneau, creneau in enumerate(self.creneaux):
                if self.planificateur.est_disponible(tuteur, creneau['jour'], creneau['heure']):
                    # Vérifier qu'il y a au moins un co-jury disponible
                    co_jurys_disponibles = self.planificateur.trouver_co_jurys_disponibles(
                        tuteur, creneau['jour'], creneau['heure']
                    )
                    if co_jurys_disponibles:
                        creneaux_possibles.append(idx_creneau)

            creneaux_valides[idx_etu] = creneaux_possibles

        return creneaux_valides

    def generer_individu_intelligent(self) -> Individu:
        """
        Génère une solution initiale valide en respectant toutes les contraintes,
        avec une optimisation pour regrouper les soutenances des mêmes tuteurs.

        Returns:
            Individu: Une solution initiale valide avec des gènes optimisés
        """
        # Initialisation des structures de données
        genes = [-1] * len(self.planificateur.etudiants)  # -1 = non planifié
        creneaux_occupes = set()  # Ensemble des créneaux déjà utilisés
        jurys_par_moment = defaultdict(set)  # Dictionnaire des jurys occupés par créneau

        # 1. ORGANISATION DES ÉTUDIANTS PAR TUTEUR
        etudiants_par_tuteur = defaultdict(list)
        for idx_etu, etudiant in enumerate(self.planificateur.etudiants):
            tuteur = etudiant["Tuteur"]
            etudiants_par_tuteur[tuteur].append(idx_etu)

        # 2. ORDRE DE TRAITEMENT DES TUTEURS (ceux avec le plus d'étudiants en premier)
        tuteurs_ordonnes = sorted(
            etudiants_par_tuteur.keys(),
            key=lambda x: len(etudiants_par_tuteur[x]),
            reverse=True
        )

        # 3. STRATÉGIE DE REMPLISSAGE OPTIMISÉ
        for tuteur in tuteurs_ordonnes:
            # 3.1. Préparation des créneaux possibles pour ce tuteur
            creneaux_disponibles = []

            for idx_etu in etudiants_par_tuteur[tuteur]:
                # Récupérer les créneaux valides pré-calculés pour cet étudiant
                for idx_creneau in self.creneaux_valides_par_etudiant[idx_etu]:
                    creneau = self.creneaux[idx_creneau]
                    creneaux_disponibles.append((
                        idx_etu,
                        idx_creneau,
                        creneau['datetime_debut'],
                        creneau['moment']
                    ))

            # 3.2. Tri des créneaux par ordre chronologique
            creneaux_disponibles.sort(key=lambda x: x[2])  # Tri par datetime

            # 3.3. Assignation intelligente des créneaux
            for idx_etu, idx_creneau, _, moment in creneaux_disponibles:
                # Vérifier si l'étudiant est déjà planifié
                if genes[idx_etu] != -1:
                    continue

                # Vérifier si le créneau est déjà occupé
                if idx_creneau in creneaux_occupes:
                    continue

                # Vérifier si le tuteur est déjà occupé à ce moment
                if tuteur in jurys_par_moment[moment]:
                    continue

                # 3.4. Trouver un co-jury disponible
                creneau = self.creneaux[idx_creneau]
                co_jurys_disponibles = self.planificateur.trouver_co_jurys_disponibles(
                    tuteur, creneau['jour'], creneau['heure']
                )

                # Filtrer les co-jurys déjà occupés à ce moment
                co_jurys_libres = [
                    cj for cj in co_jurys_disponibles
                    if cj not in jurys_par_moment[moment]
                ]

                # Si co-jury disponible, assigner le créneau
                if co_jurys_libres:
                    # Choix du co-jury le moins chargé
                    co_jury_choisi = min(
                        co_jurys_libres,
                        key=lambda x: self.planificateur.charge_jurys[x]
                    )

                    # Mise à jour des structures
                    genes[idx_etu] = idx_creneau
                    creneaux_occupes.add(idx_creneau)
                    jurys_par_moment[moment].add(tuteur)
                    jurys_par_moment[moment].add(co_jury_choisi)

                    # Mise à jour des charges
                    self.planificateur.charge_jurys[tuteur] += 1
                    self.planificateur.charge_jurys[co_jury_choisi] += 1

        # 4. GESTION DES ÉTUDIANTS NON PLANIFIÉS (2ème passe)
        for idx_etu in range(len(genes)):
            if genes[idx_etu] == -1:  # Étudiant non planifié
                etudiant = self.planificateur.etudiants[idx_etu]
                tuteur = etudiant["Tuteur"]

                # Essayer tous les créneaux valides dans l'ordre aléatoire
                creneaux_possibles = self.creneaux_valides_par_etudiant[idx_etu].copy()
                random.shuffle(creneaux_possibles)

                for idx_creneau in creneaux_possibles:
                    if idx_creneau in creneaux_occupes:
                        continue

                    creneau = self.creneaux[idx_creneau]
                    moment = creneau['moment']

                    # Vérifier disponibilité tuteur
                    if (tuteur in jurys_par_moment[moment] or
                            not self.planificateur.est_disponible(tuteur, creneau['jour'], creneau['heure'])):
                        continue

                    # Trouver co-jury disponible
                    co_jurys_disponibles = self.planificateur.trouver_co_jurys_disponibles(
                        tuteur, creneau['jour'], creneau['heure']
                    )
                    co_jurys_libres = [
                        cj for cj in co_jurys_disponibles
                        if cj not in jurys_par_moment[moment]
                    ]

                    if co_jurys_libres:
                        # Prendre le premier disponible (aléatoire car shuffle avant)
                        co_jury_choisi = co_jurys_libres[0]

                        # Assignation finale
                        genes[idx_etu] = idx_creneau
                        creneaux_occupes.add(idx_creneau)
                        jurys_par_moment[moment].add(tuteur)
                        jurys_par_moment[moment].add(co_jury_choisi)
                        break

        return Individu(genes=genes)

    def calculer_fitness_amelioree(self, individu: Individu) -> Individu:
        """Fonction de fitness complète et détaillée évaluant la qualité d'un planning"""
        planning = self.decoder_individu(individu)

        # 1. CRITÈRES PRINCIPAUX DE PLANIFICATION
        nb_soutenances = len(planning)
        nb_total = len(self.planificateur.etudiants)
        taux_planification = nb_soutenances / nb_total if nb_total > 0 else 0

        # 2. ANALYSE DES CONFLITS
        conflits_salle, conflits_jury = self._analyser_conflits_detailles(planning)
        total_conflits = conflits_salle + conflits_jury

        # 3. ÉQUILIBRAGE DES CHARGES DES JURYS
        equilibrage = self._calculer_equilibrage_charge(planning)

        # 4. ALTERNANCE MATIN/APRÈS-MIDI
        bonus_alternance = self._calculer_bonus_alternance(planning)

        # 5. OPTIMISATION DU TEMPS DES TUTEURS (NOUVEAU CRITÈRE)
        optimisation_temps = self._calculer_optimisation_temps_tuteurs(planning)

        # 6. BONUS POUR UTILISATION OPTIMALE DES CRÉNEAUX
        utilisation_creneaux = self._calculer_utilisation_creneaux(planning)

        # CALCUL DU SCORE GLOBAL (À MAXIMISER)
        fitness = (
            # Priorité absolue : planifier le maximum de soutenances
                taux_planification * 2000 +

                # Bonus progressif pour les solutions avec beaucoup de soutenances
                max(0, (nb_soutenances - 20)) * 50 +

                # Équilibrage entre les jurys (éviter les surcharges)
                equilibrage * 30 +

                # Alternance entre matin/après-midi pour les jurys
                bonus_alternance * 15 +

                # Optimisation du temps des tuteurs (regroupement des créneaux)
                optimisation_temps * 40 +

                # Utilisation optimale des créneaux (éviter les trous dans l'emploi du temps)
                utilisation_creneaux * 25 -

                # Pénalités massives pour les conflits
                total_conflits * 1000 -

                # Pénalité pour étudiants non planifiés
                (nb_total - nb_soutenances) * 200
        )

        # BONUS SPÉCIAUX
        # Solution parfaite (100% de planification sans conflits)
        if total_conflits == 0 and nb_soutenances == nb_total:
            fitness += 5000

        # Solution très propre (peu de conflits et haut taux)
        elif total_conflits <= 2 and taux_planification >= 0.95:
            fitness += 3000

        # Solution acceptable mais perfectible
        elif total_conflits <= 5 and taux_planification >= 0.85:
            fitness += 1000

        # Stockage des métriques dans l'individu
        individu.fitness = fitness
        individu.soutenances_planifiees = nb_soutenances
        individu.conflits = total_conflits

        return individu

    def _calculer_optimisation_temps_tuteurs(self, planning):
        """Calcule un score basé sur le regroupement des soutenances par tuteur"""
        if not planning:
            return 0

        soutenances_par_tuteur = defaultdict(list)
        for soutenance in planning:
            soutenances_par_tuteur[soutenance['Tuteur']].append(soutenance)
            soutenances_par_tuteur[soutenance['Co-jury']].append(soutenance)

        score_total = 0

        for tuteur, soutenances in soutenances_par_tuteur.items():
            if len(soutenances) < 2:
                continue  # Pas d'optimisation possible avec une seule soutenance

            # Trier chronologiquement
            soutenances.sort(key=lambda x: x['Début'])
            temps_morts = 0
            sequences = 0

            for i in range(1, len(soutenances)):
                delta = (soutenances[i]['Début'] - soutenances[i - 1]['Fin']).total_seconds() / 60

                # Pénalité progressive selon l'écart
                if delta > 120:  # +2h entre 2 soutenances = très mauvais
                    temps_morts += 100
                elif delta > 60:  # 1h-2h = mauvais
                    temps_morts += 50
                elif delta > 30:  # 30min-1h = acceptable
                    temps_morts += 20
                elif delta > 0:  # <30min = bon
                    temps_morts += 5
                else:  # Enchaînement parfait
                    sequences += 1

            # Score pour ce tuteur (0-100)
            if len(soutenances) > 1:
                score_tuteur = (
                        sequences * 50 -  # Bonus pour enchaînements directs
                        temps_morts / (len(soutenances) - 1)  # Pénalité pour temps morts
                )
                score_total += max(0, min(100, score_tuteur))  # Plafonné à 100

        # Normalisation (moyenne des scores)
        return score_total / len(soutenances_par_tuteur) if soutenances_par_tuteur else 0

    def _calculer_utilisation_creneaux(self, planning):
        """Évalue l'utilisation optimale des créneaux horaires"""
        if not planning:
            return 0

        # Compter le nombre de créneaux utilisés par jour/salle
        creneaux_utilises = defaultdict(int)
        creneaux_possibles = 0

        for soutenance in planning:
            cle = (soutenance['Jour'], soutenance['Salle'], soutenance['Créneau'])
            creneaux_utilises[cle] += 1

        # Calculer le taux d'occupation
        jours_uniques = set(soutenance['Jour'] for soutenance in planning)
        salles_uniques = set(soutenance['Salle'] for soutenance in planning)

        for jour in jours_uniques:
            for salle in salles_uniques:
                # Compter les créneaux possibles pour ce jour/salle
                creneaux_possibles += sum(
                    1 for c in self.creneaux
                    if c['jour'] == jour and c['salle'] == salle
                )

        if creneaux_possibles == 0:
            return 0

        taux_occupation = len(creneaux_utilises) / creneaux_possibles

        # Bonus si taux > 70%
        return 100 * max(0, taux_occupation - 0.7) * 3  # Coefficient pour importance relative

    def _analyser_conflits_detailles(self, planning):
        """Analyse détaillée des conflits"""
        conflits_salle = 0
        conflits_jury = 0

        creneaux_salle = {}
        jurys_par_moment = {}

        for soutenance in planning:
            cle_salle = f"{soutenance['Jour']}_{soutenance['Créneau']}_{soutenance['Salle']}"
            moment = f"{soutenance['Jour']}_{soutenance['Créneau']}"

            # Conflits de salle
            if cle_salle in creneaux_salle:
                conflits_salle += 1
            creneaux_salle[cle_salle] = True

            # Conflits de jurys
            if moment not in jurys_par_moment:
                jurys_par_moment[moment] = set()

            tuteur = soutenance['Tuteur']
            co_jury = soutenance['Co-jury']

            if tuteur in jurys_par_moment[moment]:
                conflits_jury += 1
            if co_jury in jurys_par_moment[moment]:
                conflits_jury += 1

            jurys_par_moment[moment].add(tuteur)
            jurys_par_moment[moment].add(co_jury)

        return conflits_salle, conflits_jury

    def _calculer_equilibrage_charge(self, planning):
        """Calcule un score d'équilibrage des charges"""
        if not planning:
            return 0

        charges = {}
        for soutenance in planning:
            tuteur = soutenance['Tuteur']
            co_jury = soutenance['Co-jury']
            charges[tuteur] = charges.get(tuteur, 0) + 1
            charges[co_jury] = charges.get(co_jury, 0) + 1

        if len(charges) <= 1:
            return 0

        valeurs_charges = list(charges.values())
        moyenne = sum(valeurs_charges) / len(valeurs_charges)
        variance = sum((x - moyenne) ** 2 for x in valeurs_charges) / len(valeurs_charges)

        # Score inversement proportionnel à la variance
        return max(0, 10 - variance)

    def _calculer_bonus_alternance(self, planning):
        """Calcule un bonus pour l'alternance des créneaux des jurys"""
        bonus = 0
        jurys_par_periode = {'matin': set(), 'apres_midi': set()}

        for soutenance in planning:
            # Déterminer si c'est matin ou après-midi
            debut = soutenance['Début']
            periode = 'matin' if debut.hour < 14 else 'apres_midi'

            jurys_par_periode[periode].add(soutenance['Tuteur'])
            jurys_par_periode[periode].add(soutenance['Co-jury'])

        # Bonus si les jurys alternent entre matin et après-midi
        jurys_equilibres = jurys_par_periode['matin'] & jurys_par_periode['apres_midi']
        bonus += len(jurys_equilibres) * 2

        return bonus

    def croisement_intelligent(self, parent1: Individu, parent2: Individu) -> Tuple[Individu, Individu]:
        """Croisement qui préserve la validité des solutions"""
        enfant1_genes = [-1] * len(parent1.genes)
        enfant2_genes = [-1] * len(parent2.genes)

        # Sélectionner aléatoirement des segments à échanger
        point_croisement = random.randint(1, len(parent1.genes) - 1)

        # Copier la première partie
        for i in range(point_croisement):
            enfant1_genes[i] = parent1.genes[i]
            enfant2_genes[i] = parent2.genes[i]

        # Pour la deuxième partie, vérifier les conflits
        creneaux_utilises_e1 = set(g for g in enfant1_genes[:point_croisement] if g != -1)
        creneaux_utilises_e2 = set(g for g in enfant2_genes[:point_croisement] if g != -1)

        for i in range(point_croisement, len(parent1.genes)):
            # Pour enfant1, prendre de parent2 si pas de conflit
            if parent2.genes[i] != -1 and parent2.genes[i] not in creneaux_utilises_e1:
                enfant1_genes[i] = parent2.genes[i]
                creneaux_utilises_e1.add(parent2.genes[i])

            # Pour enfant2, prendre de parent1 si pas de conflit
            if parent1.genes[i] != -1 and parent1.genes[i] not in creneaux_utilises_e2:
                enfant2_genes[i] = parent1.genes[i]
                creneaux_utilises_e2.add(parent1.genes[i])

        return Individu(genes=enfant1_genes), Individu(genes=enfant2_genes)

    def mutation_adaptative(self, individu: Individu) -> Individu:
        """Mutation qui améliore les solutions existantes"""
        # Pour chaque gène (étudiant)
        for i in range(len(individu.genes)):
            if random.random() < self.taux_mutation:
                # Si l'étudiant n'est pas planifié, essayer de le planifier
                if individu.genes[i] == -1:
                    creneaux_possibles = self.creneaux_valides_par_etudiant[i]
                    if creneaux_possibles:
                        # Choisir un créneau qui optimise le temps du tuteur
                        tuteur = self.planificateur.etudiants[i]["Tuteur"]
                        meilleur_creneau = self._trouver_meilleur_creneau_tuteur(tuteur, creneaux_possibles,
                                                                                 individu.genes)
                        if meilleur_creneau is not None:
                            individu.genes[i] = meilleur_creneau

                # Sinon, essayer de changer vers un meilleur créneau
                else:
                    creneaux_possibles = self.creneaux_valides_par_etudiant[i]
                    if creneaux_possibles:
                        tuteur = self.planificateur.etudiants[i]["Tuteur"]
                        meilleur_creneau = self._trouver_meilleur_creneau_tuteur(tuteur, creneaux_possibles,
                                                                                 individu.genes)
                        if meilleur_creneau is not None and meilleur_creneau != individu.genes[i]:
                            individu.genes[i] = meilleur_creneau

        return individu

    def _trouver_meilleur_creneau_tuteur(self, tuteur, creneaux_possibles, genes):
        """Trouve le meilleur créneau pour optimiser l'emploi du temps du tuteur"""
        # Trouver les créneaux déjà assignés à ce tuteur
        creneaux_tuteur = []
        for idx_etu, idx_creneau in enumerate(genes):
            if idx_creneau != -1 and self.planificateur.etudiants[idx_etu]["Tuteur"] == tuteur:
                creneaux_tuteur.append(self.creneaux[idx_creneau])

        if not creneaux_tuteur:
            # Si le tuteur n'a pas encore de créneau, choisir aléatoirement
            return random.choice(creneaux_possibles) if creneaux_possibles else None

        # Trier les créneaux existants du tuteur
        creneaux_tuteur.sort(key=lambda x: x['datetime_debut'])

        # Évaluer chaque créneau possible
        scores = []
        for idx_creneau in creneaux_possibles:
            creneau = self.creneaux[idx_creneau]
            score = 0

            # Vérifier si le créneau est adjacent à un créneau existant
            for ct in creneaux_tuteur:
                delta_avant = (creneau['datetime_debut'] - ct['datetime_fin']).total_seconds() / 60
                delta_apres = (ct['datetime_debut'] - creneau['datetime_fin']).total_seconds() / 60

                if 0 < delta_avant <= 30 or 0 < delta_apres <= 30:
                    score += 100  # Bonus pour créneau adjacent
                elif delta_avant > 0:
                    score += max(0, 100 - delta_avant)  # Moins bon si éloigné
                elif delta_apres > 0:
                    score += max(0, 100 - delta_apres)  # Moins bon si éloigné

            scores.append((idx_creneau, score / len(creneaux_tuteur)))

        # Choisir le créneau avec le meilleur score
        if scores:
            scores.sort(key=lambda x: x[1], reverse=True)
            return scores[0][0]

        return None

    def evoluer(self) -> Tuple[List[Dict], Dict]:
        """Algorithme génétique principal amélioré"""
        # Population initiale avec génération intelligente
        population = []
        for _ in range(self.taille_population):
            individu = self.generer_individu_intelligent()
            population.append(self.calculer_fitness_amelioree(individu))

        # Suivi du meilleur
        self.meilleure_solution = max(population, key=lambda x: x.fitness)
        stagnation = 0

        for generation in range(self.nb_generations):
            nouvelle_population = []

            # Élitisme renforcé
            population_triee = sorted(population, key=lambda x: x.fitness, reverse=True)
            elite_size = max(5, self.taille_population // 8)
            nouvelle_population.extend(population_triee[:elite_size])

            # Génération avec adaptation
            while len(nouvelle_population) < self.taille_population:
                if random.random() < self.taux_croisement:
                    # Sélection avec biais vers les meilleurs
                    parent1 = self.selection_tournament(population, k=5)
                    parent2 = self.selection_tournament(population, k=5)

                    enfant1, enfant2 = self.croisement_intelligent(parent1, parent2)

                    # Mutation adaptative
                    enfant1 = self.mutation_adaptative(enfant1)
                    enfant2 = self.mutation_adaptative(enfant2)

                    nouvelle_population.extend([enfant1, enfant2])
                else:
                    # Immigration : nouveaux individus intelligents
                    nouvel_individu = self.generer_individu_intelligent()
                    nouvelle_population.append(nouvel_individu)

            # Limitation et évaluation
            nouvelle_population = nouvelle_population[:self.taille_population]
            population = [self.calculer_fitness_amelioree(ind) for ind in nouvelle_population]

            # Mise à jour du meilleur
            meilleur_actuel = max(population, key=lambda x: x.fitness)
            if meilleur_actuel.fitness > self.meilleure_solution.fitness:
                self.meilleure_solution = meilleur_actuel
                stagnation = 0
            else:
                stagnation += 1

            # Relance si stagnation
            if stagnation > 50 and generation < self.nb_generations - 100:
                # Remplacer 30% de la population par de nouveaux individus
                nb_nouveaux = self.taille_population // 3
                nouveaux = [self.generer_individu_intelligent() for _ in range(nb_nouveaux)]
                population = population[:-nb_nouveaux] + nouveaux
                stagnation = 0

            # Statistiques
            fitness_moyenne = sum(ind.fitness for ind in population) / len(population)
            self.historique_fitness.append({
                'generation': generation,
                'fitness_max': meilleur_actuel.fitness,
                'fitness_moyenne': fitness_moyenne,
                'soutenances_max': meilleur_actuel.soutenances_planifiees,
                'conflits_min': meilleur_actuel.conflits
            })

        # Résultats finaux
        planning_final = self.decoder_individu(self.meilleure_solution)

        statistiques = {
            'generations': self.nb_generations,
            'fitness_finale': self.meilleure_solution.fitness,
            'soutenances_planifiees': self.meilleure_solution.soutenances_planifiees,
            'conflits': self.meilleure_solution.conflits,
            'taux_reussite': self.meilleure_solution.soutenances_planifiees / len(self.planificateur.etudiants),
            'historique': self.historique_fitness
        }

        return planning_final, statistiques

    def selection_tournament(self, population: List[Individu], k=3) -> Individu:
        """Sélection par tournoi"""
        participants = random.sample(population, min(k, len(population)))
        return max(participants, key=lambda x: x.fitness)

    def decoder_individu(self, individu: Individu) -> List[Dict]:
        """Décode un individu en planning détaillé"""
        planning = []

        for idx_etu, idx_creneau in enumerate(individu.genes):
            if idx_creneau == -1:
                continue

            etudiant = self.planificateur.etudiants[idx_etu]
            creneau = self.creneaux[idx_creneau]
            tuteur = etudiant["Tuteur"]

            # Trouver le meilleur co-jury disponible
            co_jurys_disponibles = self.planificateur.trouver_co_jurys_disponibles(
                tuteur, creneau['jour'], creneau['heure']
            )

            if co_jurys_disponibles:
                planning.append({
                    "Étudiant": f"{etudiant['Prénom']} {etudiant['Nom']}",
                    "Pays": etudiant['Pays'],
                    "Tuteur": tuteur,
                    "Co-jury": co_jurys_disponibles[0],
                    "Jour": creneau['jour'],
                    "Créneau": creneau['heure'],
                    "Salle": creneau['salle'],
                    "Début": creneau['datetime_debut'],
                    "Fin": creneau['datetime_fin']
                })

        return planning


class PlanificationOptimiseeV2:
    def __init__(self, etudiants, co_jurys, dates, disponibilites, nb_salles, duree):
        self.etudiants = etudiants
        self.co_jurys = co_jurys
        self.dates = dates
        self.disponibilites = disponibilites
        self.nb_salles = nb_salles
        self.duree = duree

        # Créer la liste complète des jurys
        self.tuteurs_referents = list(set([e["Tuteur"] for e in etudiants]))
        self.tous_jurys = list(set(self.tuteurs_referents + co_jurys))

        # Statistiques de charge pour équilibrer
        self.charge_jurys = {jury: 0 for jury in self.tous_jurys}

    def generer_creneaux_uniques(self):
        """Génère tous les créneaux possibles avec identifiants uniques"""
        creneaux = []
        creneau_id = 0

        for jour in self.dates:
            jour_str = jour.strftime("%A %d/%m/%Y")
            # Créneaux matin et après-midi
            for periode in [("08:00", "13:00"), ("14:00", "17:20")]:
                debut, fin = periode
                current = datetime.combine(jour, datetime.strptime(debut, "%H:%M").time())
                end = datetime.combine(jour, datetime.strptime(fin, "%H:%M").time())

                while current + timedelta(minutes=self.duree) <= end:
                    fin_creneau = current + timedelta(minutes=self.duree)
                    heure_str = f"{current.strftime('%H:%M')} - {fin_creneau.strftime('%H:%M')}"

                    # Créer un créneau pour chaque salle
                    for salle in range(1, self.nb_salles + 1):
                        creneaux.append({
                            'id': creneau_id,
                            'jour': jour_str,
                            'heure': heure_str,
                            'salle': f"Salle {salle}",
                            'datetime_debut': current,
                            'datetime_fin': fin_creneau,
                            'moment': f"{jour_str}_{heure_str}"  # Clé pour identifier le moment (sans salle)
                        })
                        creneau_id += 1

                    current = fin_creneau
        return creneaux

    def _calculer_temps_morts_jury(self, planning_jury):
        """
        Calcule les temps morts pour un jury selon les nouvelles règles :
        - Exclut la pause déjeuner (13h-14h)
        - Agrège sur plusieurs jours
        - Retourne un tuple (heures, minutes)
        """
        if not planning_jury:
            return (0, 0)

        # Trier les soutenances par date/heure
        planning_trie = sorted(planning_jury, key=lambda x: x['Début'])

        temps_mort_total_min = 0

        for i in range(1, len(planning_trie)):
            fin_precedente = planning_trie[i - 1]['Fin']
            debut_suivant = planning_trie[i]['Début']

            # Vérifier si c'est le même jour
            if fin_precedente.date() != debut_suivant.date():
                continue  # Traiter chaque jour séparément

            delta_min = (debut_suivant - fin_precedente).total_seconds() / 60

            # Exclure la pause déjeuner (13h-14h)
            heure_fin = fin_precedente.hour + fin_precedente.minute / 60
            heure_debut = debut_suivant.hour + debut_suivant.minute / 60

            if heure_fin <= 13 and heure_debut >= 14:
                # La pause déjeuner est incluse dans l'intervalle
                delta_min -= 60  # On retire 1h de pause déjeuner
                delta_min = max(0, delta_min)  # Éviter les valeurs négatives

            # Ajouter seulement si > 0 (éviter les chevauchements)
            if delta_min > 0:
                temps_mort_total_min += delta_min

        # Conversion en heures et minutes
        heures = int(temps_mort_total_min // 60)
        minutes = int(temps_mort_total_min % 60)

        return (heures, minutes)

    def _calculer_temps_morts_global(self, planning):
        """
        Calcule les temps morts pour tous les jurys et retourne :
        - Un dictionnaire {jury: (heures, minutes)}
        - Le total global (heures, minutes)
        """
        if not planning:
            return {}, (0, 0)

        # Organiser les soutenances par jury
        emploi_du_temps = defaultdict(list)
        for soutenance in planning:
            emploi_du_temps[soutenance['Tuteur']].append(soutenance)
            emploi_du_temps[soutenance['Co-jury']].append(soutenance)

        # Calcul pour chaque jury
        temps_morts = {}
        total_heures, total_minutes = 0, 0

        for jury, sessions in emploi_du_temps.items():
            heures, minutes = self._calculer_temps_morts_jury(sessions)
            temps_morts[jury] = (heures, minutes)
            total_heures += heures
            total_minutes += minutes

        # Normalisation du total (convertir 60+ minutes en heures)
        total_heures += total_minutes // 60
        total_minutes = total_minutes % 60

        return temps_morts, (total_heures, total_minutes)

    def _calculer_score_global(self, planning, temps_morts):
        """
        Calcule un score global de qualité du planning (0-100) en combinant:
        - Taux de planification
        - Équilibrage des charges
        - Optimisation du temps des jurys
        - Utilisation des ressources
        """
        if not planning:
            return 0

        # 1. Taux de planification (40% du score)
        nb_planifies = len(planning)
        nb_total = len(self.etudiants)
        score_planification = (nb_planifies / nb_total) * 40 if nb_total > 0 else 0

        # 2. Équilibrage des charges (25% du score)
        charge_jurys = defaultdict(int)
        for soutenance in planning:
            charge_jurys[soutenance['Tuteur']] += 1
            charge_jurys[soutenance['Co-jury']] += 1

        charges = list(charge_jurys.values())
        if charges:
            ecart_type = np.std(charges)
            # Plus l'écart-type est faible, meilleur est le score
            score_equilibre = max(0, 25 - (ecart_type * 2))  # Pénalité pour déséquilibre
        else:
            score_equilibre = 0

        # 3. Optimisation du temps des jurys (25% du score)
        if temps_morts:
            total_tm = sum(sum(tm) for tm in temps_morts.values())
            nb_jurys = len(temps_morts)
            tm_moyen = total_tm / nb_jurys

            # Moins il y a de temps morts, meilleur est le score
            score_temps = max(0, 25 - (tm_moyen * 0.2))  # 5 min de temps mort = -1 point
        else:
            score_temps = 25

        # 4. Utilisation des ressources (10% du score)
        df_planning = pd.DataFrame(planning)
        salles_utilisees = len(df_planning['Salle'].unique())
        score_ressources = min(10, salles_utilisees / self.nb_salles * 10)

        # Score final (0-100)
        score_total = (
                score_planification +
                score_equilibre +
                score_temps +
                score_ressources
        )

        return min(100, max(0, score_total))  # Assurance 0-100

    def est_disponible(self, personne, jour, heure):
        """Vérifie si une personne est disponible à un créneau donné"""
        key = f"{jour} | {heure}"
        return self.disponibilites.get(personne, {}).get(key, False)

    def trouver_co_jurys_disponibles(self, tuteur_referent, jour, heure):
        """Trouve tous les co-jurys disponibles pour un créneau, en excluant le tuteur référent"""
        co_jurys_disponibles = []

        for jury in self.tous_jurys:
            if jury != tuteur_referent and self.est_disponible(jury, jour, heure):
                co_jurys_disponibles.append(jury)

        # Trier par charge croissante pour équilibrer
        co_jurys_disponibles.sort(key=lambda x: self.charge_jurys[x])
        return co_jurys_disponibles

    def optimiser_planning_ameliore(self):
        """Algorithme d'optimisation amélioré avec gestion des conflits"""
        creneaux = self.generer_creneaux_uniques()
        planning = []

        # Structures pour suivre les allocations
        creneaux_occupes = set()  # IDs des créneaux occupés
        jurys_par_moment = defaultdict(set)  # moment -> set de jurys occupés

        # Mélanger les étudiants pour éviter les biais d'ordre
        etudiants_melanges = self.etudiants.copy()
        random.shuffle(etudiants_melanges)

        # Statistiques
        tentatives_par_etudiant = []

        for idx_etu, etudiant in enumerate(etudiants_melanges):
            tuteur_referent = etudiant["Tuteur"]
            soutenance_planifiee = False
            tentatives = 0

            # Mélanger les créneaux pour éviter les patterns
            creneaux_melanges = creneaux.copy()
            random.shuffle(creneaux_melanges)

            for creneau in creneaux_melanges:
                tentatives += 1

                # Vérifier si le créneau est déjà occupé
                if creneau['id'] in creneaux_occupes:
                    continue

                # Vérifier la disponibilité du tuteur référent
                if not self.est_disponible(tuteur_referent, creneau['jour'], creneau['heure']):
                    continue

                # Vérifier si le tuteur référent est déjà occupé à ce moment
                if tuteur_referent in jurys_par_moment[creneau['moment']]:
                    continue

                # Chercher un co-jury disponible
                co_jurys_possibles = self.trouver_co_jurys_disponibles(
                    tuteur_referent, creneau['jour'], creneau['heure']
                )

                # Filtrer les co-jurys déjà occupés à ce moment
                co_jurys_libres = [
                    cj for cj in co_jurys_possibles
                    if cj not in jurys_par_moment[creneau['moment']]
                ]

                if co_jurys_libres:
                    # Choisir le co-jury avec la charge la plus faible
                    co_jury_choisi = co_jurys_libres[0]

                    # Planifier la soutenance
                    planning.append({
                        "Étudiant": f"{etudiant['Prénom']} {etudiant['Nom']}",
                        "Pays": etudiant['Pays'],
                        "Tuteur": tuteur_referent,
                        "Co-jury": co_jury_choisi,
                        "Jour": creneau['jour'],
                        "Créneau": creneau['heure'],
                        "Salle": creneau['salle'],
                        "Début": creneau['datetime_debut'],
                        "Fin": creneau['datetime_fin']
                    })

                    # Marquer comme occupé
                    creneaux_occupes.add(creneau['id'])
                    jurys_par_moment[creneau['moment']].add(tuteur_referent)
                    jurys_par_moment[creneau['moment']].add(co_jury_choisi)

                    # Mettre à jour les charges
                    self.charge_jurys[tuteur_referent] += 1
                    self.charge_jurys[co_jury_choisi] += 1

                    soutenance_planifiee = True
                    break

            tentatives_par_etudiant.append(tentatives)

            if not soutenance_planifiee:
                st.warning(
                    f"⚠️ Impossible de planifier {etudiant['Prénom']} {etudiant['Nom']} après {tentatives} tentatives")

        # Statistiques de diagnostic
        self.afficher_diagnostics(planning, tentatives_par_etudiant)

        return planning, len(self.etudiants) - len(planning)

    def optimiser_avec_genetique(self, utiliser_genetique=False, **params_genetique):
        """Méthode hybride : algorithme classique puis génétique si nécessaire"""

        # Essayer d'abord l'algorithme classique
        planning_classique, non_planifies = self.optimiser_planning_ameliore()

        taux_reussite = len(planning_classique) / len(self.etudiants)

        # Si le taux de réussite est insuffisant, utiliser l'algorithme génétique
        if utiliser_genetique or taux_reussite < 0.8:
            st.info("🧬 Lancement de l'optimisation génétique pour améliorer les résultats...")

            # Configuration par défaut de l'algorithme génétique
            config_genetique = {
                'taille_population': 50,
                'nb_generations': 100,
                'taux_mutation': 0.1,
                'taux_croisement': 0.8,
                **params_genetique
            }

            # Créer et lancer l'algorithme génétique
            ag = AlgorithmeGenetique(self, **config_genetique)
            planning_genetique, stats_genetique = ag.evoluer()

            # Comparer les résultats
            if len(planning_genetique) > len(planning_classique):
                st.success(f"✅ L'algorithme génétique a amélioré les résultats : "
                           f"{len(planning_genetique)} vs {len(planning_classique)} soutenances planifiées")
                return planning_genetique, len(self.etudiants) - len(planning_genetique), stats_genetique
            else:
                st.info("ℹ️ L'algorithme classique reste optimal")
                return planning_classique, non_planifies, None

        return planning_classique, non_planifies, None

    def afficher_diagnostics(self, planning, tentatives_par_etudiant):
        """
        Affiche un dashboard complet de diagnostics du planning généré
        avec calcul précis des temps morts selon les nouvelles règles.
        """
        st.subheader("📊 Diagnostic complet du planning")

        # =====================================================================
        # SECTION 1 : MÉTRIQUES GLOBALES
        # =====================================================================
        st.markdown("### 📈 Métriques globales")
        cols_metrics = st.columns(4)

        with cols_metrics[0]:
            nb_planifies = len(planning)
            nb_total = len(self.etudiants)
            taux_reussite = nb_planifies / nb_total if nb_total > 0 else 0
            st.metric("Soutenances planifiées", f"{nb_planifies}/{nb_total}",
                      help="Nombre d'étudiants planifiés sur le total")

        with cols_metrics[1]:
            st.metric("Taux de réussite", f"{taux_reussite:.1%}",
                      delta=f"{nb_planifies - nb_total}",
                      help="Pourcentage d'étudiants ayant reçu un créneau")

        with cols_metrics[2]:
            if tentatives_par_etudiant:
                avg_tentatives = np.mean(tentatives_par_etudiant)
                st.metric("Tentatives moyennes", f"{avg_tentatives:.1f}",
                          help="Nombre moyen de tentatives par étudiant")

        with cols_metrics[3]:
            conflits = self.verifier_conflits(planning)
            st.metric("Conflits détectés", len(conflits),
                      help="Nombre de chevauchements de jurys/salles")

        # =====================================================================
        # SECTION 2 : TEMPS MORTS (NOUVELLE IMPLÉMENTATION)
        # =====================================================================
        st.markdown("### ⏱ Optimisation des temps morts (hors pause déjeuner)")

        # Calcul avec la nouvelle méthode
        temps_morts_par_jury, (total_heures, total_min) = self._calculer_temps_morts_global(planning)

        if temps_morts_par_jury:
            # Affichage des KPI
            cols_tm = st.columns(3)

            with cols_tm[0]:
                st.metric("Temps morts totaux",
                          f"{total_heures}h{total_min:02d}",
                          help="Temps d'attente cumulé pour tous les jurys")

            with cols_tm[1]:
                # Calcul moyenne
                nb_jurys = len(temps_morts_par_jury)
                avg_total_min = (total_heures * 60 + total_min) / nb_jurys
                avg_h, avg_m = int(avg_total_min // 60), int(avg_total_min % 60)
                st.metric("Moyenne par jury",
                          f"{avg_h}h{avg_m:02d}",
                          help="Temps d'attente moyen par jury")

            with cols_tm[2]:
                # Jury le plus impacté
                jury_max = max(temps_morts_par_jury.items(),
                               key=lambda x: x[1][0] * 60 + x[1][1])
                st.metric("Jury le plus impacté",
                          f"{jury_max[1][0]}h{jury_max[1][1]:02d}",
                          help=f"{jury_max[0]}")

            # Graphique détaillé
            df_tm = pd.DataFrame([
                {
                    'Jury': jury,
                    'Heures': tm[0],
                    'Minutes': tm[1],
                    'Total (min)': tm[0] * 60 + tm[1],
                    'Formatté': f"{tm[0]}h{tm[1]:02d}"
                }
                for jury, tm in temps_morts_par_jury.items()
            ]).sort_values('Total (min)', ascending=False)

            fig_tm = px.bar(
                df_tm,
                x='Jury',
                y='Total (min)',
                hover_data=['Heures', 'Minutes'],
                text='Formatté',
                title='Temps morts cumulés par jury (en minutes)',
                labels={'Total (min)': 'Minutes perdues'},
                color='Total (min)',
                color_continuous_scale='thermal'
            )
            fig_tm.update_traces(textposition='outside')
            fig_tm.update_layout(xaxis_tickangle=-45, hovermode='x unified')
            st.plotly_chart(fig_tm, use_container_width=True)

            # Top 3 des jurys avec le plus de temps morts
            with st.expander("🔍 Détail des jurys les plus impactés"):
                top_jurys = sorted(temps_morts_par_jury.items(),
                                   key=lambda x: x[1][0] * 60 + x[1][1],
                                   reverse=True)[:3]

                for jury, (heures, minutes) in top_jurys:
                    st.markdown(f"#### 🕒 {jury} : {heures}h{minutes:02d}")

                    # Filtrer le planning pour ce jury
                    sessions = [s for s in planning
                                if s['Tuteur'] == jury or s['Co-jury'] == jury]
                    sessions_sorted = sorted(sessions, key=lambda x: x['Début'])

                    # Afficher l'emploi du temps
                    for i in range(len(sessions_sorted)):
                        if i > 0:
                            # Calcul du temps mort entre deux sessions
                            delta = (sessions_sorted[i]['Début'] - sessions_sorted[i - 1]['Fin'])
                            delta_min = delta.total_seconds() / 60

                            # Vérifier si c'est un temps mort à comptabiliser
                            heure_fin = sessions_sorted[i - 1]['Fin'].hour + sessions_sorted[i - 1]['Fin'].minute / 60
                            heure_debut = sessions_sorted[i]['Début'].hour + sessions_sorted[i]['Début'].minute / 60

                            if not (heure_fin <= 13 and heure_debut >= 14) and delta_min > 0:
                                st.warning(f"⏳ Temps mort: {int(delta_min // 60)}h{int(delta_min % 60):02d} "
                                           f"entre {sessions_sorted[i - 1]['Fin'].strftime('%H:%M')} "
                                           f"et {sessions_sorted[i]['Début'].strftime('%H:%M')}")

                        # Afficher la session
                        session = sessions_sorted[i]
                        role = "Tuteur" if session['Tuteur'] == jury else "Co-jury"
                        st.success(
                            f"{session['Début'].strftime('%H:%M')} → {session['Fin'].strftime('%H:%M')} : "
                            f"{session['Étudiant']} ({role}, Salle {session['Salle']})"
                        )

        # =====================================================================
        # SECTION 3 : CHARGE DES JURYS
        # =====================================================================
        st.markdown("### 👥 Charge des jurys")

        charge_jurys = defaultdict(int)
        for soutenance in planning:
            charge_jurys[soutenance['Tuteur']] += 1
            charge_jurys[soutenance['Co-jury']] += 1

        if charge_jurys:
            # KPI
            cols_charge = st.columns(3)
            charges = list(charge_jurys.values())

            with cols_charge[0]:
                st.metric("Charge moyenne", f"{np.mean(charges):.1f}",
                          help="Nombre moyen de soutenances par jury")

            with cols_charge[1]:
                st.metric("Charge maximale", max(charges),
                          help="Nombre max de soutenances pour un jury")

            with cols_charge[2]:
                st.metric("Écart-type", f"{np.std(charges):.1f}",
                          help="Écart dans la répartition des charges")

            # Graphique
            df_charge = pd.DataFrame({
                'Jury': list(charge_jurys.keys()),
                'Soutenances': list(charge_jurys.values())
            }).sort_values('Soutenances', ascending=False)

            fig_charge = px.bar(
                df_charge,
                x='Jury',
                y='Soutenances',
                title='Nombre de soutenances par jury',
                labels={'Soutenances': 'Nb de soutenances'},
                color='Soutenances',
                text='Soutenances'
            )
            fig_charge.update_traces(textposition='outside')
            fig_charge.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_charge, use_container_width=True)

        # =====================================================================
        # SECTION 4 : UTILISATION DES RESSOURCES
        # =====================================================================
        st.markdown("### 🏫 Utilisation des ressources")

        if planning:
            df = pd.DataFrame(planning)

            # Par salle
            st.markdown("#### Par salle")
            salle_usage = df.groupby(['Jour', 'Salle']).size().reset_index(name='Count')
            fig_salles = px.bar(
                salle_usage,
                x='Jour',
                y='Count',
                color='Salle',
                barmode='group',
                title='Utilisation des salles par jour',
                labels={'Count': 'Nb de soutenances'}
            )
            st.plotly_chart(fig_salles, use_container_width=True)

            # Par période
            st.markdown("#### Par période")
            df['Période'] = df['Début'].apply(
                lambda x: 'Matin (8h-12h)' if x.hour < 12 else
                'Après-midi (13h-17h)' if x.hour < 17 else 'Soir'
            )
            period_usage = df.groupby('Période').size().reset_index(name='Count')
            fig_period = px.pie(
                period_usage,
                values='Count',
                names='Période',
                title='Répartition par période de la journée',
                hole=0.3
            )
            st.plotly_chart(fig_period, use_container_width=True)

        # =====================================================================
        # SECTION 5 : CONFLITS
        # =====================================================================
        conflits = self.verifier_conflits(planning)
        if conflits:
            st.markdown("### ⚠️ Conflits détectés")
            for conflit in conflits:
                st.error(conflit, icon="🚨")
        else:
            st.success("✅ Aucun conflit détecté dans le planning", icon="✅")

        # Score global
        if planning:
            score = self._calculer_score_global(planning, temps_morts_par_jury)
            st.markdown(f"### 📊 Score global d'optimisation : {score:.1f}/100")
            st.progress(score / 100)

            # Légende du score
            cols_score = st.columns(3)
            with cols_score[0]:
                st.markdown("**Composition du score :**")
                st.markdown("- Taux de planification (40%)")
                st.markdown("- Équilibrage charges (25%)")
                st.markdown("- Temps morts (25%)")
                st.markdown("- Utilisation ressources (10%)")

    def verifier_conflits(self, planning):
        """Vérifie s'il y a des conflits dans le planning"""
        conflits = []

        # Grouper par moment et salle
        creneaux_utilises = defaultdict(list)
        jurys_par_moment = defaultdict(list)

        for idx, soutenance in enumerate(planning):
            moment_salle = f"{soutenance['Jour']}_{soutenance['Créneau']}_{soutenance['Salle']}"
            moment = f"{soutenance['Jour']}_{soutenance['Créneau']}"

            # Vérifier conflits de salle
            creneaux_utilises[moment_salle].append(idx)

            # Vérifier conflits de jurys
            jurys_par_moment[moment].extend([soutenance['Tuteur'], soutenance['Co-jury']])

        # Détecter les conflits de salle
        for creneau, indices in creneaux_utilises.items():
            if len(indices) > 1:
                conflits.append(f"Conflit de salle : {creneau} utilisé par {len(indices)} soutenances")

        # Détecter les conflits de jurys
        for moment, jurys in jurys_par_moment.items():
            jurys_uniques = set(jurys)
            if len(jurys) != len(jurys_uniques):
                conflits.append(f"Conflit de jury au moment {moment}")

        return conflits


# Interface utilisateur (partie simplifiée pour l'exemple)
# [Le reste du code d'interface reste identique jusqu'à la génération]

# Importation Excel
st.sidebar.header("📥 Importation Excel")
excel_file = st.sidebar.file_uploader("Importer un fichier Excel", type=["xlsx"])

if excel_file:
    try:
        excel_data = pd.read_excel(excel_file, sheet_name=None)

        if "etudiants" in excel_data:
            etu_df = excel_data["etudiants"]
            required_cols = {"Nom", "Prénom", "Pays", "Tuteur"}
            if required_cols.issubset(etu_df.columns):
                st.session_state.etudiants = etu_df[list(required_cols)].to_dict(orient="records")
                st.sidebar.success("Étudiants importés ✅")
            else:
                st.sidebar.error("La feuille 'etudiants' doit contenir les colonnes : Nom, Prénom, Pays, Tuteur")

        if "co_jurys" in excel_data:
            cj_df = excel_data["co_jurys"]
            if "Nom" in cj_df.columns:
                st.session_state.co_jurys = cj_df["Nom"].dropna().astype(str).tolist()
                st.sidebar.success("Co-jurys importés ✅")
            else:
                st.sidebar.error("La feuille 'co_jurys' doit contenir une colonne 'Nom'")

    except Exception as e:
        st.sidebar.error(f"Erreur lors de la lecture du fichier : {e}")

# Gestion des étapes (identique au code original jusqu'à la génération)
if st.session_state.etape == "etudiants":
    afficher_navigation()
    st.header("Étape 1 : Gestion des étudiants")

    with st.form("ajout_etudiant"):
        col1, col2 = st.columns(2)
        with col1:
            nom = st.text_input("Nom")
            prenom = st.text_input("Prénom")
        with col2:
            pays = st.text_input("Pays")
            tuteur = st.text_input("Tuteur")

        if st.form_submit_button("Ajouter étudiant") and all([nom, prenom, pays, tuteur]):
            st.session_state.etudiants.append({
                "Nom": nom, "Prénom": prenom, "Pays": pays, "Tuteur": tuteur
            })
            st.success(f"Étudiant {prenom} {nom} ajouté avec succès")

    if st.session_state.etudiants:
        st.subheader("Liste des étudiants")
        df_etudiants = pd.DataFrame(st.session_state.etudiants)
        st.dataframe(df_etudiants, use_container_width=True)

    if st.button("Passer à l'étape suivante", type="primary"):
        if st.session_state.etudiants:
            st.session_state.etape = "salles"
            st.rerun()
        else:
            st.error("Veuillez ajouter au moins un étudiant")

elif st.session_state.etape == "salles":
    afficher_navigation()
    st.header("Étape 2 : Configuration des salles")
    nb_salles = st.number_input("Nombre de salles disponibles", min_value=1, max_value=10, value=2, step=1)
    if st.button("Valider et continuer", type="primary"):
        st.session_state.nb_salles = nb_salles
        st.session_state.etape = "duree_soutenance"
        st.rerun()

elif st.session_state.etape == "duree_soutenance":
    afficher_navigation()
    st.header("Étape 3 : Durée des soutenances")
    duree = st.number_input("Durée d'une soutenance (minutes)", min_value=30, max_value=120, value=50, step=10)
    if st.button("Valider et continuer", type="primary"):
        st.session_state.duree_soutenance = duree
        st.session_state.etape = "co_jury"
        st.rerun()

elif st.session_state.etape == "co_jury":
    afficher_navigation()
    st.header("Étape 4 : Gestion des co-jurys")

    with st.form("ajout_cojury"):
        nom = st.text_input("Nom du co-jury")
        if st.form_submit_button("Ajouter co-jury") and nom:
            if nom not in st.session_state.co_jurys:
                st.session_state.co_jurys.append(nom)
                st.success(f"Co-jury {nom} ajouté")
            else:
                st.warning("Ce co-jury existe déjà")

    if st.session_state.co_jurys:
        st.subheader("Liste des co-jurys")
        for idx, cj in enumerate(st.session_state.co_jurys):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"👨‍🏫 {cj}")
            with col2:
                if st.button("Supprimer", key=f"cj_suppr_{idx}"):
                    del st.session_state.co_jurys[idx]
                    st.rerun()

    if st.button("Passer à l'étape suivante", type="primary"):
        st.session_state.etape = "dates"
        st.rerun()

elif st.session_state.etape == "dates":
    afficher_navigation()
    st.header("Étape 5 : Dates des soutenances")
    nb_jours = st.number_input("Nombre de jours de soutenances", min_value=1, max_value=5, value=2)
    dates = []
    for i in range(nb_jours):
        date = st.date_input(f"Date du jour {i + 1}", key=f"date_{i}")
        dates.append(date)
    if st.button("Valider les dates", type="primary"):
        st.session_state.dates_soutenance = dates
        st.session_state.etape = "disponibilites"
        st.rerun()

elif st.session_state.etape == "disponibilites":
    afficher_navigation()
    st.header("Étape 6 : Génération des créneaux")
    if st.session_state.dates_soutenance:
        horaires_par_jour = {}
        for jour in st.session_state.dates_soutenance:
            jour_str = jour.strftime("%A %d/%m/%Y")
            creneaux = []
            for (debut, fin) in [("08:00", "13:00"), ("14:00", "17:20")]:
                current = datetime.combine(jour, datetime.strptime(debut, "%H:%M").time())
                end = datetime.combine(jour, datetime.strptime(fin, "%H:%M").time())
                while current + timedelta(minutes=st.session_state.duree_soutenance) <= end:
                    fin_creneau = current + timedelta(minutes=st.session_state.duree_soutenance)
                    creneaux.append(f"{current.strftime('%H:%M')} - {fin_creneau.strftime('%H:%M')}")
                    current = fin_creneau
            horaires_par_jour[jour_str] = creneaux
        st.session_state.horaires_par_jour = horaires_par_jour
        for jour, slots in horaires_par_jour.items():
            st.subheader(f"📅 {jour}")
            cols = st.columns(min(len(slots), 4))
            for i, slot in enumerate(slots):
                with cols[i % 4]:
                    st.write(f"🕒 {slot}")
        if st.button("Passer à la saisie des disponibilités", type="primary"):
            st.session_state.etape = "disponibilites_selection"
            st.rerun()

elif st.session_state.etape == "disponibilites_selection":
    afficher_navigation()
    st.header("Étape 7 : Saisie des disponibilités")
    tous_tuteurs = list(set([e["Tuteur"] for e in st.session_state.etudiants]))
    personnes = tous_tuteurs + st.session_state.co_jurys
    for personne in personnes:
        st.subheader(f"👨‍🏫 Disponibilités de {personne}")
        if personne not in st.session_state.disponibilites:
            st.session_state.disponibilites[personne] = {}
        for jour, creneaux in st.session_state.horaires_par_jour.items():
            st.markdown(f"**{jour}**")
            all_key = f"{personne}_{jour}_all"
            all_selected = st.checkbox("Disponible toute la journée", key=all_key)
            cols = st.columns(min(len(creneaux), 3))
            for i, creneau in enumerate(creneaux):
                with cols[i % 3]:
                    key_dispo = f"{jour} | {creneau}"
                    if all_selected:
                        st.session_state.disponibilites[personne][key_dispo] = True
                        st.checkbox(creneau, value=True, disabled=True, key=f"disabled_{personne}_{i}")
                    else:
                        current_value = st.session_state.disponibilites[personne].get(key_dispo, False)
                        checked = st.checkbox(creneau, value=current_value, key=f"{personne}_{jour}_{i}")
                        st.session_state.disponibilites[personne][key_dispo] = checked
        st.divider()
    if st.button("Générer le planning", type="primary"):
        st.session_state.etape = "generation"
        st.rerun()

elif st.session_state.etape == "generation":
    afficher_navigation()
    st.header("🚀 Génération du planning optimisé")

    # Options d'optimisation
    col1, col2 = st.columns(2)
    with col1:
        utiliser_genetique = st.checkbox("Forcer l'utilisation de l'algorithme génétique",
                                       help="L'algorithme génétique sera utilisé automatiquement si l'algorithme classique ne donne pas de bons résultats")

    with col2:
        if utiliser_genetique:
            with st.expander("⚙️ Paramètres avancés"):
                taille_population = st.slider("Taille de la population", 20, 100, 50)
                nb_generations = st.slider("Nombre de générations", 50, 200, 100)
                taux_mutation = st.slider("Taux de mutation", 0.05, 0.3, 0.1)

    if st.button("Lancer l'optimisation", type="primary"):
        with st.spinner("Optimisation en cours..."):
            # Créer l'instance d'optimisation
            optimiseur = PlanificationOptimiseeV2(
                st.session_state.etudiants,
                st.session_state.co_jurys,
                st.session_state.dates_soutenance,
                st.session_state.disponibilites,
                st.session_state.nb_salles,
                st.session_state.duree_soutenance
            )

            # Paramètres génétiques si activés
            params_genetique = {}
            if utiliser_genetique:
                params_genetique = {
                    'taille_population': taille_population,
                    'nb_generations': nb_generations,
                    'taux_mutation': taux_mutation
                }

            # Générer le planning optimisé
            planning, non_planifies, stats_genetique = optimiseur.optimiser_avec_genetique(
                utiliser_genetique=utiliser_genetique,
                **params_genetique
            )
            st.session_state.planning_final = planning

        # Affichage des statistiques génétiques si disponibles
        if stats_genetique:
            st.subheader("🧬 Statistiques de l'algorithme génétique")

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Générations", stats_genetique['generations'])
            with col2:
                st.metric("Fitness finale", f"{stats_genetique['fitness_finale']:.1f}")
            with col3:
                st.metric("Conflits", stats_genetique['conflits'])
            with col4:
                st.metric("Amélioration",
                          f"+{len(planning) - (len(st.session_state.etudiants) - non_planifies)} soutenances")

            # Graphique d'évolution
            if stats_genetique['historique']:
                import plotly.graph_objects as go

                df_hist = pd.DataFrame(stats_genetique['historique'])

                fig_evolution = go.Figure()
                fig_evolution.add_trace(go.Scatter(
                    x=df_hist['generation'],
                    y=df_hist['fitness_max'],
                    mode='lines',
                    name='Fitness maximale',
                    line=dict(color='green')
                ))
                fig_evolution.add_trace(go.Scatter(
                    x=df_hist['generation'],
                    y=df_hist['soutenances_max'],
                    mode='lines',
                    name='Soutenances planifiées',
                    yaxis='y2',
                    line=dict(color='blue')
                ))

                fig_evolution.update_layout(
                    title="Évolution de l'algorithme génétique",
                    xaxis_title="Génération",
                    yaxis_title="Fitness",
                    yaxis2=dict(title="Soutenances", overlaying='y', side='right'),
                    height=400
                )

                st.plotly_chart(fig_evolution, use_container_width=True)

        # Vérification des conflits
        if planning:
            conflits = optimiseur.verifier_conflits(planning)
            if conflits:
                st.error("⚠️ Conflits détectés dans le planning :")
                for conflit in conflits:
                    st.write(f"- {conflit}")
            else:
                st.success("✅ Aucun conflit détecté dans le planning")

            # Affichage des résultats
            st.success(f"Planning généré avec succès ! {len(planning)} soutenances planifiées.")

            if non_planifies > 0:
                st.warning(f"⚠️ {non_planifies} étudiant(s) n'ont pas pu être planifiés.")

            # Créer le DataFrame
            df_planning = pd.DataFrame(planning)

            # Affichage du tableau
            st.subheader("📋 Planning détaillé")
            st.dataframe(df_planning.drop(['Début', 'Fin'], axis=1), use_container_width=True)

            # Graphique Gantt
            if not df_planning.empty:
                st.subheader("📊 Visualisation du planning")
                df_planning["Task"] = df_planning["Étudiant"] + " (" + df_planning["Salle"] + ")"
                fig = px.timeline(
                    df_planning,
                    x_start="Début",
                    x_end="Fin",
                    y="Tuteur",
                    color="Task",
                    title="Planning des soutenances par tuteur",
                    hover_data=["Étudiant", "Co-jury", "Salle", "Pays"]
                )
                fig.update_yaxes(autorange="reversed")
                fig.update_layout(height=600)
                st.plotly_chart(fig, use_container_width=True)

                # Statistiques
                st.subheader("📈 Statistiques")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Soutenances planifiées", len(planning))
                with col2:
                    st.metric("Taux de réussite", f"{len(planning) / len(st.session_state.etudiants) * 100:.1f}%")
                with col3:
                    st.metric("Salles utilisées", df_planning['Salle'].nunique())
                with col4:
                    st.metric("Jours utilisés", df_planning['Jour'].nunique())

                # Répartition par jour
                st.subheader("📅 Répartition par jour")
                repartition = df_planning.groupby(['Jour', 'Salle']).size().reset_index(name='Nombre')
                fig_bar = px.bar(repartition, x='Jour', y='Nombre', color='Salle',
                                 title="Nombre de soutenances par jour et par salle")
                st.plotly_chart(fig_bar, use_container_width=True)

        else:
            st.error("❌ Aucune soutenance n'a pu être planifiée. Vérifiez les disponibilités et les contraintes.")

# Sidebar avec informations
with st.sidebar:
    st.markdown("---")
    st.markdown("### 📊 Résumé")
    st.write(f"**Étudiants :** {len(st.session_state.etudiants)}")
    st.write(f"**Co-jurys :** {len(st.session_state.co_jurys)}")
    st.write(f"**Salles :** {st.session_state.nb_salles}")
    st.write(f"**Durée :** {st.session_state.duree_soutenance} min")

    if st.session_state.dates_soutenance:
        st.write(f"**Dates :** {len(st.session_state.dates_soutenance)} jour(s)")

    st.markdown("---")
    st.markdown("### ℹ️ À propos")
    st.write(
        "Système de planification automatique utilisant l'optimisation par graphes bipartites pour respecter toutes les contraintes.")

