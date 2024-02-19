#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 16:29:42 2023

@author: AFANVI Kodjo Roland
"""
#%% Realisation du projet 
# SUJET 2: ALGORITHME DE GRAM-SCHMIDT ET DECOMPOSITION QR
import numpy as np
from scipy import linalg as LA
from matplotlib import pyplot as plt



#DEfinition d une fonctionn pour verifier si une matrice passeé en parametre est inversible
def isInvertible(M):
    epsilon = 0.000000001
    if(LA.det(M)>-epsilon and LA.det(M)<epsilon):
        return False
    else:
        return True
    
    
#Definition d une fonction qui me construit une base orthonormée d'un ensemble de vecteur si ces derniers  sont lineairement independant
def gram_schmidt(M):
    #on verifie si elle sont lineairement independantes avant de l'orthonormaliser
    l,n=M.shape
    Q=np.zeros((l,n))
    proj = np.zeros((l,n))
    #On itnitialise la matrice Q avec le premier vecteur normalisé
    Q[:,0] = M[:,0] / LA.norm(M[:,0])
    #On fait une boucle pour orthonormaliser les autres vecteurs
    for i in range(1, n):
        # Calcul de la projection orthogonale de M[i] sur les vecteurs déjà normalisés
        #proj = np.zeros_like(M[i],dtype=np.float64)
        for j in range(i):
            #proj += (np.dot(M[i], Q[j]) * Q[j]).astype(np.float64)
            proj[:,i] += (np.dot(M[:,i], Q[:,j]) * Q[:,j])
        # on soustrait la projection orthogonale de M[i] pour avoir un vecteur orthogonal à Q
        v = M[:,i] - proj[:,i]
        # Normaliser le vecteur v et l'ajouter à la matrice Q
        Q[:,i] =  v / LA.norm(v)
    return Q
        
    
    
    
    
    
# Definition d une fonction qui calcule la decomposition QR
def decompo_QR(A):
    if isInvertible(A):
        Q, R = qr_decomposition(A)
        Q=np.round(Q, decimals=8)
        R=np.round(R, decimals=8)
        np.set_printoptions(precision=8, suppress=True)
        print("La decomposition QR de la matrice est :\n")
        print("Q: \n", Q)
        print("R: \n", R)
    else:
        print("La matrice donnée n'est pas inversible")

    
    
def qr_decomposition(A):
    Q = gram_schmidt(A)
    R = np.dot(Q.T, A)
    Q=np.round(Q, decimals=8)
    R=np.round(R, decimals=8)
    return Q, R

def visualisation_matrice(A):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set_aspect('equal')
    plt.imshow(A, interpolation='nearest', cmap=plt.cm.ocean)
    plt.colorbar()
    plt.show()

A = np.array([[2,1,2],[4,3,3],[4,2,1]])

B = np.array([[1,2],[2,4]])
Q1, R1 = qr_decomposition(A)
Q2, R2 = LA.qr(A)

print("Résultats de qr_decomposition :")
print("Q : \n", Q1)
print("R : \n", R1)
print("\n")
print("Résultats de LA.qr :")
print("Q : \n", Q2)
print("R : \n", R2)

print("\n")
print("\n")
visualisation_matrice(A)
visualisation_matrice(Q1)
visualisation_matrice(R1)

print("Résultats de qr_decomposition :")
print("Norme d'un vecteur de Q: \n",np.dot(Q1[:,0], Q1[:,0]))
print("Produit scalaire de du vecteur a l'indice 0 et 1 : \n",np.dot(Q1[:,0], Q1[:,1]))
print("Produit scalaire de du vecteur a l'indice 0 et 1 : \n",np.dot(Q1[:,0], Q1[:,2]))
print("Produit scalaire de du vecteur a l'indice 0 et 1 : \n",np.dot(Q1[:,1], Q1[:,2]))

print("\n")
print("Résultats de LA.qr :")
print("Norme d'un vecteur de Q: \n",np.dot(Q2[:,0], Q2[:,0]))
print("Produit scalaire de du vecteur a l'indice 0 et 1 : \n",np.dot(Q2[:,0], Q2[:,1]))
print("Produit scalaire de du vecteur a l'indice 0 et 1 : \n",np.dot(Q2[:,0], Q2[:,2]))
print("Produit scalaire de du vecteur a l'indice 0 et 1 : \n",np.dot(Q2[:,1], Q2[:,2]))

#%%




