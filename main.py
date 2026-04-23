import numpy as np

def main():
    ManyBody()
    return

def Nanostructure():
    #zwraca energię i funkcje falowe 3D nanostruktury
    return

def Coulomb(Psi):
    #zwraca Elementy macierzowe <m|V_coul|n>, gdzie n,m odpowiada funkcjom falowym Psi
    return

def ManyBody(basis_size=0, single_eigval=0, single_eigvec=0, coul_matrix=0):
    #Znajduje energie układu wielocząstkowego
    basis_cutoff = 10 

    basis = []

    for i in range(1, basis_cutoff):
        for j in range(i, basis_cutoff):
            basis.append([i, j, 1, 2])
            
            if i != j:
                basis.append([i, j, 2, 1])
                basis.append([i, j, 1, 1])
                basis.append([i, j, 2, 2])

    basis = np.array(basis)
    print(basis)
    return

if __name__ == "__main__":
    main()
    
