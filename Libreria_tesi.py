import numpy as np
import matplotlib.pyplot as plt
from qutip import *


#--------------------------------------------------------------------------------------------------#

def MHQ_3level_vecchia(stato, hamiltoniano, c_ops, O_1, O_2, t_list, args = None, iniziali = False):
    """Restituisce le MHQ, probabilità congiunte TPM e probabilità esiti finali
    (per verifica marginalizzazione) per sistemi a tre livelli"""
    Q = np.zeros((3,3))
    TPM = np.zeros((3,3))

    # Determino proiettori necessari per O_1
    _, autovettori_1 = O_1.eigenstates()
    o_11, o_12, o_13 = autovettori_1
    proiettore_11 = o_11 * o_11.dag()
    proiettore_12 = o_12 * o_12.dag()
    proiettore_13 = o_13 * o_13.dag()

    # Determino proiettori necessari per O_2
    _, autovettori_2 = O_2.eigenstates()
    o_21, o_22, o_23 = autovettori_2    # Ordinati in maniera crescente degli autovalori
    proiettore_21 = o_21 * o_21.dag()
    proiettore_22 = o_22 * o_22.dag()
    proiettore_23 = o_23 * o_23.dag()

    # Grafici
    probabilita_iniziali = []

    # Schema TPM
    
    # Caso 1: misuro o_11
    rho_1 = (proiettore_11 * stato * proiettore_11).unit()
    prob_1 = np.real_if_close((proiettore_11 * stato).tr())
    probabilita_iniziali.append(prob_1)

    evoluto_1 = mesolve(hamiltoniano, rho_1, t_list, c_ops=c_ops, args=args)
    
    p_1_1 = np.real_if_close((proiettore_21 * evoluto_1.states[-1]).tr()) * prob_1 
    p_1_2 = np.real_if_close((proiettore_22 * evoluto_1.states[-1]).tr()) * prob_1
    p_1_3 = np.real_if_close((proiettore_23 * evoluto_1.states[-1]).tr()) * prob_1
    
    TPM[0, 0] = p_1_1
    TPM[0, 1] = p_1_2
    TPM[0, 2] = p_1_3

    # Caso 2: misuro o_12
    rho_2 = (proiettore_12 * stato * proiettore_12).unit()
    prob_2 = np.real_if_close((proiettore_12 * stato).tr())
    probabilita_iniziali.append(prob_2)

    evoluto_2 = mesolve(hamiltoniano, rho_2, t_list, c_ops=c_ops, args=args)
    
    p_2_1 = np.real_if_close((proiettore_21 * evoluto_2.states[-1]).tr()) * prob_2
    p_2_2 = np.real_if_close((proiettore_22 * evoluto_2.states[-1]).tr()) * prob_2
    p_2_3 = np.real_if_close((proiettore_23 * evoluto_2.states[-1]).tr()) * prob_2
    
    TPM[1, 0] = p_2_1
    TPM[1, 1] = p_2_2
    TPM[1, 2] = p_2_3

    # Caso 3: misuro o_13
    rho_3 = (proiettore_13 * stato * proiettore_13).unit()
    prob_3 = np.real_if_close((proiettore_13 * stato).tr())
    probabilita_iniziali.append(prob_3)

    evoluto_3 = mesolve(hamiltoniano, rho_3, t_list, c_ops=c_ops, args=args)
    
    p_3_1 = np.real_if_close((proiettore_21 * evoluto_3.states[-1]).tr()) * prob_3
    p_3_2 = np.real_if_close((proiettore_22 * evoluto_3.states[-1]).tr()) * prob_3
    p_3_3 = np.real_if_close((proiettore_23 * evoluto_3.states[-1]).tr()) * prob_3
    
    TPM[2, 0] = p_3_1
    TPM[2, 1] = p_3_2
    TPM[2, 2] = p_3_3

    # Post selezione (evoluzione senza misura iniziale)
    evoluto = mesolve(hamiltoniano, stato, t_list, c_ops=c_ops, args=args)

    pf_1 = np.real_if_close((proiettore_21 * evoluto.states[-1]).tr())
    pf_2 = np.real_if_close((proiettore_22 * evoluto.states[-1]).tr())
    pf_3 = np.real_if_close((proiettore_23 * evoluto.states[-1]).tr())

    # Misura non selettiva
    # Su O_1
    rho_ns = (proiettore_11 * stato * proiettore_11 + (qeye(3) - proiettore_11) * stato * (qeye(3) - proiettore_11)).unit()  # Stato dopo misura non selettiva
               
    evoluto_ns = mesolve(hamiltoniano, rho_ns, t_list, c_ops=c_ops, args=args)
    
    w_1_1 = np.real_if_close((proiettore_21 * evoluto_ns.states[-1]).tr())
    w_1_2 = np.real_if_close((proiettore_22 * evoluto_ns.states[-1]).tr())
    w_1_3 = np.real_if_close((proiettore_23 * evoluto_ns.states[-1]).tr())


    # Su O_2
    rho_ns = (proiettore_12 * stato * proiettore_12 + (qeye(3) - proiettore_12) * stato * (qeye(3) - proiettore_12)).unit()  # Stato dopo misura non selettiva
    evoluto_ns = mesolve(hamiltoniano, rho_ns, t_list, c_ops=c_ops, args=args)

    w_2_1 = np.real_if_close((proiettore_21 * evoluto_ns.states[-1]).tr())
    w_2_2 = np.real_if_close((proiettore_22 * evoluto_ns.states[-1]).tr())
    w_2_3 = np.real_if_close((proiettore_23 * evoluto_ns.states[-1]).tr())

    # Su O_3
    rho_ns = (proiettore_13 * stato * proiettore_13 + (qeye(3) - proiettore_13) * stato * (qeye(3) - proiettore_13)).unit()  # Stato dopo misura non selettiva

    evoluto_ns = mesolve(hamiltoniano, rho_ns, t_list, c_ops=c_ops, args=args)

    w_3_1 = np.real_if_close((proiettore_21 * evoluto_ns.states[-1]).tr())
    w_3_2 = np.real_if_close((proiettore_22 * evoluto_ns.states[-1]).tr())
    w_3_3 = np.real_if_close((proiettore_23 * evoluto_ns.states[-1]).tr())

    # Calcolo delle quasiprobabilità di Kirkwood-Dirac
    # Prima colonna (outcome finale 1)
    Q[0, 0] = p_1_1 + 0.5 * (pf_1 - w_1_1)
    Q[1, 0] = p_2_1 + 0.5 * (pf_1 - w_2_1)
    Q[2, 0] = p_3_1 + 0.5 * (pf_1 - w_3_1)
    
    # Seconda colonna (outcome finale 2)
    Q[0, 1] = p_1_2 + 0.5 * (pf_2 - w_1_2)
    Q[1, 1] = p_2_2 + 0.5 * (pf_2 - w_2_2)
    Q[2, 1] = p_3_2 + 0.5 * (pf_2 - w_3_2)
    
    # Terza colonna (outcome finale 3)
    Q[0, 2] = p_1_3 + 0.5 * (pf_3 - w_1_3)
    Q[1, 2] = p_2_3 + 0.5 * (pf_3 - w_2_3)
    Q[2, 2] = p_3_3 + 0.5 * (pf_3 - w_3_3)

    if iniziali:
        return Q, TPM, probabilita_iniziali

    return Q, TPM, [pf_1, pf_2, pf_3]


#--------------------------------------------------------------------------------------------------#


def MHQ_3level(stato, hamiltoniano, c_ops, O_1, O_2, t_list, args=None, iniziali=False):
    """
    Calcola le quasiprobabilità di Kirkwood-Dirac (MHQ), la matrice TPM 
    e le probabilità finali o iniziali, per un sistema a 3 livelli.
    """
    Q = np.zeros((3, 3), dtype=np.complex128)
    TPM = np.zeros((3, 3), dtype=np.float64)
    prob_iniziali = []

    # Proiettori autostati iniziali e finali
    _, eigvecs_1 = O_1.eigenstates()
    projs_1 = [v * v.dag() for v in eigvecs_1]
    
    _, eigvecs_2 = O_2.eigenstates()
    projs_2 = [v * v.dag() for v in eigvecs_2]

    # Misura iniziale selettiva
    rho_f = mesolve(hamiltoniano, stato, t_list, c_ops=c_ops, args=args).states[-1]
    pf = [np.real_if_close((P * rho_f).tr()) for P in projs_2]

    p_ij = np.zeros((3, 3), dtype=np.float64)
    w_ij = np.zeros((3, 3), dtype=np.float64)

    for i, Pi in enumerate(projs_1):
        prob_i = np.real_if_close((Pi * stato).tr())
        prob_iniziali.append(prob_i)
        rho_i = (Pi * stato * Pi).unit()
        rho_i_evol = mesolve(hamiltoniano, rho_i, t_list, c_ops=c_ops, args=args).states[-1]
        for j, Pj in enumerate(projs_2):
            p_ij[i, j] = np.real_if_close((Pj * rho_i_evol).tr()) * prob_i
            TPM[i, j] = p_ij[i, j]

        # Misura non selettiva su O1 (per ogni proiettore)
        Pi_comp = (qeye(3) - Pi)
        rho_ns = (Pi * stato * Pi + Pi_comp * stato * Pi_comp).unit()
        rho_ns_evol = mesolve(hamiltoniano, rho_ns, t_list, c_ops=c_ops, args=args).states[-1]
        for j, Pj in enumerate(projs_2):
            w_ij[i, j] = np.real_if_close((Pj * rho_ns_evol).tr())

    # Costruzione delle MHQ (Kirkwood-Dirac)
    for i in range(3):
        for j in range(3):
            Q[i, j] = p_ij[i, j] + 0.5 * (pf[j] - w_ij[i, j])

    if iniziali:
        return Q, TPM, prob_iniziali
    return Q, TPM, pf


#--------------------------------------------------------------------------------------------------#

def plot_MHQ_3level(Q, TPM, autovalori_1, autovalori_2):
    """Visualizza la MHQ e la TPM per un sistema a 3 livelli."""
    fig, ax = plt.subplots(1, 2, figsize=(13, 6))

    # Etichette
    labels_f = [f'{v:.3f}' for v in autovalori_2]
    labels_i = [f'{v:.3f}' for v in autovalori_1]

    # MHQ plot
    cax1 = ax[0].matshow(Q.real, cmap='RdBu_r', vmin=-1, vmax=1)
    fig.colorbar(cax1, ax=ax[0])
    ax[0].set_title('MHQ (wTPM)')
    ax[0].set_xlabel('Hamiltoniano finale')
    ax[0].set_ylabel('Hamiltoniano iniziale')
    ax[0].set_xticks(range(3))
    ax[0].set_yticks(range(3))
    ax[0].set_xticklabels(labels_f)
    ax[0].set_yticklabels(labels_i)

    for (i, j), val in np.ndenumerate(Q.real):
        ax[0].text(j, i, f'{val:.3f}', ha='center', va='center',
                   color='white' if abs(val) > 0.5 else 'black')

    # TPM plot
    cax2 = ax[1].matshow(TPM, cmap='RdBu_r', vmin=-1, vmax=1)
    fig.colorbar(cax2, ax=ax[1])
    ax[1].set_title('Probabilità (TPM)')
    ax[1].set_xlabel('Hamiltoniano finale')
    ax[1].set_ylabel('Hamiltoniano iniziale')
    ax[1].set_xticks(range(3))
    ax[1].set_yticks(range(3))
    ax[1].set_xticklabels(labels_f)
    ax[1].set_yticklabels(labels_i)

    for (i, j), val in np.ndenumerate(TPM):
        ax[1].text(j, i, f'{val:.3f}', ha='center', va='center',
                   color='black')

    plt.tight_layout()
    plt.show()



#--------------------------------------------------------------------------------------------------#

def prob_condizionate(initial_state, H, c_ops, O1, H_final_func, t_list, args=None):
    """
    Calcola le probabilità condizionate p(f_j | i, t) dove l'osservabile
    finale è l'Hamiltoniano istantaneo H_final_func(t).
    
    Restituisce:
    TPM_time : ndarray, shape (3, 3, len(t_list))
        TPM_time[i,j,k] = p(f_j | i, t_list[k])
    """
    # Proiettori iniziali
    _, vecs1 = O1.eigenstates()
    projs1 = [v * v.dag() for v in vecs1]
    
    # Preallocazione
    n = len(projs1)
    TPM_time = np.zeros((n, n, len(t_list)))
    
    # Ciclo su esiti iniziali
    for i, P_i in enumerate(projs1):
        rho_i = (P_i * initial_state * P_i).unit()
        result = mesolve(H, rho_i, t_list, c_ops=c_ops, args=args)
        
        # Per ogni tempo, diagonalizzo H_final e calcolo p(f|i,t)
        for idx, t in enumerate(t_list):
            rho_t = result.states[idx]
            Hf = H_final_func(t)
            _, vecs2 = Hf.eigenstates()
            projs2 = [v * v.dag() for v in vecs2]
            for j, P_f in enumerate(projs2):
                TPM_time[i, j, idx] = np.real_if_close((P_f * rho_t).tr())
    
    return TPM_time



#--------------------------------------------------------------------------------------------------#

# Plotting
def plot_prob_cond(TPM_time, t_list):
    fig, axes = plt.subplots(3, 1, figsize=(6, 8), sharex=True)
    for i in range(TPM_time.shape[0]):
        for j in range(TPM_time.shape[1]):
            axes[i].plot(t_list, TPM_time[i, j, :], label=f"f={j-1} | i={i-1}")
        axes[i].set_ylabel(f"p(f|i={i-1}, t)")
        axes[i].legend()
        axes[i].grid(True)
    axes[-1].set_xlabel("Tempo")
    plt.tight_layout()
    plt.show()
