import QuantumRingsLib
from QuantumRingsLib import QuantumRegister, AncillaRegister, ClassicalRegister, QuantumCircuit
from QuantumRingsLib import QuantumRingsProvider
from QuantumRingsLib import job_monitor
from QuantumRingsLib import JobStatus
from skopt import gp_minimize
import numpy as np
import math
import os
import logging

# import QuantumRingsLib
# from QuantumRingsLib import QuantumRegister, ClassicalRegister, QuantumCircuit
# from QuantumRingsLib import QuantumRingsProvider
# from QuantumRingsLib import job_monitor
import json
# import numpy as np
# import os
from scipy.optimize import minimize

logger = logging.getLogger(__name__)

provider = QuantumRingsProvider(token=os.environ.get('TOKEN_QUANTUMRINGS'), name=os.environ.get('ACCOUNT_QUANTUMRINGS'))
backend = provider.get_backend("scarlet_quantum_rings")
provider.active_account()
# define the operator U(B, beta)
def Operator_UB(graph, beta,qc, q, n_qubits):
    for i in range(n_qubits): qc.rx(2*beta, q[i])

# define the operator U(C,gamma)
def Operator_UC(graph, gamma, qc, q, n_qubits):
    for edge in graph:
        qc.cx(q[edge[0]], q[edge[1]])
        # multiply the gamma by the weight of the edge
        qc.rz(gamma*edge[2], q[edge[1]])
        qc.cx(q[edge[0]], q[edge[1]])

# a helper routine that computes the total weight of the cuts
def WeightOfCuts(bitstring,graph):
    totalWeight = 0
    for edge in graph:
        if bitstring[edge[0]] != bitstring[edge[1]]:
            totalWeight += edge[2]
    return totalWeight

def jobCallback(job_id, state, job):
    #print("Job Status: ", state)
    pass

# Builds the QAOA state.
def qaoaState( x, graph, p, n_qubits, expectationValue = True, shots=1024):
    gammas = []
    betas = []
    # setup the gamma and beta array
    for i in range(len(x)//2):
        gammas.append(x[i*2])
        betas.append(x[(i*2)+1])
    # Create the quantum circuit
    q = QuantumRegister(n_qubits)
    c = ClassicalRegister(n_qubits)
    qc = QuantumCircuit(q, c)

    # First set the qubits in an equal superposition state
    for i in range(n_qubits):
        qc.h(q[i])

    # Apply the gamma and beta operators in repetition
    for i in range(p):
        # Apply U(C,gamma)
        Operator_UC(graph, gammas[i], qc, q, n_qubits)

        # Apply U(B, beta)
        Operator_UB(graph, betas[i],qc, q, n_qubits)

    # Measure the qubits
    for i in range(n_qubits):
        qc.measure(q[i], c[i])

    # Execute the circuit now
    job = backend.run(qc, shots)
    job.wait_for_final_state(0, 5, jobCallback)
    counts = job.result().get_counts()

    # decide what to return
    if ( True == expectationValue ):
        # Calculate the expectation value of the measurement.
        expectation = 0
        for bitstring in counts:
            probability = float(counts[bitstring]) / shots
            expectation += WeightOfCuts(bitstring,graph) * probability
        return( expectation )
    else:
        # Just return the counts return(counts)
        return(counts)
########################## Modified from here on ###################
# Load the JSON data with esg_score and company_value
def load_esg_data(filename):
    with open(filename, 'r') as file:
        return json.load(file)

# Setup the Hamiltonian based on ESG scores and company values
def get_Paulioperator(esg_data):
    pauli_list = []
    weights = []
    # print(esg_data)
    # Create Pauli operators based on ESG scores and company values
    for company in esg_data:
        score = company["esg_score"]
        value = company["company_value"]
        
        # Combine esg_score and company_value into a single weight
        weight = score * value
        weights.append(weight)
        
        # Create a corresponding Pauli operator (assuming 4 companies)
        pauli_operator = "I" * len(esg_data)  # Use 'I' for non-target qubits
        pauli_operator = pauli_operator[:len(pauli_operator)-len(weights)] + "Z" * len(weights)
        pauli_list.append([weight, pauli_operator])

    return pauli_list

# Setting up Quantum Rings Provider
provider = QuantumRingsProvider(token=os.environ.get('TOKEN_QUANTUMRINGS'), name=os.environ.get('ACCOUNT_QUANTUMRINGS'))
backend = provider.get_backend("scarlet_quantum_rings")
shots = 1024

# print(provider.active_account())

# Define the ansatz (can use SimpleAnsatz or TwoLocalAnsatz)
def SimpleAnsatz(qc, q, n_qubits, theta_list, reps=5, insert_barriers=False):
    for i in range(reps + 1):
        for j in range(n_qubits):
            qc.ry(theta_list[(i * n_qubits) + j], q[j])
        if insert_barriers:
            qc.barrier()
    return

# Define perform_pauli_measurements function
def perform_pauli_measurements(qubitOp, param_dict, SHOTS=1024):
    avg = 0.0
    n_qubits = len(qubitOp[0][1])
    pauli_list = qubitOp

    for p in pauli_list:
        weight = p[0].real
        pauli = p[1]

        # Convert pauli string to integer representation
        pauli_int = int(p[1].replace("I", "0").replace("Z", "1").replace("X", "1").replace("Y", "1"), 2)

        qc = QuantumCircuit(n_qubits)
        for i in range(n_qubits):
            qc.ry(param_dict[i], i)

        # Apply Pauli operators
        for i in range(n_qubits):
            if pauli[i] == "Y":
                qc.sdg(i)
                qc.h(i)
            elif pauli[i] == "X":
                qc.h(i)

        qc.measure_all()
        job = backend.run(qc, shots=SHOTS, mode="sync", performance="HighestEfficiency", quiet=True)
        job_monitor(job, quiet=True)

        results = job.result()
        result_dict = results.get_counts()

        # Measurement calculation
        measurement = 0.0
        for key, value in result_dict.items():
            sign = -1.0 if (bin(int(key, 2) & pauli_int).count("1") & 1) else 1.0
            measurement += sign * value
        measurement /= SHOTS
        measurement *= weight
        avg += measurement

    return avg

# Define the cost function for optimization
def cost_function(param_dict, qubitOp, shots):
    avg_energy = perform_pauli_measurements(qubitOp, param_dict, SHOTS=shots)
    return avg_energy

# Function to calculate individual company energies
def calculate_company_energies(optimized_param_dict, esg_data):
    company_energies = []
    
    for i, company in enumerate(esg_data):
        company_weight = company['esg_score'] * company['company_value']
        pauli_op = [[company_weight, "Z" + "I" * (len(esg_data) - 1)]]
        
        energy = perform_pauli_measurements(pauli_op, optimized_param_dict, SHOTS=1024)
        company_energies.append({
            "company": company["company"],
            "energy": energy
        })
    
    return company_energies


def remove_unreliable_data(data):
    ## Filter out the companies with 
    res = list(filter(lambda x: x['esg_performance']!='LAG_PERF' and x['company_value']!=None, data))
    return res

def run(input_data,solver_params,extra_args):
    # Load ESG data from JSON with company_value
    # esg_data = load_esg_data('./small_company_esgscore_company_val.json')
    data=[]
    for i,val in enumerate(input_data.keys()):
        data.append({'esg_score':input_data[val]['Sustainability']['esgScores']['totalEsg'],
                     'company':input_data[val]['Name'],
                     'company_value':input_data[val]['Balance Sheet']['Total Assets'],
                     'esg_performance':input_data[val]['Sustainability']['esgScores']['esgPerformance']})
    esg_data = data
    esg_data = remove_unreliable_data(esg_data)
    qubitOp = get_Paulioperator(esg_data)
    
    # Initialize parameters (random starting point)
    n_qubits = len(esg_data)
    param_dict = np.random.rand((n_qubits + 1) * n_qubits)
    # Optimization using scipy.optimize.minimize
    result = minimize(cost_function, param_dict, args=(qubitOp, shots), method='COBYLA', options={'maxiter': 100})
    
    # Get optimized parameters and minimum energy
    optimized_param_dict = result.x
    min_energy = result.fun
    
    # print("Optimized parameters:", optimized_param_dict)
    # print("Minimum energy:", min_energy)
    
    # Calculate company-specific energies
    company_energies = calculate_company_energies(optimized_param_dict, esg_data)
    
    # Find the company closest to the minimum energy
    # closest_company = min(company_energies, key=lambda x: abs(x["energy"] - min_energy))
    # print(f"Company closest to minimum energy: {closest_company['company']} with energy: {closest_company['energy']}")
    
    return sorted(company_energies,key=lambda x: abs(x["energy"] - min_energy))

