# This code will return Order of best compnay that one can invest in 
import QuantumRingsLib
from QuantumRingsLib import QuantumRegister, ClassicalRegister, QuantumCircuit
from QuantumRingsLib import QuantumRingsProvider
from QuantumRingsLib import job_monitor
import json
import numpy as np
import os
from scipy.optimize import minimize

# Load the JSON data with esg_score and company_value
def load_esg_data(filename):
    with open(filename, 'r') as file:
        return json.load(file)

# Setup the Hamiltonian based on ESG scores and company values
def get_Paulioperator(esg_data):
    pauli_list = []
    weights = []

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
def run(data):
    # Load ESG data from JSON with company_value
    # esg_data = load_esg_data('./small_company_esgscore_company_val.json')
    esg_data = data
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