using JLD2
using ITensors, ITensorMPS
using PauliOperators
using DBF
using Statistics
using Printf
using Random
using LinearAlgebra
using JLD2


Lx = 7
Ly = 8
t = 1.0
U = 1.0

H = DBF.fermi_hubbard_2D_zigzag(Lx, Ly, t, U)
pauli_strings = []
    coeffs = []
    for (c, p) in H
    # println(string(c))
        push!(pauli_strings, string(c))
        push!(coeffs, real(p))
    end
os = OpSum()
for (pstr, c) in zip(pauli_strings, coeffs)
        opsites = []
    for (i, p) in enumerate(pstr)
            if p != 'I'
                push!(opsites, (string(p), i))
            end
    end

    if length(opsites) > 0
            # Flatten operator-site pairs and splat the tuple for OpSum
            flat_opsites = Tuple([x for pair in opsites for x in pair])
            # println("Adding term: ", c, " * ", opsites)
            os += c, flat_opsites...
    end
end
bitstring = "1001100110011010011001100110100110011001101001100110011010011001100110100110011001101001100110011010011001100110"
N = length(bitstring)
sites = siteinds("S=1/2", N; conserve_qns=false)
# Convert bitstring to array of states

states = [bitstring[n] == '0' ? "0" : "1" for n in 1:N]
sites = siteinds("Qubit", N)
H_ = MPO(os, sites)
psi0 = MPS(Float64, sites, states)
nsweeps = 30
maxdims = [100, 200, 200,200,300,300,400,400]
cutoff = [1.0e-10]
noise = [1.0e-6, 1.0e-7, 1.0e-8,1.0e-8,1.0e-8,1.0e-8,1.0e-8,1.0e-8,1.0e-8,1.0e-8,1.0e-8,1.0e-8,1.0e-8,1.0e-8,1.0e-8,1.0e-8,1.0e-8,1.0e-8,1.0e-8,1.0e-8,1.0e-8,1.0e-8,1.0e-8,1.0e-8,1.0e-8,0.0,0.0,0.0,0.0,0.0 ]

energy, psi = dmrg(H_, psi0; nsweeps, maxdim=maxdims, cutoff, noise) 


function run()

    energies = Float64[]
    energies_corr = Float64[]
    Lx=7
    Ly=8
    N_sites = Lx * Ly
    t=1.0
    U=1.0
    H = DBF.fermi_hubbard_2D_zigzag(Lx,Ly, t, U)
    DBF.coeff_clip!(H)
    ψ= Ket{Int(2*N_sites)}(0)
    println(ψ)
    N = 2*N_sites
    bitstring = "1001100110011010011001100110100110011001101001100110011010011001100110100110011001101001100110011010011001100110"
    print(" Initial bitstring = $bitstring \n")
    # print("N = $N \n")
    for i in 1:N
        if bitstring[i] == '1'
           H = Pauli(N, X=[i]) * H * Pauli(N, X=[i])
        end
    end
    println(" Hamiltonian constructed.")

    H0 = deepcopy(H)
    # display(ψ)
    e0 = expectation_value(H,ψ)
    # e0_ = expectation_value(H_,ψ)

    @printf(" Reference = %12.8f\n", e0)
    # @printf(" expectation value of chemical potential = %12.8f\n", e0_)

    println("\n ########################")
    @time res = DBF.dbf_groundstate(H, ψ,
                                        verbose=1,
                                        max_iter=100,
                                        conv_thresh=1e-4,
                                        evolve_coeff_thresh=1e-4,
                                        grad_coeff_thresh=1e-6,
                                        energy_lowering_thresh=1e-6,
                                        max_rots_per_grad=50,
                                        checkfile = "t1e4_N_$(N_sites)_dbf.jld2"
                                        )

    push!(energies, res["energies"][end]/N)
    push!(energies_corr, (res["energies"][end] - res["accumulated_error"][end] + res["pt2_per_grad"][end])/N)
    println("===========FINAL================")

    println("DBF")
    display(energies)
    println("Corrected Energies")
    display(energies_corr)

    return res
end


#@profilehtml run(4)
res=run()