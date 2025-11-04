using DBF
using PauliOperators
using Printf
using Random
using LinearAlgebra

function n_op(N, site_i)
    O = PauliSum(N, ComplexF64)
    O += DBF.JWmapping(N, i=site_i, j=site_i)
    return O
end

function n_op(N, site_i, site_j)
    O = PauliSum(N, ComplexF64)
    O += DBF.JWmapping(N, i=site_i, j=site_j)
    return O
end

function n(N, site_i)
    O = PauliSum(N, ComplexF64)
    O += Pauli(N) - Pauli(N, Z=[site_i])
    O *= 0.5
    return O
end

function n(N, site_i, site_j)
    O = PauliSum(N, ComplexF64)
    O += Pauli(N, X=[site_i, site_j])
    O += Pauli(N, Y=[site_i, site_j])
    O += 1im*Pauli(N, X=[site_i], Y=[site_j])
    O += -1im*Pauli(N, Y=[site_i], X=[site_j])
    O *= 0.25
    return O
end

function run()
     # Parameters for Hubbard model
    Lx = 8
    Ly = 1
    Nsites = Lx * Ly
    N = 2 * Nsites   # 2 spin states per site
    Nparticles = Nsites  # Half-filling

    ψ = Ket{N}(Int128(0))
    ψ_occ, occ = DBF.particle_ket(N, Nparticles, 0.0; mode=:Neel, flavor=:Aup)

    for site in 1:Nsites
        O = n(N, site) #+ n_op(N, site_dn)
        ev = expectation_value(O, ψ_occ)
        @printf(" Site %3d: <n> = %12.8f\n", site, ev)
    end
    return
end

run()