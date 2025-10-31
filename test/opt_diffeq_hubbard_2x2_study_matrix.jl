using DBF
using PauliOperators
using Printf
using Random
using LinearAlgebra
using JLD2

"""
  Here we perform calculations for different approximations to the projector:
  single particle, two particle, etc.
  Using diffeq approach.
"""

# --- Helpers for particle counts / Sz ---

# Count number of occupied modes (bits == 1)
function count_particles(bits::AbstractVector{Int})
    return sum(bits)
end

# Compute total spin-z = (N_up - N_down)/2
# Convention: up spins at even indices (1,3,5,7...), down at odd (2,4,6,8...)
function spin_z(bits::AbstractVector{Int})
    up   = sum(bits[1:2:end])
    down = sum(bits[2:2:end])
    return (up - down) / 2
end

# Return indices of basis states with given N and/or Sz
function indices_sector(basis::Vector{Vector{Int}}; N::Union{Int,Nothing}=nothing, Sz::Union{Float64,Nothing}=nothing)
    inds = Int[]
    for (i, bits) in enumerate(basis)
        if (N === nothing || count_particles(bits) == N) &&
           (Sz === nothing || spin_z(bits) == Sz)
            push!(inds, i)
        end
    end
    return inds
end

"""
    particle_ket(N::Int, Nparticles::Int; mode=:first)

Return a `Ket` of length `N` with exactly `Nparticles` ones.

`mode` options:
  • `:first`   → first `Nparticles` sites occupied (|111000...>)
  • `:random`  → random positions for the 1s
  • `:alternate` → alternating pattern |101010...> (if Nparticles ≈ N/2)

If the alternating pattern has more or fewer 1s than `Nparticles`,
the result is truncated or padded with zeros.
"""
function particle_ket(N::Int, Nparticles::Int; mode=:first)
    @assert 0 ≤ Nparticles ≤ N "Number of particles must be between 0 and N"
    occ = zeros(Int, N)

    if mode == :random
        occ[randperm(N)[1:Nparticles]] .= 1

    elseif mode == :alternate
        occ[1:2:N] .= 1  # pattern 1,0,1,0,...
        # adjust to requested number of particles
        n1 = count(==(1), occ)
        if n1 > Nparticles
            # trim some 1s from the end
            idxs = findall(==(1), occ)
            occ[idxs[(Nparticles+1):end]] .= 0
        elseif n1 < Nparticles
            # add 1s where there are zeros
            idxs = findall(==(0), occ)
            occ[idxs[1:(Nparticles - n1)]] .= 1
        end

    elseif mode == :first
        occ[1:Nparticles] .= 1

    else 
        error("Unknown mode: $mode")

    end

    return Ket(occ), occ
end

function run()
    Random.seed!(2)
    Lx = 2
    Ly = 2
    Nsites = Lx * Ly
    N = 2 * Nsites   # 2 spin states per site
    t = 1.0
    U = 1.0
    H = DBF.fermi_hubbard_2D(Lx, Ly, t, U)
    #H = DBF.fermi_hubbard_2D_snake(Lx, Ly, t, U; snake_ordering=true)
    #H = DBF.hubbard_model_1D(Nsites, t, U)

    println(" Original H:")
    display(H)

    
    Nparticles = 2
    ψ, occ = particle_ket(N, Nparticles, mode=:random)
    display(ψ)
    println(" Occupation: ", occ)

    # Transform H to make |000> the most stable bitstring
    for i in 1:N
        if occ[i] == 1
            H = Pauli(N, X=[i]) * H * Pauli(N, X=[i])
        end
    end 
    
    H0 = deepcopy(H)
    println(" Transformed H:")
    display(H)
    
    Hmat = Matrix(H)
    evals = eigvals(Hmat)
    @show minimum(evals)
    gne = minimum(evals)

    ψ = Ket([0 for i in 1:N])
    display(ψ)

    e0 = expectation_value(H,ψ)
    @printf(" Reference = %12.8f\n", e0)
    
    println("\n ########################")
    @time DBF.groundstate_diffeq_matrix(H0, ψ, n_body=0, 
                                verbose=1, 
                                max_iter=1000, gne=gne, conv_thresh=1e-2, 
                                stepsize=.0001)

    return nothing
end


run()