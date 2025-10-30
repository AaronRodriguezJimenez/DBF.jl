using DBF
using PauliOperators
using Printf
using Random
using LinearAlgebra
using JLD2


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

    ψ = Ket([0 for i in 1:N])
    display(ψ)

    e0 = expectation_value(H,ψ)
    @printf(" Reference = %12.8f\n", e0)
    
    g = Vector{PauliBasis{N}}([]) 
    θ = Vector{Float64}([]) 

    println("\n ########################")
    @time H, g, θ = DBF.groundstate_diffeq_test(H, ψ, n_body=2, 
                                verbose=1, 
                                max_iter=1000, conv_thresh=1e-4, 
                                evolve_coeff_thresh=1e-8,
                                grad_coeff_thresh=1e-8,
                                stepsize=.001)
    
    # @save "out_$(i).jld2" N ψ H0 H g θ
    
    return 
    # return


    println("\n Now reroptimize with higher accuracy:")
    @show length(θ)
    Ht = deepcopy(H0)
    err = 0
    ecurr = expectation_value(Ht,ψ)
    @printf(" Initial energy: %12.8f %8i\n", ecurr, length(Ht))
    for (i,gi) in enumerate(g)
            
        θj, costi = DBF.optimize_theta_expval_test(Ht, gi, ψ, verbose=0)
        Ht = DBF.evolve(Ht, gi, θj)
        θ[i] = θj
        
        e1 = expectation_value(Ht,ψ)
        DBF.coeff_clip!(Ht, thresh=1e-5)
        e2 = expectation_value(Ht,ψ)

        err += e2 - e1
        if i%100 == 0
            @printf(" Error: %12.8f\n", err)
            e0, e2 = DBF.pt2(Ht, ψ)
            @printf(" E0 = %12.8f E2 = %12.8f EPT2 = %12.8f \n", e0, e2, e0+e2)
            e0, e, v, basis = DBF.cepa(Ht, ψ, thresh=1e-6, tol=1e-2, verbose=0)
            e0, e, v, basis = DBF.fois_ci(Ht, ψ, thresh=1e-6, tol=1e-2, verbose=0)
        end
    end    
    ecurr = expectation_value(Ht,ψ)
    @printf(" ecurr %12.8f err %12.8f %8i\n", ecurr, err, length(Ht))
   

end


run()