using PauliOperators
using Random

"""
    indices_with_nparticles(N::Int, Nparticles::Int)

Return a vector of integer indices such that `Ket{N}(idx)` has exactly
`Nparticles` ones in its binary representation.
"""
function indices_with_nparticles(N::Int, Nparticles::Int)
    inds = Int[]
    max_idx = 2^N - 1
    for idx in 0:max_idx
        if count_ones(idx) == Nparticles
            push!(inds, idx)
        end
    end
    return inds
end

# provide posible indices for a 4qubit state with 2 particles
indx_vec = indices_with_nparticles(10,5)  # Example usage

for idx in indx_vec
    println("Index: ", idx)
    ket = Ket{10}(idx)
    display(ket)
end


"""
    occupancy_to_index(bits::AbstractVector{<:Integer})

Convert a 0/1 vector into its integer index (binary -> decimal).
"""
function occupancy_to_index(bits::AbstractVector{<:Integer})
    N = length(bits)
    idx = 0
    for i in 1:N
        idx = (idx << 1) | bits[i]
    end
    return idx
end

bits = [1,0,1,1]
occupancy_to_index(bits)  # → 11 (binary 1011)


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

    else  # :first
        occ[1:Nparticles] .= 1
    end

    return Ket(occ)
end


# provide a 6qubit state with 3 particles
ket_example = particle_ket(50, 25, mode=:random)
println("Ket with 3 particles in 6 sites:")
display(ket_example)