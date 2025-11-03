"""
 Compute the Majorana weight of a Pauli string.
"""
function majorana_weight(Pb::Union{PauliBasis{N}, Pauli{N}}) where N
    w = 0
    control = true
    # tmp = Pb.z & ~Pb.x  # Bitwise AND with bitwise NOT
    Ibits = ~(Pb.z|Pb.x)
    Zbits = Pb.z & ~Pb.x

    for i in reverse(1:N)  # Iterate from N down to 1
        xbit = (Pb.x >> (i - 1)) & 1 != 0
        Zbit = (Zbits >> (i - 1)) & 1 != 0
        Ibit = (Ibits >> (i - 1)) & 1 != 0
        #println("i=$i, xbit=$xbit, Zbit=$Zbit, Ibit=$Ibit, control=$control, w=$w")
        if Zbit && control || Ibit && !control
            w += 2
        elseif xbit
            control = !control
            w += 1
        end
    end
    return w
end

"""
 Compute the Pauli weight of a Pauli string.
"""
function pauli_weight(Pb::Union{PauliBasis{N}, Pauli{N}}) where N
    w = 0
    for i in 1:N
        xbit = (Pb.x >> (i - 1)) & 1
        zbit = (Pb.z >> (i - 1)) & 1

        if xbit != 0 || zbit != 0
            w += 1
        end
    end
    return w
end

"""
HS norm in Pauli basis: sum_P |c_P|^2.
"""
function hs_norm2(ps::PauliSum)
    s = 0.0
    @inbounds for (_, c) in ps
        s += abs2(c)
    end
    return s
end

function inner_product(O1::PauliSum{N,T}, O2::PauliSum{N,T}) where {N,T}
    out = T(0)
    if length(O1) < length(O2)
        for (p1,c1) in O1
            if haskey(O2,p1)
                out += c1'*O2[p1]
            end
        end
    else
        for (p2,c2) in O2
            if haskey(O1,p2)
                out += c2*O1[p2]'
            end
        end
    end
    return out
end

function largest_diag(ps::PauliSum{N,T}) where {N,T}
    argmax(kv -> abs(last(kv)), filter(p->p.first.x == 0, ps))
end
    
function largest(ps::PauliSum{N,T}) where {N,T}
    max_val, max_key = findmax(v -> abs(v), ps)

    return PauliSum{N,T}(max_key => ps[max_key])
end
    
function LinearAlgebra.diag(ps::PauliSum{N,T}) where {N,T}
    filter(p->p.first.x == 0, ps)
end

function offdiag(ps::PauliSum{N,T}) where {N,T}
    filter(p->p.first.x != 0, ps)
end

function LinearAlgebra.norm(p::PauliSum{N,T}) where {N,T}
    out = T(0)
    for (p,c) in p 
        out += abs2(c) 
    end
    return sqrt(real(out))
end

function LinearAlgebra.norm(p::KetSum{N,T}) where {N,T}
    out = T(0)
    for (p,c) in p 
        out += abs2(c) 
    end
    return sqrt(real(out))
end

function weight(p::PauliBasis) 
    return count_ones(p.x | p.z)
end

function coeff_clip!(ps::KetSum{N}; thresh=1e-16) where {N}
    return filter!(p->abs(p.second) > thresh, ps)
end

"""
    Clip based on Pauli weight.
    Performs the pruning by removing all terms with weight > max_weight.
"""
function weight_clip!(ps::PauliSum{N}, max_weight::Int) where {N}
    return filter!(p->weight(p.first) <= max_weight, ps)
end

function majorana_weight_clip!(ps::PauliSum{N}, max_weight::Int) where {N}
    return filter!(p->majorana_weight(p.first) <= max_weight, ps)
end

"""
    Combined weight and coefficient clipping.
    w_type = 0 : Pauli weight
    w_type = 1 : Majorana weight
"""
function clip_thresh_weight!(ps::PauliSum{N}; thresh=1e-16, lc = 0, w_type = 0) where {N}
    if w_type == 0 
        filter!(p->(weight(p.first) <= lc) && (abs(p.second) > thresh) , ps)
    else
        filter!(p->(majorana_weight(p.first) <= lc) && (abs(p.second) > thresh) , ps)
    end     
end

function reduce_by_1body(p::PauliBasis{N}, ψ) where N
    out = PauliSum(N)
    # for i in 1:N
    n_terms = length(PauliOperators.get_on_bits(p.z|p.x)) 
    for i in PauliOperators.get_on_bits(p.z|p.x) 
        mask = 1 << (i - 1) 
        tmp1 = PauliBasis{N}(p.z & ~mask, p.x & ~mask)
        tmp2 = PauliBasis{N}(p.z & mask, p.x & mask)
        tmp3 = tmp1*tmp2
        if isapprox(coeff(tmp3), 1) == false || PauliBasis(tmp3) != p
            throw(ErrorException)
        end 
        # println(string(tmp1), "*", string(tmp2), "=", string(tmp1*tmp2))
        out += tmp1 * (expectation_value(tmp2, ψ) / n_terms)
        out += tmp1 * (expectation_value(tmp2, ψ) / n_terms)
        # display(PauliBasis{N}(p.z & ~mask, p.x & ~mask))
    end
    out = out * (1/norm(out))
    # for (p,c) in out
    #     println(string(p), " ", weight(p))
    # end
    return out
end

function meanfield_reduce!(O::PauliSum{N},s, weightclip) where N
    tmp = PauliSum(N)
    for (p,c) in O
        if weight(p) > weightclip 
            tmp += reduce_by_1body(p,s)
            O[p] = 0
        end
    end 
    O += tmp
end


function find_top_k(dict, k=10)
    """Optimized for when k << length(dict)"""
    
    # Pre-allocate arrays
    top_keys = Vector{keytype(dict)}(undef, k)
    top_vals = Vector{valtype(dict)}(undef, k) 
    top_abs = Vector{Float64}(undef, k)
    
    n_found = 0
    min_val = 0.0
    min_idx = 1
    
    @inbounds for (key, val) in dict
        abs_val = abs(val)
        
        if n_found < k
            # Still filling up
            n_found += 1
            top_keys[n_found] = key
            top_vals[n_found] = val  
            top_abs[n_found] = abs_val
            
            # Update minimum
            if abs_val < min_val || n_found == 1
                min_val = abs_val
                min_idx = n_found
            end
            
        elseif abs_val > min_val
            # Replace minimum
            top_keys[min_idx] = key
            top_vals[min_idx] = val
            top_abs[min_idx] = abs_val
            
            # Find new minimum
            min_val = top_abs[1]
            min_idx = 1
            for i in 2:k
                if top_abs[i] < min_val
                    min_val = top_abs[i]
                    min_idx = i
                end
            end
        end
    end
    
    # Sort the results
    p = sortperm(view(top_abs, 1:n_found), rev=true)
    return [top_keys[p[i]] => top_vals[p[i]] for i in 1:n_found]
end

function find_top_k_offdiag(dict, k=10)
    """Optimized for when k << length(dict)"""
    
    # Pre-allocate arrays
    top_keys = Vector{keytype(dict)}(undef, k)
    top_vals = Vector{valtype(dict)}(undef, k) 
    top_abs = Vector{Float64}(undef, k)
    
    n_found = 0
    min_val = 0.0
    min_idx = 1
    
    @inbounds for (key, val) in dict
        key.x != 0 || continue
        abs_val = abs(val)
        
        if n_found < k
            # Still filling up
            n_found += 1
            top_keys[n_found] = key
            top_vals[n_found] = val  
            top_abs[n_found] = abs_val
            
            # Update minimum
            if abs_val < min_val || n_found == 1
                min_val = abs_val
                min_idx = n_found
            end
            
        elseif abs_val > min_val
            # Replace minimum
            top_keys[min_idx] = key
            top_vals[min_idx] = val
            top_abs[min_idx] = abs_val
            
            # Find new minimum
            min_val = top_abs[1]
            min_idx = 1
            for i in 2:k
                if top_abs[i] < min_val
                    min_val = top_abs[i]
                    min_idx = i
                end
            end
        end
    end
    
    # Sort the results
    p = sortperm(view(top_abs, 1:n_found), rev=true)
    return [top_keys[p[i]] => top_vals[p[i]] for i in 1:n_found]
end


function get_weight_counts(O::PauliSum{N}) where N
    counts = zeros(Int, N)
    for (p,c) in O
        counts[weight(p)] += 1
    end
    return counts
end


function get_weight_probs(O::PauliSum{N}) where N
    probs = zeros(N)
    for (p,c) in O
        probs[weight(p)] += abs2(c) 
    end
    return probs 
end

function add_single_excitations(k::Ket{N}) where N
    s = KetSum(N)
    s[k] = 1
    for i in 1:N
        for j in 1:N
            i != j || continue
            c,b = Pauli(N, X=[i, j]) * k
            # count_ones(k.v) == count_ones(b.v) || continue 
            coeff = get(s, b, 0)
            s[b] = coeff + c
        end
    end
    # for (k,c) in s 
    #     @show count_ones(k.v)
    # end
    return s
end

"""
    Base.Matrix(p::PauliSum{N,T}, Vector{Ket{N}}) where {N,T}

Build Matrix representation of `p` in the space dfined by `S`
"""
# function Base.Matrix(p::PauliSum{N,T}, S::Vector{Ket{N}}) where {N,T}
#     nS = length(S)
#     M = zeros(T,nS,nS)
#     for i in 1:nS
#         M[i,i] = expectation_value(p,S[i])
#         for j in i+1:nS
#             M[i,j] = matrix_element(S[i]',p,S[j])
#             M[j,i] = matrix_element(S[j]',p,S[i])
#         end
#     end
#     return M
# end
function Base.Matrix(O::PauliSum{N,T}, S::Vector{Ket{N}}) where {N,T}
    nS = length(S)
    
    o = pack_x_z(O)

    def = Vector{Tuple{Int128, Float64}}()

    M = zeros(T,nS,nS)
    for i in 1:nS
        M[i,i] = expectation_value(O,S[i])
        for j in i+1:nS
            x = S[i].v ⊻ S[j].v
            ox = get(o, x, def)
            for (z,c) in ox
                p = PauliBasis{N}(z,x)
                phase, k = p*S[j]
                M[i,j] += phase*c

                phase, k = p*S[i]
                M[j,i] += phase*c
            end
        end
    end
    return M
end

function Base.Matrix(O::XZPauliSum{T}, basis::Vector{Ket{N}}) where {N,T}
    n = length(basis)
    
    def = Dict{Int128, Float64}()

    M = zeros(ComplexF64,n,n)
    for (i, keti) in enumerate(basis)
        # M[i,i] = expectation_value(O,keti)

        for (j, ketj) in enumerate(basis)
            j >= i || continue
            x = keti.v ⊻ ketj.v
            ox = get(O, x, def)
            for (z,c) in ox
            # for (z,c) in o[x]
                p = PauliBasis{N}(z,x)
                phase,_ = p*ketj
                M[i,j] += phase*c

                j > i || continue
                phase,_  = p*keti
                M[j,i] += phase*c
            end
        end
    end
    return M
end

function Base.Matrix(k::KetSum{N,T}, S::Vector{Ket{N}}) where {N,T}
    nS = length(S)
    v = zeros(T,nS,1)
    length(k) == length(S) || throw(DimensionMismatch)
    for (i,keti) in enumerate(S)
        v[i,1] = k[keti]
    end
    return v
end

function Base.Vector(k::KetSum{N,T}, S::Vector{Ket{N}}) where {N,T}
    nS = length(S)
    v = zeros(T,nS)
    length(k) == length(S) || throw(DimensionMismatch)
    for (i,keti) in enumerate(S)
        v[i] = k[keti]
    end
    return v
end




function Base.:*(O::PauliSum{N,T}, k::Ket{N}) where {N,T}
    out = KetSum(N)
    for (p,c) in O
        c2,k2 = p*k
        tmp = get(out, k2, 0.0)
        out[k2] = tmp + c2*c
    end
    return out 
end

function PauliOperators.expectation_value(O::PauliSum, v::KetSum)
    ev = 0
    for (p,c) in O
        for (k1,c1) in v
            ev += expectation_value(p,k1)*c*c1'*c1
            for (k2,c2) in v
                k2 != k1 || continue
                ev += matrix_element(k2', p, k1)*c*c2'*c1
            end
        end
    end
    return ev
end

function PauliOperators.expectation_value(O::XZPauliSum, v::KetSum{N,T}) where {N,T}
    ev = 0
    for (x,zs) in O
        for (z,c) in zs 
            p = PauliBasis{N}(z,x)
            for (k1,c1) in v
                ev += expectation_value(p,k1)*c*c1'*c1
                for (k2,c2) in v
                    k2 != k1 || continue
                    ev += matrix_element(k2', p, k1)*c*c2'*c1
                end
            end
        end
    end
    return ev
end
function PauliOperators.expectation_value(O::XZPauliSum, v::Ket{N}) where {N}
    ev = 0
    haskey(O,0) || return 0.0
    for (z,c) in O[0]
        p = PauliBasis{N}(z,Int128(0))
        ev += expectation_value(p,v)*c
    end
    return ev
end

"""
    pack_x_z(H::PauliSum{N,T}) where {N,T}

Convert PauliSum into a Dict{Int128,Vector{Tuple{Int128,Float64}}}
This allows us to access Pauli's by first specifying `x`, 
then 'z'. 
"""
function pack_x_z(H::PauliSum{N,T}) where {N,T}
    # Build [X][Z] container
    h = Dict{Int128,Vector{Tuple{Int128,T}}}()
    for (p,c) in H
        dx = get(h, p.x, Vector{Tuple{Int128,T}}())
        push!(dx, (p.z,c))
        h[p.x] = dx
    end
    return h
end

# --------------------- Hubbard model Helpers ---------------------
# --- Helpers for particle counts / Sz ---

"""
    count_particles(bits::AbstractVector{<:Integer}) -> Int

Count number of occupied modes (1s) in `bits`.
"""
count_particles(bits::AbstractVector{<:Integer}) = sum(bits)

"""
    spin_z(bits::AbstractVector{<:Integer}) -> Real

Compute total S_z = (N_up - N_down)/2 assuming up spins are in indices
1,3,5,... and down spins are in indices 2,4,6,... 
"""
function spin_z(bits::AbstractVector{<:Integer})
    @assert length(bits) % 2 == 0 "Number of modes should be even (pairs up/down per site)."
    up   = sum(bits[1:2:end])
    down = sum(bits[2:2:end])
    return (up - down) / 2
end

"""
    indices_sector(basis::Vector{Vector{Int}}; N::Union{Int,Nothing}=nothing, Sz::Union{Real,Nothing}=nothing)

Return indices (1-based) of `basis` vectors (a list of bits) that satisfy particle number `N` and/or total `Sz`.
If `N` or `Sz` is `nothing`, that constraint is ignored.
"""
function indices_sector(basis::Vector{Vector{Int}}; N::Union{Int,Nothing}=nothing, Sz::Union{Real,Nothing}=nothing)
    inds = Int[]
    for (i, bits) in enumerate(basis)
        if (N === nothing || count_particles(bits) == N) && (Sz === nothing || spin_z(bits) == Sz)
            push!(inds, i)
        end
    end
    return inds
end

using Random
"""
    build_from_coupling(coupling::Vector{Vector{Int}}, N::Int, Nparticles::Int, Sz::Union{Real,Nothing};
                        rng=GLOBAL_RNG)

Attempt to construct an occupation vector `occ::Vector{Int}` of length `N` that:

 - places exactly `Nparticles` electrons,
 - satisfies total spin-z `Sz` if not `nothing`, and
 - is consistent with the grouping `coupling`, where each element of `coupling`
   is a vector of mode indices that must have identical *site-occupations*
   (i.e., the total occupation per group is identical across all groups).

Returns `occ` if construction succeeded, otherwise throws an informative error.

Notes:
 - The construction returns a Neel-based state if Sz=0 or a state with spins as spread as possible otherwise,
 - `coupling` indices are 1-based mode indices (Julia convention).
 - The routine tries to distribute particles equally across groups, then assign
   up/down within groups to meet `Sz`.

   How to choose `coupling`:
   For a 2D square lattice of size Lx x Ly, with sites numbered row-wise from 1 to Lx*Ly,
   each site has two modes: up (odd index) and down (even index).
   A common choice is to group modes by site, e.g., for a 2x2 lattice:       
   neighbors = [
    (1,2),  # site0-site1 (horizontal top)
    (2,3),  # site1-site2 (vertical right)
    (3,4),  # site2-site3 (horizontal bottom)
    (4,1)   # site3-site0 (vertical left)
]
"""
function build_from_coupling(coupling::Vector{<:AbstractVector{<:Integer}},
                             N::Int, Nparticles::Int, Sz::Union{Real,Nothing}=nothing; rng::AbstractRNG=Random.GLOBAL_RNG, kwargs...)
    ngroups = length(coupling)
    @assert ngroups > 0 "coupling must contain at least one group."
    @assert all(map(g -> all(1 .<= g .<= N), coupling)) "All coupling indices must be in 1..N."

    # --- Reject overlapping groups for now (require a partition) ---
    all_indices = vcat(coupling...)
    if length(all_indices) != length(unique(all_indices))
        error("Overlapping groups detected in `coupling`. This builder assumes non-overlapping groups (a partition of modes). " *
              "If you intend overlapping constraints, please convert to a non-overlapping site partition or request an overlapping-aware solver.")
    end

    # helper: return `k` indices spread across 1..n
    function spread_indices(n::Int, k::Int)
        if k <= 0
            return Int[]
        elseif k == 1
            return [clamp(Int(round((1 + n) / 2)), 1, n)]
        else
            idxs = unique(round.(Int, range(1, stop = n, length = k)))
            # pad if uniqueness reduced the count
            j = 1
            while length(idxs) < k
                if !(j in idxs)
                    push!(idxs, j)
                end
                j += 1
                if j > n
                    break
                end
            end
            sort!(idxs)
            return idxs
        end
    end

    # Each group must have same total occupation per_group.
    if Nparticles % ngroups != 0
        error("Total Nparticles=$(Nparticles) is not divisible by number of groups=$(ngroups). Can't make identical per-group occupations.")
    end
    per_group = div(Nparticles, ngroups)

    # Special-case Néel pattern when groups are size-2 (site pairs), half filling, no Sz
    is_site_pairs = all(length(g) == 2 for g in coupling)
    default_flavor = get(kwargs, :flavor, :Aup)
    if is_site_pairs && (Nparticles == ngroups) && Sz === nothing
        occ = zeros(Int, N)
        for site in 0:(ngroups-1)
            idxs = coupling[site+1]
            if default_flavor == :Aup
                if isodd(site)
                    occ[idxs[1]] = 0
                    occ[idxs[2]] = 1
                else
                    occ[idxs[1]] = 1
                    occ[idxs[2]] = 0
                end
            elseif default_flavor == :Bup
                if isodd(site)
                    occ[idxs[1]] = 1
                    occ[idxs[2]] = 0
                else
                    occ[idxs[1]] = 0
                    occ[idxs[2]] = 1
                end
            else
                error("Unknown Neel flavor: $default_flavor. Use :Aup or :Bup.")
            end
        end
        return occ
    end

    # If Sz provided, compute required N_up and N_down
    N_up = nothing
    if Sz !== nothing
        val = Nparticles/2 + Sz
        if abs(round(val) - val) > 1e-12
            error("Requested Sz=$(Sz) incompatible with integer N_up given Nparticles=$(Nparticles).")
        end
        N_up = Int(round(val))
        @assert 0 <= N_up <= Nparticles "Computed N_up outside allowed range."
    end

    # Precompute up/down mode indices within each group (odd indices are up, even are down)
    group_up_idxs = Vector{Vector{Int}}(undef, ngroups)
    group_down_idxs = Vector{Vector{Int}}(undef, ngroups)
    for (i,g) in enumerate(coupling)
        group_up_idxs[i] = [j for j in g if isodd(j)]
        group_down_idxs[i] = [j for j in g if iseven(j)]
    end

    occ = zeros(Int, N)

    # If no Sz constraint, choose arbitrary per-group occupations
    if Sz === nothing
        for i in 1:ngroups
            slots = coupling[i]
            if per_group > length(slots)
                error("per-group occupation $(per_group) exceeds number of modes in group $(i).")
            end
            chosen = randperm(rng, length(slots))[1:per_group]
            for c in chosen
                occ[slots[c]] = 1
            end
        end
        return occ
    end

    # --- Sz provided: compute feasible bounds for up-count per group ---
    # For group i, ucnt must satisfy:
    #   lb[i] = max(0, per_group - (#down_slots))
    #   ub[i] = min(#up_slots, per_group)
    ng = ngroups
    lb = zeros(Int, ng)
    ub = zeros(Int, ng)
    for i in 1:ng
        nd = length(group_down_idxs[i])
        nu = length(group_up_idxs[i])
        lb[i] = max(0, per_group - nd)   # must place at least this many ups so downs fit
        ub[i] = min(nu, per_group)       # cannot place more ups than up-slots or per_group
        if lb[i] > ub[i]
            error("Group $(i) cannot realize per-group total=$(per_group) given its up/down slot capacities (lb=$(lb[i]) > ub=$(ub[i])).")
        end
    end

    # Check global feasibility
    min_total = sum(lb)
    max_total = sum(ub)
    if N_up < min_total || N_up > max_total
        error("No feasible distribution of N_up=$(N_up) into groups. Allowed total up electrons are in [$min_total, $max_total].")
    end

    # Start from lower bounds and distribute the remaining 'rem' up electrons across groups (spread-out)
    up_counts = copy(lb)
    rem = N_up - sum(up_counts)
    if rem > 0
        # distribute using spread order while respecting ub
        positions = collect(1:ng)
        # choose spread order (spread_indices with k=rem)
        spread_pos = spread_indices(ng, rem)
        # but we need to possibly add >1 to same position if ub allows and rem large;
        # greedy loop: iterate through spread sequence round-robin
        idx = 1
        while rem > 0
            pos = spread_pos[(idx - 1) % length(spread_pos) + 1]
            addable = ub[pos] - up_counts[pos]
            if addable > 0
                up_counts[pos] += 1
                rem -= 1
            end
            idx += 1
            # if we've cycled many times and cannot assign, fall back to any group with spare capacity
            if idx > 10 * ng && rem > 0
                for i in 1:ng
                    add = min(ub[i] - up_counts[i], rem)
                    if add > 0
                        up_counts[i] += add
                        rem -= add
                        if rem == 0
                            break
                        end
                    end
                end
                if rem > 0
                    error("Unable to distribute remaining up electrons despite feasibility check.")
                end
            end
        end
    end

    # Now assign actual indices within each group, using spread selection for which slots within up/down candidates
    for i in 1:ng
        ucnt = up_counts[i]
        dcnt = per_group - ucnt
        ups = group_up_idxs[i]
        downs = group_down_idxs[i]
        if ucnt > 0
            if length(ups) == ucnt
                chosen_up = ups
            else
                pos = spread_indices(length(ups), ucnt)
                chosen_up = ups[pos]
            end
            for idx in chosen_up
                occ[idx] = 1
            end
        end
        if dcnt > 0
            if length(downs) == dcnt
                chosen_down = downs
            else
                pos = spread_indices(length(downs), dcnt)
                chosen_down = downs[pos]
            end
            for idx in chosen_down
                occ[idx] = 1
            end
        end
    end

    @assert count_particles(occ) == Nparticles "constructed occ has wrong number of particles"
    @assert isapprox(spin_z(occ), Sz; atol=1e-12) "constructed occ has incorrect Sz"

    return occ
end


# ---------------------
# particle_ket function (enhanced)
# ---------------------

"""
    particle_ket(N::Int, Nparticles::Int, Sz::Real; mode=:first, basis=nothing, coupling=nothing, rng=GLOBAL_RNG, kwargs...)

Return `(ket_obj, occ)` where `occ` is a Vector{Int} of length `N` with exactly `Nparticles` ones
(obeying `Sz` if possible).  `ket_obj` is `Ket(occ)` if a `Ket` constructor exists, otherwise `nothing`.

New:
 - `coupling` (optional): a Vector of groups (each group is a vector of 1-based mode indices).
    When provided and `mode==:coupling` (or when `mode==:c4_invariant` and `coupling` is given),
    the function will attempt to construct a state consistent with the coupling mapping
    (equal per-group occupations), without requiring a `basis`.
"""
function particle_ket(N::Int, Nparticles::Int, Sz::Union{Real,Nothing}; mode=:first,
                      basis::Union{Vector{Vector{Int}},Nothing}=nothing,
                      coupling::Union{Vector{Vector{Int}},Nothing}=nothing, use_rng=true,
                      rng::AbstractRNG=Random.GLOBAL_RNG, kwargs...)
    @assert 0 ≤ Nparticles ≤ N "Number of particles must be between 0 and N"
    @assert iseven(N) "Number of modes N should be even (pairs up/down per site assumed)."

    occ = zeros(Int, N)

    if mode == :random
        idxs = randperm(rng, N)[1:Nparticles]
        occ[idxs] .= 1

    elseif mode == :alternate
        occ[1:2:N] .= 1  # 1,0,1,0,...
        # adjust to requested number of particles
        n1 = count_particles(occ)
        if n1 > Nparticles
            idxs = findall(x->x==1, occ)
            occ[idxs[(Nparticles+1):end]] .= 0
        elseif n1 < Nparticles
            idxs = findall(x->x==0, occ)
            occ[idxs[1:(Nparticles - n1)]] .= 1
        end

    elseif mode == :first
        occ[1:Nparticles] .= 1

    elseif mode == :subspace
        if basis === nothing
            error("mode=:subspace requires a `basis` argument (Vector of bit-vectors).")
        end
        # find indices in given basis that satisfy N and Sz
        sector_inds = indices_sector(basis; N=Nparticles, Sz=Sz)
        if isempty(sector_inds)
            error("No states found in provided basis with N=$(Nparticles), Sz=$(Sz).")
        end
        if use_rng
            chosen = rand(rng, sector_inds)
        else
            chosen = sector_inds[1]
        end
        occ .= basis[chosen]  # copy bits into occ

    elseif mode == :Neel
        # Néel classical AFM: one electron per site, alternating up/down on sublattices.
        # Require half-filling: Nparticles == number_sites (N/2)
        number_sites = div(N, 2)
        @assert Nparticles == number_sites "Neel mode expects half-filling: Nparticles must equal number of sites (N/2)."
        flavor = get(kwargs, :flavor, :Aup)
        for site in 0:(number_sites-1)
            if flavor == :Aup
                if isodd(site) # B sublattice -> down
                    occ[2*site + 1] = 0   # up
                    occ[2*site + 2] = 1   # down
                else            # A sublattice -> up
                    occ[2*site + 1] = 1
                    occ[2*site + 2] = 0
                end
            elseif flavor == :Bup
                if isodd(site) # B sublattice -> up
                    occ[2*site + 1] = 1
                    occ[2*site + 2] = 0
                else            # A sublattice -> down
                    occ[2*site + 1] = 0
                    occ[2*site + 2] = 1
                end
            else
                error("Unknown Neel flavor: $flavor. Use :Aup or :Bup.")
            end
        end

    elseif mode == :coupling
        # coupling must be provided for general behavior. For legacy :c4_invariant, we accept the old behavior when N==8
        if coupling === nothing
            error("mode=$(mode) requires a `coupling` argument (Vector of mode-index groups).")
        end
        # Use build_from_coupling to create occ consistent with coupling
        occ .= build_from_coupling(coupling, N, Nparticles, Sz; rng=rng)

    else
        error("Unknown mode: $mode")
    end

    # sanity check
    @assert count_particles(occ) == Nparticles "Constructed occupation vector has wrong particle number."

    # Ensure Sz matches (if not nothing)
    if Sz !== nothing
        @assert isapprox(spin_z(occ), Sz; atol=1e-12) "Constructed occupation vector has Sz=$(spin_z(occ)) but requested Sz=$(Sz)."
    end

    # Try to create a Ket if available; otherwise return nothing for ket_obj.
    ket_obj = nothing
    try
        if @isdefined(Ket)
            ket_obj = Ket(occ)   # user environment may provide Ket constructor
        end
    catch err
        # ignore errors constructing Ket, return nothing in ket_obj
        ket_obj = nothing
    end

    return ket_obj, occ
end
