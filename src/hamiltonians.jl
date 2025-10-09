using PauliOperators
using Random

function heisenberg_1D(N, Jx, Jy, Jz; x=0, y=0, z=0)
    H = PauliSum(N, Float64)
    for i in 0:N-1
        H += -2*Jx * Pauli(N, X=[i+1,(i+1)%(N)+1])
        H += -2*Jy * Pauli(N, Y=[i+1,(i+1)%(N)+1])
        H += -2*Jz * Pauli(N, Z=[i+1,(i+1)%(N)+1])
    end 
    for i in 1:N
        H += x * Pauli(N, X=[i])
        H += y * Pauli(N, Y=[i])
        H += z * Pauli(N, Z=[i])
    end 
    return H
end



"""
    heisenberg_2D(Nx, Ny, Jx, Jy, Jz; x=0, y=0, z=0, periodic=true)

Create a 2D Heisenberg Hamiltonian on an Nx × Ny square lattice.

# Arguments
- `Nx`, `Ny`: Lattice dimensions  
- `Jx`, `Jy`, `Jz`: Nearest-neighbor coupling constants
- `x`, `y`, `z`: External magnetic field components
- `periodic`: Whether to use periodic boundary conditions (default: true)

# Returns
- `PauliSum` representing the 2D Heisenberg Hamiltonian

# Notes
The Hamiltonian has the form:
H = Σ_{⟨i,j⟩} [-2Jx σᵢˣσⱼˣ - 2Jy σᵢʸσⱼʸ - 2Jz σᵢᶻσⱼᶻ] + Σᵢ [x σᵢˣ + y σᵢʸ + z σᵢᶻ]

Sites are indexed in row-major order: site (i,j) → index = i + j*Nx + 1
"""
function heisenberg_2D(Nx, Ny, Jx, Jy, Jz; x=0, y=0, z=0, periodic=true)
    N_total = Nx * Ny
    H = PauliSum(N_total, Float64)
    
    # Helper function to convert 2D coordinates to 1D index (1-based)
    coord_to_index(i, j) = i + j * Nx + 1
    
    # Nearest-neighbor interactions
    for j in 0:Ny-1  # Row index
        for i in 0:Nx-1  # Column index
            current_site = coord_to_index(i, j)
            
            # Right neighbor (i+1, j)
            if i < Nx - 1 || periodic
                right_i = periodic ? (i + 1) % Nx : i + 1
                right_site = coord_to_index(right_i, j)
                
                H += -2*Jx * Pauli(N_total, X=[current_site, right_site])
                H += -2*Jy * Pauli(N_total, Y=[current_site, right_site])
                H += -2*Jz * Pauli(N_total, Z=[current_site, right_site])
            end
            
            # Up neighbor (i, j+1)  
            if j < Ny - 1 || periodic
                up_j = periodic ? (j + 1) % Ny : j + 1
                up_site = coord_to_index(i, up_j)
                
                H += -2*Jx * Pauli(N_total, X=[current_site, up_site])
                H += -2*Jy * Pauli(N_total, Y=[current_site, up_site])
                H += -2*Jz * Pauli(N_total, Z=[current_site, up_site])
            end
        end
    end
    
    # External magnetic field terms
    for site in 1:N_total
        H += x * Pauli(N_total, X=[site])
        H += y * Pauli(N_total, Y=[site]) 
        H += z * Pauli(N_total, Z=[site])
    end
    
    return H
end



function heisenberg_2D_zigzag(Nx, Ny, Jx, Jy, Jz; x=0, y=0, z=0, periodic=true)
    N_total = Nx * Ny
    H = PauliSum(N_total, Float64)

    # Zigzag (snake-like) row-major indexing
    coord_to_index(i, j) = isodd(j) ? (j - 1) * Nx + i : j * Nx - i + 1

    # Helper functions for periodic wrapping
    right(i) = periodic ? (i % Nx) + 1 : (i < Nx ? i + 1 : nothing)
    up(j) = periodic ? (j % Ny) + 1 : (j < Ny ? j + 1 : nothing)

    # Nearest-neighbor interactions
    for j in 1:Ny        # rows
        for i in 1:Nx    # columns
            current_site = coord_to_index(i, j)
            # Right neighbor (i+1, j)
            i_r = right(i)
            if i_r !== nothing
                right_site = coord_to_index(i_r, j)
                # display(current_site)
                # display(right_site)
                # println("++++++++++")
                H += -2*Jx * Pauli(N_total, X=[current_site, right_site])
                H += -2*Jy * Pauli(N_total, Y=[current_site, right_site])
                H += -2*Jz * Pauli(N_total, Z=[current_site, right_site])
            end

            # Up neighbor (i, j+1)
            j_u = up(j)
            if j_u !== nothing
                up_site = coord_to_index(i, j_u)
                # display(current_site)
                # display(up_site)
                # println("==============")
                H += -2*Jx * Pauli(N_total, X=[current_site, up_site])
                H += -2*Jy * Pauli(N_total, Y=[current_site, up_site])
                H += -2*Jz * Pauli(N_total, Z=[current_site, up_site])
            end
        end
    end

    # External field terms
    for site in 1:N_total
        H += x * Pauli(N_total, X=[site])
        H += y * Pauli(N_total, Y=[site])
        H += z * Pauli(N_total, Z=[site])
    end

    return H
end

"""
 - - - Heisenberg model 2D (snake/zizag ordering) - - -
"""
function heisenberg_2D_snake(Lx::Int, Ly::Int, Jx::Float64, Jy::Float64, Jz::Float64; x=0, y=0, z=0, snake_ordering::Bool=false)
    Nsites  = Lx * Ly
    N_total = Nsites
    H = PauliSum(N_total, Float64)

    # Spin indices for site j (1-based site indexing)
    up(j) = j
    dn(j) = j + Nsites

    # Map (row x, col y) -> site index j in [1, Nsites]
    @inline function site_index(x::Int, y::Int)
        if !snake_ordering
            # Row-major
            return (x - 1) * Ly + y
        else
            # Snake (zigzag) row-major:
            # odd rows (x=1,3,...) go left->right
            # even rows (x=2,4,...) go right->left
            if isodd(x)
                return (x - 1) * Ly + y
            else
                return x * Ly - (y - 1)
            end
        end
    end

# Nearest-neighbor interactions
    for j in 0:Ly-1  # Row index
        for i in 0:Lx-1  # Column index
            current_site = site_index(i, j)
            
            # Right neighbor (i+1, j)
            if i < Lx - 1 || periodic
                right_i = periodic ? (i + 1) % Lx : i + 1
                right_site = site_index(right_i, j)
                
                H += -2*Jx * Pauli(N_total, X=[current_site, right_site])
                H += -2*Jy * Pauli(N_total, Y=[current_site, right_site])
                H += -2*Jz * Pauli(N_total, Z=[current_site, right_site])
            end
            
            # Up neighbor (i, j+1)  
            if j < Ly - 1 || periodic
                up_j = periodic ? (j + 1) % Ly : j + 1
                up_site = site_index(i, up_j)
                
                H += -2*Jx * Pauli(N_total, X=[current_site, up_site])
                H += -2*Jy * Pauli(N_total, Y=[current_site, up_site])
                H += -2*Jz * Pauli(N_total, Z=[current_site, up_site])
            end
        end
    end
    
    # External magnetic field terms
    for site in 1:N_total
        H += x * Pauli(N_total, X=[site])
        H += y * Pauli(N_total, Y=[site]) 
        H += z * Pauli(N_total, Z=[site])
    end

    DBF.coeff_clip!(H, thresh=eps_coeff)
    return H
end

# - - - - - - - - - - - - -
# Fermionic Hamiltonians
# - - - - - - - - - - - - -
"""
   HELPERS
 The following functions are designed to perform the Jordan-Wigner mapping.
"""
# --- helpers for JWmapping ---
@inline ubit(i::Int) = UInt128(1) << (i-1)                         # bit at site i (1-based)
@inline umask_lt(i::Int) = i==1 ? UInt128(0) : (ubit(i) - UInt128(1))  # bits < i
@inline umask_le(i::Int) = umask_lt(i) | ubit(i)                      # bits ≤ i

# --- JW mapping (original real ± form; your coeff() supplies +i on ZX sites) ---
function JWmapping(N; i::Int, j::Int)
    1 <= i <= N || throw(DimensionMismatch("site i=$i out of 1:$N"))
    1 <= j <= N || throw(DimensionMismatch("site j=$j out of 1:$N"))

    # X pieces with Z-strings
    ax = Pauli{N}(1, reinterpret(Int128, umask_lt(i)), reinterpret(Int128, ubit(i)))  # Z^{<i} X_i
    bx = Pauli{N}(1, reinterpret(Int128, umask_lt(j)), reinterpret(Int128, ubit(j)))  # Z^{<j} X_j

    # "Y" pieces = Z^{≤i} X_i, Z^{≤j} X_j  (no explicit im; your coeff() turns ZX into iY)
    ay = Pauli{N}(1, reinterpret(Int128, umask_le(i)), reinterpret(Int128, ubit(i)))
    by = Pauli{N}(1, reinterpret(Int128, umask_le(j)), reinterpret(Int128, ubit(j)))

    # c†_i = (X_i - Y_i)/2,  c_j = (X_j + Y_j)/2   in your convention
    c_dagg_i = 0.5 * (ax - ay)
    c_j      = 0.5 * (bx + by)

    return c_dagg_i * c_j
end

"""
 1D Fermi-Hubbard model 
Generate a 1D Fermi-Hubbard Hamiltonian (open boundaries, no PBC)
using JW mapping into Pauli operators.

Arguments:
- o::Pauli{N} : reference Pauli object
- L::Int       : number of sites
- t::Float64   : hopping amplitude
- U::Float64   : on-site interaction
- k::Int       : number of Trotter steps (can be used later for evolution)

Returns:
- generators::Vector{Pauli{N}}
- parameters::Vector{Float64}
"""

function hubbard_model_1D(L::Int64, t::Float64, U::Float64)
    
    N_total = 2 * L   # Total number of fermionic modes (spin up and down)
    H = PauliSum(N_total, Float64)

    # Hopping terms
    for i in 1:L-1
        # spin-up
        a_up = 2*i - 1
        b_up = 2*(i+1) - 1
        hopping_up = JWmapping(N_total, i=a_up, j=b_up) + JWmapping(N_total, i=b_up, j=a_up)

        # spin-down
        a_dn = 2*i
        b_dn = 2*(i+1)
        hopping_dn = JWmapping(N_total, i=a_dn, j=b_dn) + JWmapping(N_total, i=b_dn, j=a_dn)
        
        # Add both
        H += -t * (hopping_up + hopping_dn)
    end

    # On-site interaction terms``
    for i in 1:L
        a_up = 2*i - 1   # spin-up orbital index
        a_dn = 2*i       # spin-down orbital index
        interaction_term = U *JWmapping(N_total, i=a_up, j=a_up) * JWmapping(N_total, i=a_dn, j=a_dn)

        H += interaction_term
    end

    #Filter zero coefficients
    DBF.coeff_clip!(H)

    return H    
end


"""
    fermi_hubbard_2D(Lx, Ly, t, U; reverse_ordering=false)

Construct generators and parameters for the 2D spinful Hubbard model on Lx×Ly
(physical sites). Each physical site has two spin-orbitals (up, down), so
total qubits N must equal 2 * Lx * Ly.

Returns (generators::Vector{Pauli{N}}, parameters::Vector{Float64}).
"""
function fermi_hubbard_2D(Lx::Int, Ly::Int, t::Float64, U::Float64)
    Nsites = Lx * Ly
    N_total = 2 * Nsites   # Total number of fermionic modes (spin up and down)
    H = PauliSum(N_total, Float64)

    if 2 * Nsites != N_total
        throw(ArgumentError("Total qubits N must equal 2 * Lx * Ly. Got N=$N_total, Lx*Ly=$Nsites"))
    end

    up(j) = 2*j - 1
    dn(j) = 2*j
    linear_index(x,y) = (x - 1) * Ly + y   # x in 1:Lx, y in 1:Ly

    # small tolerance for dropping tiny coeffs
    eps_coeff = 1e-12

    # HOPPING: loop nearest-neighbour pairs once, add c_i^† c_j + c_j^† c_i (both spins)
    for x in 1:Lx, y in 1:Ly
        jsite = linear_index(x, y)
         # neighbor +x (right in x)
        if x < Lx
            isite = linear_index(x + 1, y)
            for spin in (up, dn)
                m = spin(jsite)   # mode index for j
                n = spin(isite)   # mode index for i
                term = JWmapping(N_total, i=m, j=n) + JWmapping(N_total, i=n, j=m)
                H += -t * term
            end
        end
         # neighbor +y (right in y)
         if y < Ly
            isite = linear_index(x, y + 1)
            for spin in (up, dn)
                m = spin(jsite)
                n = spin(isite)
                term = JWmapping(N_total, i=m, j=n) + JWmapping(N_total, i=n, j=m)
                H += -t * term
            end
        end
    end

    for i in 1:Nsites
        a_up = 2*i - 1   # spin-up orbital index
        a_dn = 2*i       # spin-down orbital index
        interaction_term = U *JWmapping(N_total, i=a_up, j=a_up) * JWmapping(N_total, i=a_dn, j=a_dn)

        H += interaction_term
    end

    # Filter zero coefficients
    DBF.coeff_clip!(H, thresh=eps_coeff)

    return H
end


function fermi_hubbard_2D_zigzag(Lx::Int, Ly::Int, t::Float64, U::Float64)
    Nsites = Lx * Ly
    N_total = 2 * Nsites   # Total number of fermionic modes (spin up and down)
    H = PauliSum(N_total, Float64)

    if 2 * Nsites != N_total
        throw(ArgumentError("Total qubits N must equal 2 * Lx * Ly. Got N=$N_total, Lx*Ly=$Nsites"))
    end

    up(j) = 2*j - 1
    dn(j) = 2*j
    # linear_index(x,y) = (x - 1) * Ly + y   # x in 1:Lx, y in 1:Ly

    linear_index(i, j) = isodd(j) ? (j - 1) * Lx + i : j * Lx - i + 1

    # small tolerance for dropping tiny coeffs
    eps_coeff = 1e-12

    # HOPPING: loop nearest-neighbour pairs once, add c_i^† c_j + c_j^† c_i (both spins)
    for y in 1:Ly, x in 1:Lx
        println(x, "  ", y)
        jsite = linear_index(x, y)
        display(jsite)
         # neighbor +x (right in x)
        if x < Lx
            isite = linear_index(x + 1, y)
            for spin in (up, dn)
                m = spin(jsite)   # mode index for j
                n = spin(isite)   # mode index for i
                term = JWmapping(N_total, i=m, j=n) + JWmapping(N_total, i=n, j=m)
                H += -t * term
            end
        end
         # neighbor +y (right in y)
         if y < Ly
            isite = linear_index(x, y + 1)
            for spin in (up, dn)
                m = spin(jsite)
                n = spin(isite)
                term = JWmapping(N_total, i=m, j=n) + JWmapping(N_total, i=n, j=m)
                H += -t * term
            end
        end
    end

    for i in 1:Nsites
        a_up = 2*i - 1   # spin-up orbital index
        a_dn = 2*i       # spin-down orbital index
        interaction_term = U *JWmapping(N_total, i=a_up, j=a_up) * JWmapping(N_total, i=a_dn, j=a_dn)

        H += interaction_term
    end

    # Filter zero coefficients
    DBF.coeff_clip!(H, thresh=eps_coeff)

    return H
end

# # Test Hubbard 1D
# H = hubbard_model_1D(2, 5.0, 2.0)
# display(H)
# println("Number of terms in Hubbard 1D Hamiltonian: ", length(H))

# # Test Hubbard 2D
# H = fermi_hubbard_2D(1, 2, 5.0, 2.0)
# display(H)
# println("Number of terms in Hubbard 2x1 Hamiltonian: ", length(H))

function heisenberg_central_spin(N, Jx, Jy, Jz; x=0, y=0, z=0, α=0, seed=1)
    # All spins coupled through site 1
    H = PauliSum(N, Float64)
    Random.seed!(seed)
    for i in 2:N
        ϵ = randn() * α
        H += (-2*Jx + ϵ) * Pauli(N, X=[1,i]) 
        H += (-2*Jy + ϵ) * Pauli(N, Y=[1,i]) 
        H += (-2*Jz + ϵ) * Pauli(N, Z=[1,i]) 
    end 
    for i in 1:N
        H += x * Pauli(N, X=[i])
        H += y * Pauli(N, Y=[i])
        H += z * Pauli(N, Z=[i])
    end 
    return H
end

function heisenberg_sparse(N, Jx, Jy, Jz, sparsity; x=0, y=0, z=0, seed=1, α=1)
    # All spins coupled through site 1
    Random.seed!(seed)
    H = PauliSum(N, Float64)
    for i in 1:N
        for j in i+1:N
            rand() < sparsity || continue
            coupling = randn() * α
            H += -2*Jx * Pauli(N, X=[i,j]) * coupling 
            H += -2*Jy * Pauli(N, Y=[i,j]) * coupling
            H += -2*Jz * Pauli(N, Z=[i,j]) * coupling
        end 
    end 
    for i in 1:N
        rand() < sparsity || continue
        H += x * Pauli(N, X=[i])
        H += y * Pauli(N, Y=[i])
        H += z * Pauli(N, Z=[i])
    end 
    return H
end

function graph_laplacian(O::PauliSum{N,T}) where {N,T}
    A = graph_adjacency(O) 
    
    L = -1*A
    for i in 1:N
        L[i,i] = sum(A[:,i])
    end
    return L 
end 

function graph_adjacency(O::PauliSum{N,T}) where {N,T}
    A = zeros(Float64, N, N)
    for (p,c) in O
        on = PauliOperators.get_on_bits(p.z|p.x)
        # @show string(p)
        # display(on)
        for i in 1:length(on)
            for j in i+1:length(on)
                ii = on[i]
                jj = on[j]
                A[ii,jj] += abs(c)
                A[jj,ii] += abs(c)
            end
        end
    end
    
    return A 
end 
"""
 - - - Fermi-Hubbard model 2D (snake/zizag ordering) - - -
 The following function constructs the Hamiltonian for the 2D Fermi-Hubbard model
    using a snake-like (zigzag) ordering of the lattice sites.
  Constucts the model on a Lx x Ly lattive of physical sites,
  each site has two spin-orbitals (up, down), so the total number of fermionic modes is 
  N_total = 2 * Lx * Ly.
  If 'snake_ordering' is true, the physical sites follow a row-wise "snake/zigzag" pattern.
  Returns a PauliSum representing the Hamiltonian.
"""
function fermi_hubbard_2D_snake(Lx::Int, Ly::Int, t::Float64, U::Float64; snake_ordering::Bool=false)
    Nsites  = Lx * Ly
    N_total = 2 * Nsites
    H = PauliSum(N_total, Float64)

    # Spin-orbital indices for site j (1-based site indexing)
    up(j) = 2*j - 1
    dn(j) = 2*j

    # Map (row x, col y) -> site index j in [1, Nsites]
    @inline function site_index(x::Int, y::Int)
        if !snake_ordering
            # Row-major
            return (x - 1) * Ly + y
        else
            # Snake (zigzag) row-major:
            # odd rows (x=1,3,...) go left->right
            # even rows (x=2,4,...) go right->left
            if isodd(x)
                return (x - 1) * Ly + y
            else
                return x * Ly - (y - 1)
            end
        end
    end

    # small tolerance for dropping tiny coeffs
    eps_coeff = 1e-12

    # HOPPING: nearest-neighbor pairs (right and down) — add h.c.; both spins
    for x in 1:Lx, y in 1:Ly
        jsite = site_index(x, y)

        # neighbor in +x (next row)
        if x < Lx
            isite = site_index(x + 1, y)
            for spin in (up, dn)
                m = spin(jsite)
                n = spin(isite)
                term = JWmapping(N_total, i=m, j=n) + JWmapping(N_total, i=n, j=m)
                H += -t * term
            end
        end

        # neighbor in +y (next column)
        if y < Ly
            isite = site_index(x, y + 1)
            for spin in (up, dn)
                m = spin(jsite)
                n = spin(isite)
                term = JWmapping(N_total, i=m, j=n) + JWmapping(N_total, i=n, j=m)
                H += -t * term
            end
        end
    end

    # On-site interaction U * n_up * n_dn
    for i in 1:Nsites
        a_up = 2*i - 1
        a_dn = 2*i
        interaction_term = U * JWmapping(N_total, i=a_up, j=a_up) *
                               JWmapping(N_total, i=a_dn, j=a_dn)
        H += interaction_term
    end

    DBF.coeff_clip!(H, thresh=eps_coeff)
    return H
end