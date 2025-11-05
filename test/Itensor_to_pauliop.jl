using ITensors, ITensorMPS
using PauliOperators
using DBF
using Statistics

# Helper: construct bond in canonical (i<j) order
function _pushbond!(bonds::Vector{LatticeBond}, i::Int, j::Int, xi::Int, yi::Int, xj::Int, yj::Int)
    if i == j
        return
    end
    if i < j
        push!(bonds, LatticeBond(i, j, xi, yi, xj, yj))
    else
        push!(bonds, LatticeBond(j, i, xj, yj, xi, yi))
    end
end

"""
build_square_lattice(Nx, Ny; layout=:standard, order=:row, xperiodic=false, yperiodic=false)

Returns a Vector{LatticeBond} for a Nx-by-Ny square lattice.

Arguments:
- Nx, Ny : positive integers grid dimensions
- layout : :standard or :zigzag (default :standard)
- order  : :row or :col — which axis is the fast index in the 1D flattening
           :row -> n = (y-1)*Nx + x  (row-major)
           :col -> n = (x-1)*Ny + y  (column-major)
- xperiodic, yperiodic : booleans for periodic boundary in x or y

Returns: (bonds::Vector{LatticeBond}, idx::Function)
- bonds : canonical undirected bond list
- idx(site_x, site_y) -> linear index (1..Nx*Ny) according to chosen order
"""
function build_square_lattice(Nx::Int, Ny::Int; layout::Symbol=:standard,
                              order::Symbol=:row, xperiodic::Bool=false,
                              yperiodic::Bool=false)

    @assert Nx >= 1 && Ny >= 1 "Nx and Ny must be >= 1"
    @assert layout in (:standard, :zigzag) "layout must be :standard or :zigzag"
    @assert order in (:row, :col) "order must be :row or :col"

    # index mapping functions for row- or column-major flattening
    if order == :row
        idx = (x,y) -> (y-1)*Nx + x
        idx_zigzag = (x,y) -> begin
            if isodd(y)
                return (y-1)*Nx + x
            else
                return (y-1)*Nx + (Nx - x + 1)
            end
        end
    else # :col
        idx = (x,y) -> (x-1)*Ny + y
        idx_zigzag = (x,y) -> begin
            if isodd(x)
                return (x-1)*Ny + y
            else
                return (x-1)*Ny + (Ny - y + 1)
            end
        end
    end

    bonds = Vector{LatticeBond}()

    # Determine the bond order for the lattice:
    # If layout==:standard, bond order runs in a standard grid fashion.
    # If layout==:zigzag, bond order is created in a zigzag pattern.
    
    # choose which axis is fast (inner loop) and which is slow (outer)
    if order == :row
        # fast = x, slow = y
        fast_range = 1:Nx
        slow_range = 1:Ny
        fast_axis = :x
    else
        # fast = y, slow = x
        fast_range = 1:Ny
        slow_range = 1:Nx
        fast_axis = :y
    end

    for slow in slow_range
        # compute direction for this slow-line (for zigzag)
        forward = true
        if layout == :zigzag
            # alternate direction: even slow index -> reverse (or choose pattern)
            # Use parity of slow to alternate: odd -> forward, even -> backward
            forward = isodd(slow)
        end

        fiter = forward ? fast_range : reverse(fast_range)
        for fast in fiter
            # map back to (x,y)
            if order == :row
                x, y = fast, slow
            else
                x, y = slow, fast
            end

            if layout == :zigzag
                n = idx_zigzag(x, y)
            else
                n = idx(x, y)
            end
            

            # neighbor in fast direction (within same slow-line)
            if forward
                # attempt link to next fast (forward)
                if fast < last(fast_range)
                    f2 = fast + 1
                    if order == :row
                        x2, y2 = f2, slow
                    else
                        x2, y2 = slow, f2
                    end
                    if layout == :zigzag
                        n2 = idx_zigzag(x2, y2)
                    else
                        n2 = idx(x2, y2)
                    end
                    
                    _pushbond!(bonds, n, n2, x, y, x2, y2)
                elseif (layout == :zigzag) && (forward) && false
                    # no-op: we don't automatically wrap along fast axis unless periodicities enabled
                end
            else
                # reversed direction, attempt link to previous fast
                if fast > first(fast_range)
                    f2 = fast - 1
                    if order == :row
                        x2, y2 = f2, slow
                    else
                        x2, y2 = slow, f2
                    end
                    if layout == :zigzag
                        n2 = idx_zigzag(x2, y2)
                    else
                        n2 = idx(x2, y2)
                    end
                    
                    _pushbond!(bonds, n, n2, x, y, x2, y2)
                end
            end

            # neighbor in slow direction (connect to next slow line, same fast position)
            # compute target slow'
            if slow < last(slow_range)
                slow2 = slow + 1
                if order == :row
                    x2, y2 = x, slow2
                else
                    x2, y2 = slow2, y
                end
                if layout == :zigzag
                    n2 = idx_zigzag(x2, y2)
                else
                    n2 = idx(x2, y2)
                end

                _pushbond!(bonds, n, n2, x, y, x2, y2)
            elseif slow == last(slow_range) && ( (order==:row && yperiodic) || (order==:col && xperiodic) )
                # wrap periodic in slow direction:
                slow2 = first(slow_range)
                if order == :row
                    x2, y2 = x, slow2
                else
                    x2, y2 = slow2, y
                end
                if layout == :zigzag
                    n2 = idx_zigzag(x2, y2)
                else
                    n2 = idx(x2, y2)
                end
                _pushbond!(bonds, n, n2, x, y, x2, y2)
            end

            # horizontal / fast-direction periodic wrap if enabled and at boundary
            # (this handles periodicity across fast axis boundaries)
            if forward && (fast == last(fast_range))
                # at end of fast axis in forward direction
                if (order == :row && xperiodic) || (order == :col && yperiodic)
                    # wrap to first fast in same slow
                    f2 = first(fast_range)
                    if order == :row
                        x2, y2 = f2, slow
                    else
                        x2, y2 = slow, f2
                    end
                    if layout == :zigzag
                        n2 = idx_zigzag(x2, y2)
                    else
                        n2 = idx(x2, y2)
                    end
                    _pushbond!(bonds, n, n2, x, y, x2, y2)
                end
            elseif (!forward) && (fast == first(fast_range))
                # at start in reversed direction and periodic desired
                if (order == :row && xperiodic) || (order == :col && yperiodic)
                    f2 = last(fast_range)
                    if order == :row
                        x2, y2 = f2, slow
                    else
                        x2, y2 = slow, f2
                    end
                    if layout == :zigzag
                        n2 = idx_zigzag(x2, y2)
                    else
                        n2 = idx(x2, y2)
                    end
                    _pushbond!(bonds, n, n2, x, y, x2, y2)
                end
            end
        end
    end

    # sanity check: expected undirected bond count (no double-count)
    expected = (Nx-1)*Ny + Nx*(Ny-1) + (xperiodic ? Ny : 0) + (yperiodic ? Nx : 0)

    return bonds#, idx
end


"""
    fermi_hubbard_from_lattice(lattice, t, U; eps_coeff=1e-12)

Build a 2D Fermi-Hubbard Hamiltonian (PauliSum) from a lattice object.

Arguments
- `lattice` : an iterable of bond objects (each bond should expose `.s1` and `.s2` site identifiers).
- `t`       : hopping amplitude (real).
- `U`       : on-site interaction strength (real).

Keyword
- `eps_coeff` : threshold for coefficient clipping (default 1e-12).

Returns
- `H::PauliSum` : the Hamiltonian in PauliSum form using JW mapping.
"""
function fermi_hubbard_from_lattice(Lx, Ly, t, U)

    lattice = build_square_lattice(Lx, Ly; layout = :zigzag, order = :row,
                                   xperiodic = false, yperiodic = false)
    N = Lx * Ly
    N_total = 2 * N  # Total number of spin-orbitals
    H = PauliOperators.PauliSum(N_total, Float64)

    # Hopping terms
    for b in lattice
        s1_up = 2 * (b.s1 - 1) + 1  # Up spin site index
        s1_dn = 2 * (b.s1 - 1) + 2  # Down spin site index
        s2_up = 2 * (b.s2 - 1) + 1  # Up spin site index
        s2_dn = 2 * (b.s2 - 1) + 2  # Down spin site index
        # Up spin hopping
        term = DBF.JWmapping(N_total, i=s1_up, j=s2_up) + DBF.JWmapping(N_total, i=s2_up, j=s1_up)
        H += -t * term
        # Down spin hopping
        term = DBF.JWmapping(N_total, i=s1_dn, j=s2_dn) + DBF.JWmapping(N_total, i=s2_dn, j=s1_dn)
        H += -t * term
    end
    # On-site interaction terms
    for n in 1:N
        up_index = 2 * (n - 1) + 1
        dn_index = 2 * (n - 1) + 2
        term = DBF.JWmapping(N_total, i=up_index, j=up_index) * DBF.JWmapping(N_total, i=dn_index, j=dn_index)
        H += U * term
    end

    DBF.coeff_clip!(H)
    return H

end

function main(; Nx = 2 , Ny = 2, U = 2.0, t = 1.0,conserve_qns_ = true)
    N = Nx * Ny

    #Create an array of `N` physical site indices of type `electron` with quantum number conservation
    sites = siteinds("Electron", N; conserve_qns = conserve_qns_)
    # println("sites:")
    # println(sites)
    # println(typeof(sites))

    lattice = build_square_lattice(Nx, Ny; layout = :zigzag, order = :col,
                                   xperiodic = false, yperiodic = false)
    println("LATTICE:")
    # display(lattice)

    os = OpSum()
    # Hopping terms -> bonds in the lattice define Hopping terms as:
    for b in lattice
        println(b.s1, " -- ", b.s2)
        os -= t, "Cdagup", b.s1, "Cup", b.s2 #up spin
        os -= t, "Cdagup", b.s2, "Cup", b.s1 #up spin
        os -= t, "Cdagdn", b.s1, "Cdn", b.s2 #down spin
        os -= t, "Cdagdn", b.s2, "Cdn", b.s1 #down spin
    end

    # On-site interaction terms
    for n in 1:N
        os += U, "Nupdn", n
    end

    println("os:")
    # display(os)

    H_itensor = MPO(os, sites)
    println("Hamiltonian in Matrix product operator (MPO) format:") 

    # Half filling
    state = [isodd(n) ? "Up" : "Dn" for n in 1:N]
    println("state:")
    println(state)

    # Initialize wavefunction to a random MPS
    # of bond-dimension 10 with same quantum
    # numbers as `state`
    psi0 = random_mps(sites, state)

    H_pauli = fermi_hubbard_from_lattice(Nx, Ny, t, U)
    # println("Hamiltonian from lattice in PauliSum format:")
  

    H = DBF.fermi_hubbard_2D_zigzag(Nx, Ny, t, U)
    # println("Hamiltonian in PauliSum format from DBF.fermi_hubbard_2D:")
    # display(H)


    return H_itensor, psi0,H_pauli
end

main()

lattice = build_square_lattice(3, 3; layout = :zigzag, order = :col,
                               xperiodic = false, yperiodic = false)

println("LATTICE 3x3 zigzag row-major:")
display(lattice)
