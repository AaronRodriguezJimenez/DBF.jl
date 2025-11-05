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