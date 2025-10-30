using DBF
using PauliOperators
using Printf
using Random
using LinearAlgebra
using JLD2


# create a trivial basis for N=8 (all 256 bitstrings)
basis = [collect(digits(i, base=2, pad=8)) for i in 0:255]      # digits gives least-significant-first
basis = [reverse(b) for b in basis]  # now bit 1 corresponds to leftmost

println("Basis:")
for b in basis
    println(b)
end
println("Total basis size: ", length(basis))

# pick a random half-filled Sz=0 state from the basis
ket, occ = DBF.particle_ket(8, 4, 0.0; mode=:subspace, basis=basis, rng=MersenneTwister(1234))
println("Random ket at a subspace with N=4 and Sz=0 from basis:")
display(ket)

# Get The Neel state at half-filling (N=8, Nparticles=4, Sz=0) (standard 2D Lattice)
ket, occ = DBF.particle_ket(8, 4, 0.0; mode=:Neel, flavor=:Aup)
println("Neel state at half-filling (flavor :Aup):")
display(ket)

# Result from coupling map
#coupling_map = [[1,2],[3,4],[5,6],[7,8]]  #2D lattice pattern
coupling_map = [[1,2],[3,4],[7,8],[5,6]]  #zig-zag lattice =1D pattern
ket, occ = DBF.particle_ket(8, 4, 0, coupling=coupling_map; mode=:coupling)
println("State from coupling map:")
display(ket)

coupling_map_3d = [[1,2],[3,4],[5,6],[7,8],[9,10],[11,12],[13,14],[15,16],[17,18]]  #3D lattice pattern
ket, occ = DBF.particle_ket(18, 9, 0.5, coupling=coupling_map_3d; mode=:coupling)
println("State from coupling map 3x3 square lattice:")
display(ket)

coupling_map_3d = [[1,2],[3,4],[5,6],[11,12],[9,10],[7,8],[13,14],[15,16],[17,18]]  #3D lattice pattern
ket, occ = DBF.particle_ket(18, 9, 0.5, coupling=coupling_map_3d; mode=:coupling)
println("State from coupling map 3x3 zig-zag lattice:")
display(ket)